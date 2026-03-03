"""
Pipeline Stage Profiling Utilities (Part 5).

Lightweight profiling context managers and decorators for classifying the
computational bottleneck of each Medallion Architecture stage.

Bottleneck Classification
--------------------------
Stage               | Bottleneck Type       | Rationale
--------------------|----------------------|------------------------------------------
Bronze ingestion    | I/O bound             | HDFS/local disk read of 1.61 GB .dat files;
                    |                       | CPU idle while waiting for disk throughput.
Silver cleaning     | Shuffle bound         | Broadcast join resolves in memory, but
                    |                       | repartition + null-filter scan touches all
                    |                       | 2.7 M rows across executors.
Gold aggregation    | Shuffle heavy         | groupBy(subject, activity, window_id) hashes
                    |                       | rows to 162 groups; partial aggregation
                    |                       | reduces shuffle volume by ~80%.
Spark MLlib         | Distributed compute   | MLP gradient descent: matrix multiply at each
                    |                       | layer; Spark broadcasts model params per iter.
CNN training (CPU)  | Compute bound         | Conv1d GEMM operations dominate; single-
                    |                       | threaded Python overhead is negligible vs MAC.
CNN training (GPU)  | Compute bound (GPU)   | CUDA parallelises Conv1d across batch × time
                    |                       | dimensions simultaneously; memory transfer
                    |                       | (host→device) is one-time cost at epoch 0.

Usage
-----
As a context manager:

    from src.profiling import StageProfiler, BottleneckType

    with StageProfiler("Bronze ingestion", BottleneckType.IO) as p:
        df = spark.read.parquet(BRONZE_OUTPUT)
    # → logs: [PROFILE] Bronze ingestion | I/O bound | 12.4 s

As a decorator:

    from src.profiling import profile_stage, BottleneckType

    @profile_stage("Silver cleaning", BottleneckType.SHUFFLE)
    def run():
        ...

Spark stage labelling (marks each Spark job in Spark UI):

    from src.profiling import set_spark_job_description
    set_spark_job_description(spark, "Gold aggregation — groupBy window_id")
"""

import time
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)


class BottleneckType:
    """String constants for bottleneck classification labels."""
    IO              = "I/O bound"
    SHUFFLE         = "Shuffle bound"
    SHUFFLE_HEAVY   = "Shuffle heavy"
    DISTRIBUTED     = "Distributed compute bound"
    COMPUTE_CPU     = "Compute bound (CPU)"
    COMPUTE_GPU     = "Compute bound (GPU)"


STAGE_BOTTLENECK_MAP = {
    "Bronze ingestion":          BottleneckType.IO,
    "Silver cleaning":           BottleneckType.SHUFFLE,
    "Silver preprocessing":      BottleneckType.SHUFFLE,
    "Gold aggregation":          BottleneckType.SHUFFLE_HEAVY,
    "Gold sequence tensor":      BottleneckType.SHUFFLE,
    "Spark MLlib training":      BottleneckType.DISTRIBUTED,
    "CNN training CPU":          BottleneckType.COMPUTE_CPU,
    "CNN training GPU":          BottleneckType.COMPUTE_GPU,
}

_profile_records: list = []


def get_profile_records() -> list:
    """Return all profiling records collected during the session."""
    return list(_profile_records)


def reset_profile_records() -> None:
    """Clear all accumulated profiling records."""
    _profile_records.clear()


def log_stage_profile(stage_name: str, elapsed_s: float,
                      bottleneck_type: str) -> None:
    """
    Emit a structured INFO log line for a completed pipeline stage.

    Format:
        [PROFILE] <stage_name> | <bottleneck_type> | <elapsed>s

    Also appends to the module-level _profile_records list so a post-run
    summary table can be constructed without re-parsing log files.
    """
    record = {
        "stage":      stage_name,
        "bottleneck": bottleneck_type,
        "elapsed_s":  round(elapsed_s, 2),
    }
    _profile_records.append(record)
    log.info(
        f"[PROFILE] {stage_name:<30s} | {bottleneck_type:<28s} | {elapsed_s:>8.2f}s"
    )


@contextmanager
def StageProfiler(stage_name: str, bottleneck_type: str = "Unknown"):
    """
    Context manager that times a code block and logs its bottleneck class.

    Parameters
    ----------
    stage_name     : human-readable name (matches STAGE_BOTTLENECK_MAP keys)
    bottleneck_type: BottleneckType constant (auto-resolved from map if omitted
                     and stage_name is a known key)

    Example
    -------
        with StageProfiler("Bronze ingestion", BottleneckType.IO):
            run_bronze_ingestion(spark)
    """
    resolved = STAGE_BOTTLENECK_MAP.get(stage_name, bottleneck_type)
    t0 = time.time()
    try:
        yield
    finally:
        log_stage_profile(stage_name, time.time() - t0, resolved)


def profile_stage(stage_name: str, bottleneck_type: str = None):
    """
    Decorator that wraps a function with StageProfiler timing.

    Parameters
    ----------
    stage_name     : human-readable stage label
    bottleneck_type: BottleneckType constant or None (auto-resolved)

    Example
    -------
        @profile_stage("Gold aggregation", BottleneckType.SHUFFLE_HEAVY)
        def run():
            ...
    """
    resolved = bottleneck_type or STAGE_BOTTLENECK_MAP.get(stage_name, "Unknown")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            log_stage_profile(stage_name, time.time() - t0, resolved)
            return result
        return wrapper
    return decorator


def set_spark_job_description(spark, description: str) -> None:
    """
    Set the Spark UI job group description for the next submitted job.

    Labels the job in the Spark Web UI (default: localhost:4040) so that
    each pipeline stage is identifiable by name in the DAG view.

    Parameters
    ----------
    spark       : active SparkSession
    description : human-readable label (shown in Spark UI "Description" column)
    """
    spark.sparkContext.setJobDescription(description)


def print_profile_summary() -> None:
    """
    Print a formatted summary of all profiled stages to the console.

    Call at the end of a pipeline run to produce a bottleneck report.
    """
    records = get_profile_records()
    if not records:
        log.warning("No profiling records — wrap stages with StageProfiler first.")
        return

    total = sum(r["elapsed_s"] for r in records)
    log.info("\n" + "=" * 70)
    log.info("PIPELINE BOTTLENECK PROFILE SUMMARY")
    log.info("=" * 70)
    log.info(f"  {'Stage':<30}  {'Bottleneck':<28}  {'Time (s)':>8}  {'% total':>8}")
    log.info("  " + "-" * 66)
    for r in records:
        pct = 100 * r["elapsed_s"] / total if total > 0 else 0
        log.info(
            f"  {r['stage']:<30}  {r['bottleneck']:<28}  "
            f"{r['elapsed_s']:>8.2f}  {pct:>7.1f}%"
        )
    log.info("  " + "-" * 66)
    log.info(f"  {'TOTAL':<30}  {'':28}  {total:>8.2f}  {'100.0%':>8}")
    log.info("=" * 70)
