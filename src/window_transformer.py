"""
Custom PySpark ML Transformer for sliding-window feature extraction.

This module implements WindowFeatureTransformer, a domain-specific
Transformer that segments time-series sensor data into fixed-duration
windows and computes statistical aggregates (mean, std, min, max)
per window. It integrates directly with PySpark's Pipeline API.

Usage inside a Pipeline:
    from src.window_transformer import WindowFeatureTransformer

    wft = WindowFeatureTransformer(
        window_duration=5.0,
        sample_rate=100,
        timestamp_col="timestamp",
        group_cols=["subject_id", "activity_id"],
        min_window_fill=0.5,
    )

    pipeline = Pipeline(stages=[wft, assembler, scaler, classifier])
    model = pipeline.fit(train_df)
"""

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import (
    col, lit, floor, count,
    mean as F_mean,
    stddev as F_stddev,
    min as F_min,
    max as F_max,
)
from pyspark.sql import Window


class WindowFeatureTransformer(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Custom PySpark Transformer that performs sliding-window aggregation
    on time-series sensor data.

    For each group (e.g. subject + activity), rows are assigned to
    fixed-duration windows based on their timestamp. Within each window,
    statistical features (mean, std, min, max) are computed for every
    numeric sensor column.

    Parameters
    ----------
    window_duration : float
        Window length in seconds (default: 5.0).
    sample_rate : int
        Expected sampling rate in Hz, used to calculate the minimum
        number of samples per window (default: 100).
    timestamp_col : str
        Name of the timestamp column (default: "timestamp").
    group_cols : list[str]
        Columns to partition by before windowing. Windows never cross
        group boundaries (default: ["subject_id", "activity_id"]).
    min_window_fill : float
        Minimum fraction of expected samples a window must contain
        to be kept. Windows below this threshold are discarded
        (default: 0.5, i.e. at least 50% full).
    """

    window_duration = Param(
        Params._dummy(),
        "window_duration",
        "Window length in seconds",
        typeConverter=TypeConverters.toFloat,
    )

    sample_rate = Param(
        Params._dummy(),
        "sample_rate",
        "Expected sampling rate in Hz",
        typeConverter=TypeConverters.toInt,
    )

    timestamp_col = Param(
        Params._dummy(),
        "timestamp_col",
        "Name of the timestamp column",
        typeConverter=TypeConverters.toString,
    )

    group_cols = Param(
        Params._dummy(),
        "group_cols",
        "Columns to partition by before windowing",
        typeConverter=TypeConverters.toListString,
    )

    min_window_fill = Param(
        Params._dummy(),
        "min_window_fill",
        "Minimum fraction of expected samples per window (0-1)",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(
        self,
        window_duration=5.0,
        sample_rate=100,
        timestamp_col="timestamp",
        group_cols=None,
        min_window_fill=0.5,
    ):
        super().__init__()
        if group_cols is None:
            group_cols = ["subject_id", "activity_id"]
        self._setDefault(
            window_duration=5.0,
            sample_rate=100,
            timestamp_col="timestamp",
            group_cols=["subject_id", "activity_id"],
            min_window_fill=0.5,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def getWindowDuration(self):
        return self.getOrDefault(self.window_duration)

    def getSampleRate(self):
        return self.getOrDefault(self.sample_rate)

    def getTimestampCol(self):
        return self.getOrDefault(self.timestamp_col)

    def getGroupCols(self):
        return self.getOrDefault(self.group_cols)

    def getMinWindowFill(self):
        return self.getOrDefault(self.min_window_fill)

    def _identify_sensor_cols(self, df):
        """Return sorted list of DoubleType columns excluding metadata."""
        meta = set(self.getGroupCols()) | {self.getTimestampCol(), "session_type"}
        return sorted([
            c for c in df.columns
            if c not in meta
            and isinstance(df.schema[c].dataType, DoubleType)
        ])

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform raw time-series data into windowed feature rows.

        Steps:
        1. Assign each row to a window_id based on timestamp offset
           within each group.
        2. Aggregate per window: mean, std, min, max for each sensor.
        3. Filter out windows that are too sparse.
        4. Drop helper columns (window_id, sample_count).
        """
        win_dur = self.getWindowDuration()
        rate = self.getSampleRate()
        ts_col = self.getTimestampCol()
        grp_cols = self.getGroupCols()
        min_fill = self.getMinWindowFill()

        expected_samples = int(win_dur * rate)
        min_samples = int(expected_samples * min_fill)

        # -- Identify numeric sensor columns ----------------------
        sensor_cols = self._identify_sensor_cols(df)

        # -- Step 1: assign window IDs ----------------------------
        # Compute per-group minimum timestamp, then window_id =
        # floor((t - t_min) / window_duration)
        seg_window = Window.partitionBy(*grp_cols)
        df_win = (
            df
            .withColumn("_t0", F_min(col(ts_col)).over(seg_window))
            .withColumn(
                "window_id",
                floor((col(ts_col) - col("_t0")) / lit(win_dur)).cast("long"),
            )
            .drop("_t0")
        )

        # -- Step 2: build aggregation expressions ----------------
        agg_exprs = [count("*").alias("sample_count")]
        for c in sensor_cols:
            agg_exprs.extend([
                F_mean(col(c)).alias(f"{c}_mean"),
                F_stddev(col(c)).alias(f"{c}_std"),
                F_min(col(c)).alias(f"{c}_min"),
                F_max(col(c)).alias(f"{c}_max"),
            ])

        group_keys = list(grp_cols) + ["window_id"]
        df_agg = df_win.groupBy(*group_keys).agg(*agg_exprs)

        # -- Step 3: quality gate ---------------------------------
        df_filtered = df_agg.filter(col("sample_count") >= min_samples)

        # -- Step 4: drop helper columns --------------------------
        df_out = df_filtered.drop("window_id", "sample_count")

        return df_out
