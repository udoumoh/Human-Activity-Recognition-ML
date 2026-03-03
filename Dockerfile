# PAMAP2 Human Activity Recognition — Medallion Pipeline
# Base image: Python 3.11 slim with OpenJDK 17 (required by PySpark)

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Python dependencies ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────
COPY . .

# ── Spark local mode settings ─────────────────────────────────
ENV PYSPARK_PYTHON=python3
ENV SPARK_LOCAL_IP=127.0.0.1

# ── Default command: run the full pipeline end-to-end ────────
CMD ["bash", "-c", "\
    python -m bronze.ingestion && \
    python -m silver.cleaning && \
    python -m silver.preprocessing && \
    python -m gold.feature_engineering && \
    python -m gold.training && \
    python -m gold.evaluation && \
    python -m gold.extended_evaluation && \
    python -m gold.tableau_export \
"]
