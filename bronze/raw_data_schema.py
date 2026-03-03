"""
Bronze Layer — Raw Data Schema Definition for PAMAP2.

Defines the 54-column schema matching the PAMAP2 UCI dataset specification.
Used by the ingestion script to enforce typed parsing during distributed CSV
loading via Spark's DataFrame API.

Schema layout (per the PAMAP2 readme):
    Col  1       : timestamp (seconds, DoubleType)
    Col  2       : activityID (IntegerType)
    Col  3       : heart_rate (bpm, DoubleType)
    Col  4–20    : IMU_hand   (17 columns)
    Col 21–37    : IMU_chest  (17 columns)
    Col 38–54    : IMU_ankle  (17 columns)

Each IMU block contains 17 signals:
    temperature, acc_16g (xyz), acc_6g (xyz), gyro (xyz),
    mag (xyz), orientation quaternion (4 values)
"""

from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType,
)

# 17 signals per IMU body location
IMU_SUFFIXES = [
    "temperature",
    "acc_16g_x", "acc_16g_y", "acc_16g_z",
    "acc_6g_x",  "acc_6g_y",  "acc_6g_z",
    "gyro_x",    "gyro_y",    "gyro_z",
    "mag_x",     "mag_y",     "mag_z",
    "orient_0",  "orient_1",  "orient_2", "orient_3",
]


def build_imu_fields(location: str) -> list:
    """
    Generate 17 StructFields for one IMU placement.

    Parameters
    ----------
    location : str
        Body location identifier ('hand', 'chest', or 'ankle').

    Returns
    -------
    list[StructField]
        17 nullable DoubleType fields named '{location}_{suffix}'.
    """
    return [
        StructField(f"{location}_{s}", DoubleType(), True)
        for s in IMU_SUFFIXES
    ]


PAMAP2_SCHEMA = StructType(
    [
        StructField("timestamp",   DoubleType(),  False),
        StructField("activity_id", IntegerType(), False),
        StructField("heart_rate",  DoubleType(),  True),
    ]
    + build_imu_fields("hand")
    + build_imu_fields("chest")
    + build_imu_fields("ankle")
)

assert len(PAMAP2_SCHEMA.fields) == 54, (
    f"Expected 54 fields, got {len(PAMAP2_SCHEMA.fields)}"
)
