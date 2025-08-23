import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

from datetime import datetime

import pyspark.sql.functions as F
import pyspark.sql.types as T

def create_temporal_features(df):
    # Extract core temporal features
    df = df.withColumn("year", F.year("date")) \
           .withColumn("month", F.month("date")) \
           .withColumn("day_of_week", F.dayofweek("date")) \
           .withColumn("quarter", F.quarter("date")) \
           .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
    return df


def create_derived_features(df):
    df = df.withColumn("compliance_rate",
                       F.when(F.col("count_of_audit") > 0,
                              (F.col("count_of_compliant_audit") / F.col("count_of_audit")).cast("double"))
                        .otherwise(0.0))
    df = df.withColumn("audit_density",
                       F.when(F.col("departures") > 0,
                              (F.col("count_of_audit") / F.col("departures")).cast("double"))
                        .otherwise(0.0))
    df = df.withColumn("compliance_per_departure",
                       F.when(F.col("departures") > 0,
                              (F.col("count_of_compliant_audit") / F.col("departures")).cast("double"))
                        .otherwise(0.0))
    return df

def create_lag_features(df):
    windowSpec = Window.partitionBy("airport", "reference_number").orderBy("date")
    df = df.withColumn("prev_compliance_rate", F.lag("compliance_rate", 1).over(windowSpec))
    df = df.fillna({"prev_compliance_rate": 0.0})
    return df

def handle_nan_inf(df):
    print(f"Original Dataframe: {df.count()}")
    df_processed = df.dropna()
    print(f"Updated Dataframe: {df_processed.count()}")
    # for col_name in input_cols:
    #     df = df.withColumn(
    #         col_name,
    #         F.when(~(F.isnan(F.col(col_name)) | F.isnull(F.col(col_name)) | F.col(col_name).isin([float('inf'), -float('inf')])),
    #                F.col(col_name))
    #          .otherwise(0.0)
    #     )
    return df_processed