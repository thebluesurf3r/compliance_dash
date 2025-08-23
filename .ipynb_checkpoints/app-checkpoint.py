import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import pyspark
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Initialize Spark
spark = SparkSession.builder \
    .appName("AuditComplianceStreamlitApp") \
    .getOrCreate()

# Feature engineering functions
def create_temporal_features(df):
    return (
        df.withColumn("year", F.year("date"))
          .withColumn("month", F.month("date"))
          .withColumn("day_of_week", F.dayofweek("date"))
          .withColumn("quarter", F.quarter("date"))
          .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
    )

def create_derived_features(df):
    df = df.withColumn(
        "compliance_rate",
        F.when(F.col("count_of_audit") > 0,
               (F.col("count_of_compliant_audit") / F.col("count_of_audit")).cast("double"))
         .otherwise(0.0)
    ).withColumn(
        "audit_density",
        F.when(F.col("departures") > 0,
               (F.col("count_of_audit") / F.col("departures")).cast("double"))
         .otherwise(0.0)
    ).withColumn(
        "compliance_per_departure",
        F.when(F.col("departures") > 0,
               (F.col("count_of_compliant_audit") / F.col("departures")).cast("double"))
         .otherwise(0.0)
    )
    return df

def create_lag_features(df):
    windowSpec = Window.partitionBy("airport", "reference_number").orderBy("date")
    df = df.withColumn("prev_compliance_rate", F.lag("compliance_rate", 1).over(windowSpec))
    return df.fillna({"prev_compliance_rate": 0.0})

def handle_nan_inf(df):
    return df.dropna()

# Load data
@st.cache_data
def load_data(path: str):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", F.to_date("date", "yyyy-MM-dd"))
    return df

# Main Streamlit app
st.title("Audit Compliance Dashboard")

data_path = st.text_input("Enter path to CSV data file:", "data/audit_data.csv")
if os.path.exists(data_path):
    raw_df = load_data(data_path)

    # Feature engineering pipeline
    df = create_temporal_features(raw_df)
    df = create_derived_features(df)
    df = create_lag_features(df)
    df = handle_nan_inf(df)

    # Convert to Pandas for plotting
    pdf = df.toPandas()

    # Sidebar filters
    airports = sorted(pdf['airport'].unique())
    selected_airports = st.sidebar.multiselect("Select Airport(s):", airports, default=airports)

    date_slider = st.sidebar.date_input(
        "Date Range:",
        [pdf['date'].min(), pdf['date'].max()]
    )

    mask = (
        pdf['airport'].isin(selected_airports) &
        (pdf['date'] >= pd.to_datetime(date_slider[0])) &
        (pdf['date'] <= pd.to_datetime(date_slider[1]))
    )
    filtered = pdf.loc[mask]

    # Visualization 1: Compliance Rate Over Time
    st.subheader("Compliance Rate Over Time")
    fig1 = px.line(
        filtered,
        x="date", y="compliance_rate",
        color="airport",
        title="Daily Compliance Rate by Airport"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Visualization 2: Audit Density Distribution
    st.subheader("Audit Density Distribution")
    fig2 = px.histogram(
        filtered,
        x="audit_density",
        color="airport",
        nbins=30,
        title="Distribution of Audit Density"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Visualization 3: Compliance Rate vs. Previous Day
    st.subheader("Compliance Rate vs. Previous Day")
    fig3 = px.scatter(
        filtered,
        x="prev_compliance_rate", y="compliance_rate",
        color="airport",
        trendline="ols",
        title="Day-over-Day Compliance Rate"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Key Metrics
    st.subheader("Key Metrics")
    avg_comp = filtered['compliance_rate'].mean()
    avg_density = filtered['audit_density'].mean()
    st.metric("Avg. Compliance Rate", f"{avg_comp:.2%}")
    st.metric("Avg. Audit Density", f"{avg_density:.4f}")
else:
    st.error(f"File not found: {data_path}")
