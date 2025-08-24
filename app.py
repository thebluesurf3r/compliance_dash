import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ---------------------- Streamlit Config ----------------------
st.set_page_config(
    page_title="Audit Compliance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom Color Palette ------------------
CUSTOM_PALETTE = [
    "#6583fd", #0
    "#3960fb", #1
    "#0c39ed", #2
    "#2a08b5", #3
    "#051966", #4
    ]

CUSTOM_COLOR_SCALE = [
    [0.0, "#6583fd"],
    [0.16, "#3960fb"],
    [0.33, "#0c39ed"],
    [0.5, "#2a08b5"],
    [0.66, "#051966"],
]

# ---------------------- Global Plotly Theme -------------------
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = CUSTOM_PALETTE


# ---------------------- Custom CSS -----------------------------
st.markdown(
    f"""
    <style>
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
        color: #fff !important;
    }}

    /* Score cards */
    .score-card {{
        background: linear-gradient(145deg, {CUSTOM_PALETTE[2]}, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
        border-radius: 9px;
        padding: 18px;
        text-align: center;
        color: #ffffff !important; /* Force black text */
        box-shadow: 0 6px 12px rgba(0,0,0,0.6);
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        transition: transform 0.2s ease-in-out;
    }}
    .score-card:hover {{
        transform: translateY(-4px);
    }}
    .score-label {{
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 4px;
        display: block;
        color: ##ffffff !important; /* black labels */
    }}
    .score-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: ##ffffff !important; /* black values */
    }}

    /* Unchecked box background */
    div[role="checkbox"] > div:first-child {{
        border: 2px solid {CUSTOM_PALETTE[0]} !important;
        background-color: transparent !important;
        border-radius: 4px;
        width: 18px !important;
        height: 18px !important;
    }}

    /* Checked box background */
    div[role="checkbox"][aria-checked="true"] > div:first-child {{
        background-color: {CUSTOM_PALETTE[1]} !important;
        border-color: {CUSTOM_PALETTE[1]} !important;
    }}

    /* Tick mark color inside the box */
    div[role="checkbox"][aria-checked="true"] svg {{
        stroke: white !important;
    }}

    /* Checkbox label */
    label[data-testid="stMarkdownContainer"] p {{
        color: white !important;
        margin-left: 6px;
    }}

    /* Card Wrapper */
    .st-card {{
        background: linear-gradient(160deg, {CUSTOM_PALETTE[2]}, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
        border-radius: 12px;
        padding: 18px 22px;
        margin: 10px 5px;
        color: #fff;
        box-shadow: 0 6px 14px rgba(0,0,0,0.6);
        transition: all 0.25s ease-in-out;
    }}
    .st-card:hover {{
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 20px rgba(0,0,0,0.8);
    }}

    /* Section Headers */
    .section-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 12px;
    }}

    /* Key Insights Styling */
    .insights-card {{
        background: linear-gradient(145deg, #111, {CUSTOM_PALETTE[3]});
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        color: #f0f0f0;
    }}
    .insights-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.8);
    }}
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Data Processing ------------------------
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek + 1
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df['compliance_rate'] = np.where(
        df['count_of_audit'] > 0,
        df['count_of_compliant_audit'] / df['count_of_audit'],
        0.0
    )
    df['audit_density'] = np.where(
        df['departures'] > 0,
        df['count_of_audit'] / df['departures'],
        0.0
    )
    df['compliance_per_departure'] = np.where(
        df['departures'] > 0,
        df['count_of_compliant_audit'] / df['departures'],
        0.0
    )
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['airport', 'reference_number', 'date'])
    df['prev_compliance_rate'] = (
        df.groupby(['airport', 'reference_number'])['compliance_rate']
          .shift(1)
          .fillna(0.0)
    )
    return df


def handle_nan_inf(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = df.columns
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

def engineer_departure_std_bins(df: pd.DataFrame, col="departures"):
    mean, std = df[col].mean(), df[col].std()
    min_val, max_val = df[col].min(), df[col].max()

    # ensure bins are strictly increasing
    bins = [min_val - 1, max(mean - std, min_val), mean, mean + std, max_val]
    bins = sorted(set(bins))  # remove duplicates if mean==mean±std

    labels = ["Low", "Average", "Above Avg", "High"][:len(bins)-1]

    df[f"{col}_cat"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


@st.cache_data
def load_and_process(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df = create_temporal_features(df)
    df = create_derived_features(df)
    df = create_lag_features(df)
    df = handle_nan_inf(df, [
        'compliance_rate', 'audit_density',
        'compliance_per_departure', 'prev_compliance_rate'
    ])
    df = engineer_departure_std_bins(df)
    return df



# ---------------------- Visualization Functions ----------------
import plotly.express as px

def plot_record_count_by_reference(filtered: pd.DataFrame):
    # Aggregate counts correctly
    count_ref = (
        filtered['reference_number']
        .value_counts()
        .reset_index()
    )
    count_ref.columns = ['reference_number', 'count']  # overwrite cleanly
    count_ref = count_ref.sort_values('count', ascending=False)

    # Base bar chart
    fig = px.bar(
        count_ref,
        x='reference_number',
        y='count',
        color='count',  # Gradient color by count
        color_continuous_scale=CUSTOM_PALETTE,
        title="Count of Audits per Audit Group"
    )

    # Add data labels
    fig.update_traces(
        text=count_ref['count'],
        textposition="outside",
        marker_line=dict(width=0.8, color="black")
    )

    # Add reference line for average
    avg_val = count_ref['count'].mean()
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red"
    )

    # Add annotation manually at center
    fig.add_annotation(
        x=0.5,  # middle of the x-axis (in paper coords)
        y=avg_val,
        xref="paper",  # relative to whole plot width
        yref="y",
        text=f"Avg: {avg_val:.1f}",
        showarrow=False,
        font=dict(color="red", size=12, weight="bold"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )


    # Update layout
    fig.update_layout(
        xaxis=dict(title="Reference Number", tickangle=-45, showgrid=False),
        yaxis=dict(title="Record Count", zeroline=False, showgrid=True),
        bargap=0.25,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )

    return fig


def plot_boxplot_distribution(filtered: pd.DataFrame, dimension: str):
    x_col = "airline" if dimension == "airline" else "reference_number"
    color_col = "reference_number" if dimension == "airline" else "airline"
    title = f"Compliance Rate Distribution by {x_col.capitalize()}"

    fig = px.box(
        filtered,
        x=x_col, y="compliance_rate",
        color=color_col, points="outliers",
        boxmode="overlay", title=title
    )
    fig.update_layout(
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.8)
    )
    return fig


def plot_parallel_categories(filtered: pd.DataFrame):
    pcat_cols = ['reference_number', 'quarter', 'year', 'departures_cat']
    if 'airline' in filtered.columns:
        pcat_cols.insert(1, 'airline')

    filtered = filtered.copy()
    filtered['quarter'] = filtered['quarter'].astype(str)

    # ✅ Define custom labels
    label_map = {
        "reference_number": "Audit Group",
        "airline": "Airline",
        "quarter": "Quarter",
        "year": "Year",
        "departures_cat": "Departure Traffic"
    }

    # Build dimensions with manual labels
    dimensions = [
        dict(values=filtered[col], label=label_map.get(col, col))
        for col in pcat_cols
    ]

    # Handle airline category codes safely
    if "airline" in filtered.columns:
        categories = filtered["airline"].astype(str).unique()
        cat_to_num = {cat: i for i, cat in enumerate(categories)}
        filtered["_cat_code"] = filtered["airline"].map(cat_to_num)
        color_col = "_cat_code"
    else:
        filtered["_cat_code"] = 0
        color_col = "_cat_code"

    # Build figure
    fig = go.Figure(data=[go.Parcats(
        dimensions=dimensions,
        line=dict(
            color=filtered[color_col],
            colorscale=CUSTOM_COLOR_SCALE,
            shape="hspline"
        )
    )])

    fig.update_layout(title="Parallel Categories Distribution")
    return fig



def plot_heatmap_table(filtered: pd.DataFrame):
    # Aggregate per reference_number
    agg_df = (
        filtered.groupby("reference_number")
        .agg({
            "count_of_audit": "sum",
            "count_of_compliant_audit": "sum",
            "compliance_rate": "mean"
        })
        .reset_index()
    )

    # Add total row (weighted compliance for accuracy)
    total_row = pd.DataFrame({
        "reference_number": ["Total"],
        "count_of_audit": [agg_df["count_of_audit"].sum()],
        "count_of_compliant_audit": [agg_df["count_of_compliant_audit"].sum()],
        "compliance_rate": [
            agg_df["count_of_compliant_audit"].sum() / agg_df["count_of_audit"].sum(),
        ]  # weighted compliance rate
    })

    agg_df = pd.concat([agg_df, total_row], ignore_index=True)

    # Build row colors: darker shade for Total row
    row_colors = []
    for i, ref in enumerate(agg_df["reference_number"]):
        if ref == "Total":
            row_colors.append(CUSTOM_PALETTE[4])
        else:
            row_colors.append(CUSTOM_PALETTE[3])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Reference Number",
                        "Count of Audits",
                        "Count of Compliant Audits",
                        "Avg. Compliance Rate"
                    ],
                    fill_color=CUSTOM_PALETTE[4],
                    align="center",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=[
                        agg_df["reference_number"],
                        agg_df["count_of_audit"],
                        agg_df["count_of_compliant_audit"],
                        (agg_df["compliance_rate"] * 100).round(2).astype(str) + "%"  # % format
                    ],
                    fill_color=[
                        row_colors,                      # Reference column (white vs gray for Total)
                        row_colors,                      # Keep counts aligned with row shading
                        row_colors,
                        row_colors
                    ],
                    align="center",
                    font=dict(color="black", size=11)
                )
            )
        ]
    )

    fig.update_layout(title="Aggregated Metrics by Audit Group")
    return fig

def plot_airline_table(filtered: pd.DataFrame):
    # Aggregate per airline
    agg_df = (
        filtered.groupby("airline")
        .agg({
            "count_of_audit": "sum",
            "count_of_compliant_audit": "sum",
            "compliance_rate": "mean"
        })
        .reset_index()
    )

    # Add total row (weighted compliance for accuracy)
    total_row = pd.DataFrame({
        "airline": ["Total"],
        "count_of_audit": [agg_df["count_of_audit"].sum()],
        "count_of_compliant_audit": [agg_df["count_of_compliant_audit"].sum()],
        "compliance_rate": [
            agg_df["count_of_compliant_audit"].sum() / agg_df["count_of_audit"].sum()
        ]
    })

    agg_df = pd.concat([agg_df, total_row], ignore_index=True)

    # Row colors (darker shade for Total row)
    row_colors = []
    for ref in agg_df["airline"]:
        if ref == "Total":
            row_colors.append(CUSTOM_PALETTE[1])  # darker
        else:
            row_colors.append(CUSTOM_PALETTE[3])  # lighter

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Airline",
                        "Count of Audits",
                        "Count of Compliant Audits",
                        "Avg. Compliance Rate"
                    ],
                    fill_color=CUSTOM_PALETTE[0],
                    align="center",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=[
                        agg_df["airline"],
                        agg_df["count_of_audit"],
                        agg_df["count_of_compliant_audit"],
                        (agg_df["compliance_rate"] * 100).round(2).astype(str) + "%"
                    ],
                    fill_color=[row_colors, row_colors, row_colors, row_colors],
                    align="center",
                    font=dict(color="black", size=11)
                )
            )
        ]
    )

    fig.update_layout(title="Aggregated Metrics by Airline")
    return fig



def styled_metric(label, value):
    return f"""
    <div class="score-card">
        <span class="score-label">{label}</span>
        <span class="score-value">{value}</span>
    </div>
    """


# ---------------------- Main App -------------------------------
st.sidebar.header("Dashboard Configuration")

data_path = st.sidebar.text_input("CSV file path", value="data/case_study.csv")

if os.path.exists(data_path):
    df = load_and_process(data_path)

    # Sidebar filters
    min_date, max_date = df['date'].min(), df['date'].max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    selected_airlines = []

    if 'airline' in df.columns:
        df = df.dropna()
        airlines = sorted(df['airline'].unique().tolist())
        selected_airlines = st.sidebar.multiselect("Airline(s)", airlines, default=airlines)

    refs = sorted(df['reference_number'].unique().tolist())
    selected_refs = st.sidebar.multiselect("Reference Number(s)", refs, default=refs)

    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect("Year(s)", years, default=years)

    # Filtering
    mask = (
        df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) &
        df['reference_number'].isin(selected_refs) &
        df['year'].isin(selected_years)
    )
    if selected_airlines:
        mask &= df['airline'].isin(selected_airlines)
    filtered = df.loc[mask].copy()

    # ---------------- Summary Metrics ----------------
    st.markdown("## Audit Compliance Dashboard")
    cols = st.columns(4)
    metrics = [
        ("Avg. Compliance Rate", f"{filtered['compliance_rate'].mean():.2%}"),
        ("Avg. Audit Density", f"{filtered['audit_density'].mean():.4f}"),
        ("Total Audits", f"{int(filtered['count_of_audit'].sum())}"),
        ("Total Compliant Audits", f"{int(filtered['count_of_compliant_audit'].sum())}")
    ]
    for col, (label, val) in zip(cols, metrics):
        with col:
            st.markdown(styled_metric(label, val), unsafe_allow_html=True)

    # ---------------- Visualizations -----------------
    st.markdown("## Visualizations")
    tab1, tab2, tab3 = st.tabs(["Audit Counts", "Compliance Distribution", "Parallel Categories"])

    with tab1:
        st.plotly_chart(plot_record_count_by_reference(filtered), use_container_width=True)

    with tab2:
        dimension = st.radio("Distribution by:", ["airline", "reference_number"], horizontal=True)
        st.plotly_chart(plot_boxplot_distribution(filtered, dimension), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_parallel_categories(filtered), use_container_width=True)

    # ---------------- Table Section ------------------
    st.markdown("## Aggregated Tables")
    tab_table1, tab_table2 = st.tabs(["By Reference Number", "By Airline"])

    with tab_table1:
        st.plotly_chart(plot_heatmap_table(filtered), use_container_width=True)

    with tab_table2:
        st.plotly_chart(plot_airline_table(filtered), use_container_width=True)

    # ---------------------- Key Insights ------------------
    avg_compliance = filtered['compliance_rate'].mean()
    worst_ref = (filtered.groupby('reference_number')['compliance_rate']
    .mean().idxmin())
    worst_ref_val = (filtered.groupby('reference_number')['compliance_rate']
    .mean().min())


    most_frequent_category = filtered['departures_cat'].mode()[0]


    st.markdown(f"""
    ### Key Insights
    - Overall compliance rate is **{avg_compliance:.2%}**.
    - The lowest performing audit group is **{worst_ref}** with **{worst_ref_val:.2%}** compliance.
    - Departures are most frequently categorized as **{most_frequent_category}**.
    - **{filtered['reference_number'].nunique()} unique audit groups** are being tracked.
    - The dataset covers audits from **{filtered['year'].min()} to {filtered['year'].max()}**.
    - Compliance shows variation across **quarters** — useful for seasonal trend analysis.
    - Around **{len(filtered)} total records** provide a robust sample size.
    """)


else:
    st.sidebar.error(f"File not found: {data_path}")
    st.error("Please provide a valid CSV path in the sidebar.")
