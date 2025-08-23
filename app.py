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
    "#273C76",  # Rhapsody Blue
    "#005BAC",  # United Blue
    "#1CA9C9",  # Pacific Blue
    "#5C6670",  # Runway Gray
    "#87CEEB",  # Sky Blue
    "#6F42C1",  # Atlantic Amethyst
    "#800080"   # Premium Purple
]

CUSTOM_COLOR_SCALE = [
    [0.0, "#273C76"],
    [0.16, "#005BAC"],
    [0.33, "#1CA9C9"],
    [0.5, "#5C6670"],
    [0.66, "#87CEEB"],
    [0.83, "#6F42C1"],
    [1.0, "#800080"]
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
        background: linear-gradient(180deg, {CUSTOM_PALETTE[0]}, {CUSTOM_PALETTE[3]});
        color: #fff !important;
    }}

    /* Score cards */
    .score-card {{
        background: linear-gradient(145deg, {CUSTOM_PALETTE[0]}, {CUSTOM_PALETTE[1]}, {CUSTOM_PALETTE[2]});
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #000 !important; /* Force black text */
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
        color: #000 !important; /* black labels */
    }}
    .score-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #000 !important; /* black values */
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
    return df


# ---------------------- Visualization Functions ----------------
def plot_record_count_by_reference(filtered: pd.DataFrame):
    count_ref = filtered['reference_number'].value_counts().reset_index()
    count_ref.columns = ['reference_number', 'count']
    count_ref = count_ref.sort_values('count', ascending=False)

    fig = px.bar(
        count_ref,
        x='reference_number',
        y='count',
        color='reference_number',
        title="Count of Audits per Audit Group"
    )
    fig.update_traces(
        text=count_ref['count'],
        textposition='outside',
        hovertemplate="<b>Reference:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
    )
    fig.update_layout(
        xaxis=dict(title="Reference Number", tickangle=-45, showgrid=False),
        yaxis=dict(title="Record Count"),
        bargap=0.3
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
    pcat_cols = ['reference_number', 'quarter', 'year']
    if 'airline' in filtered.columns:
        pcat_cols.insert(1, 'airline')

    filtered = filtered.copy()
    filtered['quarter'] = filtered['quarter'].astype(str)

    dimensions = [dict(values=filtered[col], label=col.capitalize()) for col in pcat_cols]

    categories = filtered.get("airline", pd.Series([])).astype(str).unique()
    cat_to_num = {cat: i for i, cat in enumerate(categories)}
    filtered["_cat_code"] = filtered.get("airline", pd.Series([])).map(cat_to_num)

    fig = go.Figure(data=[go.Parcats(
        dimensions=dimensions,
        line=dict(color=filtered["_cat_code"], colorscale=CUSTOM_COLOR_SCALE, shape="hspline")
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
            row_colors.append(CUSTOM_PALETTE[1])
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
                    fill_color=CUSTOM_PALETTE[0],
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
    st.markdown("## 📈 Visualizations")
    tab1, tab2, tab3 = st.tabs(["Audit Counts", "Compliance Distribution", "Parallel Categories"])

    with tab1:
        st.plotly_chart(plot_record_count_by_reference(filtered), use_container_width=True)

    with tab2:
        dimension = st.radio("Distribution by:", ["airline", "reference_number"], horizontal=True)
        st.plotly_chart(plot_boxplot_distribution(filtered, dimension), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_parallel_categories(filtered), use_container_width=True)

    # ---------------- Table Section ------------------
    st.markdown("## 📊 Aggregated Tables")
    tab_table1, tab_table2 = st.tabs(["By Reference Number", "By Airline"])

    with tab_table1:
        st.plotly_chart(plot_heatmap_table(filtered), use_container_width=True)

    with tab_table2:
        st.plotly_chart(plot_airline_table(filtered), use_container_width=True)


else:
    st.sidebar.error(f"File not found: {data_path}")
    st.error("Please provide a valid CSV path in the sidebar.")
