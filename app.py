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
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes fadeInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes fadeInRight {{
        from {{
            opacity: 0;
            transform: translateX(30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}

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
        box-shadow: 0 0px 3px {CUSTOM_PALETTE[4]};
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
        box-shadow: {CUSTOM_PALETTE[1]};
        transition: all 0.25s ease-in-out;
    }}
    .st-card:hover {{
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 20px {CUSTOM_PALETTE[1]};
    }}

    /* Insights Card Enhancement */
    .insights-card {{
        background: linear-gradient(145deg, rgba(17,17,17,0.9), {CUSTOM_PALETTE[3]});
        border-radius: 16px;
        padding: 24px;
        margin-top: 24px;
        transition: all 0.4s ease;
        box-shadow: 0 4px 8px {CUSTOM_PALETTE[4]};
        color: #f0f0f0;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .insights-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {CUSTOM_PALETTE[0]}, {CUSTOM_PALETTE[2]});
    }}
    
    .insights-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.5);
    }}
    
    .insights-card ul {{
        list-style: none;
        padding: 0;
    }}
    
    .insights-card li {{
        padding: 8px 0;
        position: relative;
        padding-left: 24px;
        transition: all 0.3s ease;
    }}
    
    .insights-card li::before {{
        content: '✓';
        position: absolute;
        left: 0;
        color: {CUSTOM_PALETTE[1]};
        font-weight: bold;
    }}
    
    .insights-card li:hover {{
        color: {CUSTOM_PALETTE[1]};
        transform: translateX(4px);
    }}

    /* Loading Animation */
    .loading-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }}
    
    .loading-spinner {{
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255,255,255,0.1);
        border-left: 4px solid {CUSTOM_PALETTE[1]};
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .score-card {{
            margin-bottom: 16px;
        }}
        .section-header {{
            font-size: 1.4rem;
        }}
    }}
    
    /* Dashboard Title */
    .dashboard-title {{
        background: linear-gradient(135deg, {CUSTOM_PALETTE[1]}, {CUSTOM_PALETTE[2]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 32px;
        animation: fadeInUp 0.8s ease-out;
    }}
    
    /* Plotly Chart Styling */
    .js-plotly-plot {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}

    /* Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        padding: 9px;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 3px;
        padding: 9px;
        transition: all 0.3s ease;
        color: rgba(255,255,255,0.7);
        font-weight: 250;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {CUSTOM_PALETTE[1]}, {CUSTOM_PALETTE[2]}) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    
    /* Section Headers with Icons */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin: 32px 0 20px 0;
        padding: 16px 0;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        position: relative;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .section-header::before {{
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, {CUSTOM_PALETTE[1]}, {CUSTOM_PALETTE[2]});
    }}

    /* Enhanced Chart Containers */
    .chart-container {{
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
    }}
    
    .chart-container:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
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


    # Key Insights in Bootstrap container
    avg_compliance = filtered['compliance_rate'].mean()
    worst_ref = (filtered.groupby('reference_number')['compliance_rate'].mean().idxmin())
    worst_ref_val = (filtered.groupby('reference_number')['compliance_rate'].mean().min())
    most_frequent_category = filtered['departures_cat'].mode()[0]
    
    insights_html = f"""
    <div class="bootstrap-container">
        <div class="insights-card">
            <div class="section-header">
                <i class="bi bi-lightbulb"></i> Key Insights
            </div>
            <ul>
                <li>Overall compliance rate is <strong>{avg_compliance:.2%}</strong></li>
                <li>The lowest performing audit group is <strong>{worst_ref}</strong> with <strong>{worst_ref_val:.2%}</strong> compliance</li>
                <li>Departures are most frequently categorized as <strong>{most_frequent_category}</strong></li>
                <li><strong>{filtered['reference_number'].nunique()} unique audit groups</strong> are being tracked</li>
                <li>The dataset covers audits from <strong>{filtered['year'].min()} to {filtered['year'].max()}</strong></li>
                <li>Compliance shows variation across quarters — useful for seasonal trend analysis</li>
                <li>Around <strong>{len(filtered):,} total records</strong> provide a robust sample size</li>
            </ul>
        </div>
    </div>
    """
    
    st.markdown(insights_html, unsafe_allow_html=True)


else:
    st.sidebar.error(f"File not found: {data_path}")
    st.error("Please provide a valid CSV path in the sidebar.")
