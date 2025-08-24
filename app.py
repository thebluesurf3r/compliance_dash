import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ---------------------- Streamlit Config ----------------------
st.set_page_config(
    page_title="Airline Compliance Dashboard",
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
        60% {{ transform: scale(1.01); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -900px 0; }}
        100% {{ background-position: 900px 0; }}
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(145deg, rgba(17,17,17,0.9), rgba(42, 8, 181, 0.8)), 
                url('https://user-gen-media-assets.s3.amazonaws.com/gpt4o_images/e51f93e6-7344-4fc9-8110-cd84c6203f43.png');
                # url('https://images.pexels.com/photos/2646237/pexels-photo-2646237.jpeg');
                # url('https://images.pexels.com/photos/2253921/pexels-photo-2253921.jpeg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        # background: linear-gradient(180deg, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
        color: #fff !important;
    }}    

    /* Score cards */
    .score-card {{
        # background: linear-gradient(145deg, {CUSTOM_PALETTE[2]}, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
        background: linear-gradient(145deg, rgba(17,17,17,0.9), rgba(42, 8, 181, 0.8));
        border-radius: 9px;
        padding: 18px;
        text-align: center;
        color: #ffffff !important; /* Force black text */
        box-shadow: 0 0px 3px {CUSTOM_PALETTE[4]};
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        transition: transform 0.2s ease-in-out;
    }}
    .score-card:hover {{
        transform: translateY(-3px);
    }}
    .score-label {{
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 3px;
        display: block;
        color: ##ffffff !important; /* black labels */
    }}
    .score-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: ##ffffff !important; /* black values */
    }}

    /* Unchecked box background */
    div[role="checkbox"] > div:first-child {{
        border: 3px solid {CUSTOM_PALETTE[0]} !important;
        background-color: transparent !important;
        border-radius: 3px;
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
        padding: 18px 21px;
        margin: 9px 6px;
        color: #fff;
        box-shadow: {CUSTOM_PALETTE[1]};
        transition: all 0.3s ease-in-out;
    }}
    .st-card:hover {{
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 12px 18px {CUSTOM_PALETTE[1]};
    }}

    /* Insights Card Enhancement */
    .insights-card {{
        background: linear-gradient(145deg, rgba(17,17,17,0.9), {CUSTOM_PALETTE[3]}), 
                url('https://img.freepik.com/premium-photo/modern-airplane-model-hightech-circuit-board-background-aviation-industry_1121645-6059.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        # background: linear-gradient(145deg, rgba(17,17,17,0.9), {CUSTOM_PALETTE[3]});
        border-radius: 12px;
        padding: 15px;
        margin-top: 0px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 9px {CUSTOM_PALETTE[4]};
        color: #f0f0f0;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .insights-card::before {{
        position: relative;
        top: 0;
        left: 0;
        right: 0;
        height: 0;
        background: linear-gradient(180deg, {CUSTOM_PALETTE[1]}, {CUSTOM_PALETTE[4]});
    }}
    
    .insights-card:hover {{
        # transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.5);
    }}
    
    .insights-card ul {{
        list-style: none;
        padding: 0;
    }}
    
    .insights-card li {{
        padding: 3px 0;
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
        # transform: translateX(4px);
    }}

    /* Loading Animation */
    .loading-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 240px;
    }}
    
    .loading-spinner {{
        width: 30px;
        height: 30px;
        border: 3px solid rgba(255,255,255,0.1);
        border-left: 3px solid {CUSTOM_PALETTE[1]};
        border-radius: 60%;
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
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
        animation: fadeInUp 0.9s ease-out;
    }}
    
    /* Plotly Chart Styling */
    .js-plotly-plot {{
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 3px 18px rgba(0,0,0,0.2);
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
        font-weight: 240;
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
        margin: 3px 0 6px 0; /* reduced top margin from 32px to 8px */
        padding: 3px 0;       /* reduced padding from 16px to 8px */
        border-bottom: 2px solid rgba(255,255,255,0.1);
        position: relative;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .section-header::before {{
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, {CUSTOM_PALETTE[3]}, {CUSTOM_PALETTE[4]});
    }}

    /* Enhanced Chart Containers */
    .chart-container {{
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
    }}
    
    .chart-container:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }}

    /* Table Section Height Control */
    .table-section-container {{
        min-height: 270px;
        max-height: 300px;
        margin-bottom: 2rem;
    }}

    .table-chart-container {{
        height: 300px;
        overflow: hidden;
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

def plot_record_count(filtered: pd.DataFrame, x_axis: str = "reference_number"):
    # Validate the x_axis input
    assert x_axis in ["reference_number", "airline"], "x_axis must be 'reference_number' or 'airline'"

    # Aggregate counts by selected x_axis
    count_df = (
        filtered[x_axis]
        .value_counts()
        .reset_index()
    )
    count_df.columns = [x_axis, 'count']  # rename columns cleanly
    count_df = count_df.sort_values('count', ascending=False)

    # Set plot title and x-axis label depending on x_axis
    title_map = {
        "reference_number": "Count of Audits per Audit Group",
        "airline": "Count of Audits per Airline"
    }
    layout_xaxis_title = {
        "reference_number": "Reference Number",
        "airline": "Airline"
    }
    color_scale = "CUSTOM_PALETTE"  # Assuming you want the same palette; replace if needed

    fig = px.bar(
        count_df,
        x=x_axis,
        y='count',
        color='count',
        color_continuous_scale=CUSTOM_PALETTE,
        title=title_map[x_axis]
    )

    # Add data labels
    fig.update_traces(
        text=count_df['count'],
        textposition="outside",
        marker_line=dict(width=0.8, color="black")
    )

    # Add reference line for average count
    avg_val = count_df['count'].mean()
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red"
    )

    # Add annotation at center for average
    fig.add_annotation(
        x=0.5,
        y=avg_val,
        xref="paper",
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
        xaxis=dict(title=layout_xaxis_title[x_axis], tickangle=-45, showgrid=False),
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
                    font=dict(color="white", size=11)
                )
            )
        ]
    )

    fig.update_layout(title="Aggregated Metrics by Audit Group", height=270)
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
                    fill_color=CUSTOM_PALETTE[4],
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
                    font=dict(color="white", size=11)
                )
            )
        ]
    )

    fig.update_layout(title="Aggregated Metrics by Airline", height=300)
    return fig

def plot_compliance_timeseries(filtered: pd.DataFrame, metric: str = "Compliance Rate"):
    import numpy as np

    if metric == "Compliance Rate":
        timeseries = (
            filtered.groupby('date')['compliance_rate']
            .mean()
            .reset_index()
            .sort_values('date')
        )
        if len(timeseries) > 7:
            timeseries['value_smooth'] = timeseries['compliance_rate'].rolling(window=7, min_periods=1).mean()
        else:
            timeseries['value_smooth'] = timeseries['compliance_rate']

        y_title = "Compliance Rate"
        y_values = 'value_smooth'
        line_color = "#3960fb"
        marker_color = "#6583fd"
        hover_template = "%{y:.2%}"
        show_avg_line = True
        avg_val = timeseries['compliance_rate'].mean()

        fig = px.line(
            timeseries,
            x='date',
            y=y_values,
            title=f"{y_title} Over Time",
            color_discrete_sequence=[line_color],
        )
        fig.add_scatter(
            x=timeseries['date'],
            y=timeseries[y_values],
            mode='markers',
            marker=dict(color=marker_color, size=6, opacity=0.6),
            name="Data Points"
        )

        if show_avg_line:
            fig.add_hline(y=avg_val, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=0.5,
                y=avg_val,
                xref="paper",
                yref="y",
                text=f"Avg: {avg_val:.2%}",
                showarrow=False,
                font=dict(color="red", size=12),
                bgcolor="black",
                bordercolor="red",
                borderwidth=1,
                borderpad=4
            )

    else:  # Count of Audits with multiple statistics
        # Calculate stats per date
        stats_df = filtered.groupby('date').agg(
            count=('date', 'size'),
            mean=('count_of_audit', 'mean'),
            median=('count_of_audit', 'median'),
            std=('count_of_audit', 'std'),
            min_val=('count_of_audit', 'min'),
            max_val=('count_of_audit', 'max')
        ).reset_index().sort_values('date')

        y_title = "Count of Audits"
        hover_template = "%{y}"

        # Base line plot for 'count' (number of records)
        fig = px.line(
            stats_df,
            x='date',
            y='count',
            title=f"{y_title} Over Time with Statistics",
            color_discrete_sequence=["#39a8fb"],
        )
        fig.add_scatter(
            x=stats_df['date'],
            y=stats_df['count'],
            mode='markers',
            marker=dict(color="#6fbfff", size=6, opacity=0.6),
            name="Count of Audits"
        )

        # Add lines for mean, median, min, max, std (std shown as +/-)
        fig.add_scatter(
            x=stats_df['date'],
            y=stats_df['mean'],
            mode='lines',
            line=dict(color='green', dash='dot'),
            name='Mean of count_of_audit'
        )
        fig.add_scatter(
            x=stats_df['date'],
            y=stats_df['median'],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Median of count_of_audit'
        )
        fig.add_scatter(
            x=stats_df['date'],
            y=stats_df['min_val'],
            mode='lines',
            line=dict(color='red', dash='dashdot'),
            name='Min of count_of_audit'
        )
        fig.add_scatter(
            x=stats_df['date'],
            y=stats_df['max_val'],
            mode='lines',
            line=dict(color='purple', dash='longdash'),
            name='Max of count_of_audit'
        )
        # For std, plot mean +/- std as shaded area or two lines (optional):
        # Here we add mean+std and mean-std lines if std is not NaN
        if not stats_df['std'].isnull().all():
            fig.add_scatter(
                x=stats_df['date'],
                y=stats_df['mean'] + stats_df['std'],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='Mean + STD',
                opacity=0.4
            )
            fig.add_scatter(
                x=stats_df['date'],
                y=stats_df['mean'] - stats_df['std'],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='Mean - STD',
                opacity=0.4
            )

    fig.update_layout(
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title=y_title),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )

    fig.update_traces(hovertemplate=hover_template)

    return fig


def styled_metric(label, value):
    return f"""
    <div class="score-card">
        <span class="score-label">{label}</span>
        <span class="score-value">{value}</span>
    </div>
    """


def filter_valid_invalid(df: pd.DataFrame, mode="Invalid Data") -> pd.DataFrame:
    """
    Returns either valid or invalid records depending on mode.
    'Valid Data' -> removes nulls, NaNs, 0s in numeric columns
    'Invalid Data' -> keeps only rows with nulls, NaNs, or 0s
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if mode == "Invalid Data":
        mask = df[numeric_cols].isna().any(axis=1) | (df[numeric_cols] == 0).any(axis=1)
        return df[mask].copy()
    else:  # Valid Data
        mask = df[numeric_cols].notna().all(axis=1) & (df[numeric_cols] != 0).all(axis=1)
        return df[mask].copy()



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
    st.markdown("## Airline Compliance Dashboard")
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
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Audit Counts", "Outliers", "Parallel Categories"])

    # with tab2:
    #     st.plotly_chart(plot_record_count_by_reference(filtered), use_container_width=True)

    with tab3:
        dimension = st.radio(
            "Group by:", 
            ["airline", "reference_number"], 
            horizontal=True,
            format_func=lambda x: "Audit Group" if x == "reference_number" else "Airline"
        )
        st.plotly_chart(plot_boxplot_distribution(filtered, dimension), use_container_width=True)


    with tab2:
        x_axis_choice = st.radio("Count by:", ["airline", "reference_number"], format_func=lambda x: "Audit Group" if x=="reference_number" else "Airline", horizontal=True)
        st.plotly_chart(plot_record_count(filtered, x_axis=x_axis_choice), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_parallel_categories(filtered), use_container_width=True)

    with tab1:
        metric = st.radio("Select Metric:", ["Compliance Rate", "Count of Audits"], horizontal=True)
        st.plotly_chart(plot_compliance_timeseries(filtered, metric=metric), use_container_width=True)

    # ---------------- Table Section ------------------
    st.markdown("## Aggregated Tables")
    tab_table1, tab_table2 = st.tabs(["Audit Group", "Airline"])

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


    # ---------------------- Key Insights ------------------
    avg_compliance = filtered['compliance_rate'].mean()
    worst_ref = filtered.groupby('reference_number')['compliance_rate'].mean().idxmin()
    worst_ref_val = filtered.groupby('reference_number')['compliance_rate'].mean().min()
    best_ref = filtered.groupby('reference_number')['compliance_rate'].mean().idxmax()
    best_ref_val = filtered.groupby('reference_number')['compliance_rate'].mean().max()
    most_frequent_category = filtered['departures_cat'].mode()[0]

    # Top and bottom performing airlines
    airline_compliance = filtered.groupby('airline')['compliance_rate'].mean()
    best_airline = airline_compliance.idxmax()
    best_airline_val = airline_compliance.max()
    worst_airline = airline_compliance.idxmin()
    worst_airline_val = airline_compliance.min()

    # Date coverage
    start_date, end_date = filtered['date'].min(), filtered['date'].max()

    # Avg and total audits
    total_audits = filtered['count_of_audit'].sum()
    total_compliant_audits = filtered['count_of_compliant_audit'].sum()
    avg_audit_density = filtered['audit_density'].mean()

    # Compliance variability
    std_compliance = filtered['compliance_rate'].std()
    max_compliance = filtered['compliance_rate'].max()
    min_compliance = filtered['compliance_rate'].min()

    # Temporal insights
    quarterly_compliance = filtered.groupby('quarter')['compliance_rate'].mean()
    best_quarter = quarterly_compliance.idxmax()
    worst_quarter = quarterly_compliance.idxmin()
    weekend_compliance = filtered.groupby('is_weekend')['compliance_rate'].mean()
    weekend_diff = weekend_compliance.get(1,0) - weekend_compliance.get(0,0)

    insights_html = f"""
    <div class="bootstrap-container">
        <div class="insights-card">
            <div class="section-header">
                <i class="bi bi-lightbulb"></i> Key Insights
            </div>
            <ul>
                <li>Overall compliance rate is <strong>{avg_compliance:.2%}</strong></li>
                <li>Audit group with minimum average compliance: <strong>{worst_ref}</strong> ({worst_ref_val:.2%})</li>
                <li>Audit group with maximum average compliance: <strong>{best_ref}</strong> ({best_ref_val:.2%})</li>
                <li>Most frequent departure category: <strong>{most_frequent_category}</strong></li>
                <li><strong>{filtered['reference_number'].nunique()} unique audit groups</strong> are included in the dataset</li>
                <li>Dataset covers audits from <strong>{start_date.date()} to {end_date.date()}</strong></li>
                <li>Quarterly compliance shows natural variation — useful for seasonal trend analysis</li>
                <li>Total records analyzed: <strong>{len(filtered):,}</strong></li>
                <li>Total audits conducted: <strong>{total_audits:,}</strong></li>
                <li>Total compliant audits: <strong>{total_compliant_audits:,}</strong></li>
                <li>Average audit density: <strong>{avg_audit_density:.4f}</strong></li>
                <li>Airline with maximum average compliance: <strong>{best_airline}</strong> ({best_airline_val:.2%})</li>
                <li>Airline with minimum average compliance: <strong>{worst_airline}</strong> ({worst_airline_val:.2%})</li>
                <li>Compliance rate standard deviation: <strong>{std_compliance:.2%}</strong></li>
                <li>Maximum compliance observed: <strong>{max_compliance:.2%}</strong></li>
                <li>Minimum compliance observed: <strong>{min_compliance:.2%}</strong></li>
                <li>Quarter with maximum compliance: <strong>Q{best_quarter}</strong>; Quarter with minimum compliance: <strong>Q{worst_quarter}</strong></li>
                <li>Difference in compliance between weekends and weekdays: <strong>{weekend_diff:.2%}</strong></li>
            </ul>
        </div>
    </div>
    """


    st.markdown(insights_html, unsafe_allow_html=True)

else:
    st.sidebar.error(f"File not found: {data_path}")
    st.error("Please provide a valid CSV path in the sidebar.")
