import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def extract_columns(df, user_input):
    columns = []
    user_input = user_input.lower()
    for col in df.columns:
        col_lower = col.lower()
        if re.search(rf"\b{re.escape(col_lower)}\b", user_input):
            columns.append(col)
    if not columns and " of " in user_input:
        last_part = user_input.split("of")[-1].strip()
        for col in df.columns:
            if col.lower() in last_part:
                columns.append(col)
    if not columns and " between " in user_input and " and " in user_input:
        parts = user_input.split("between")[-1].split(" and ")
        if len(parts) >= 2:
            col1 = parts[0].strip()
            col2 = parts[1].split()[0].strip()
            if col1 in df.columns and col2 in df.columns:
                columns = [col1, col2]
    if not columns and " for " in user_input:
        last_part = user_input.split("for")[-1].strip()
        for col in df.columns:
            if col.lower() in last_part:
                columns.append(col)
    return columns

def get_visualization(df, user_input, viz_type):
    columns = extract_columns(df, user_input)
    bins = 20
    if "bins" in user_input:
        try:
            bins_match = re.search(r'(\d+)\s+bins', user_input)
            bins = int(bins_match.group(1)) if bins_match else 20
        except:
            pass
    if not columns:
        raise ValueError("Couldn't identify columns from your query")
    if viz_type == "histogram":
        fig = histogram(df, columns[0], bins)
    elif viz_type == "boxplot":
        fig = boxplot(df, columns[0])
    elif viz_type == "scatter":
        if len(columns) < 2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols and columns[0] in numeric_cols:
                columns.append(numeric_cols[0])
            if len(columns) < 2:
                raise ValueError("Need at least two columns for scatter plot")
        fig = scatterplot(df, columns[0], columns[1])
    elif viz_type == "bar":
        fig = barchart(df, columns[0])
    elif viz_type == "line":
        fig = linechart(df, columns[0])
    elif viz_type == "piechart":
        fig = piechart(df, columns[0])
    elif viz_type == "heatmap":
        fig = heatmap(df)
    elif viz_type == "radar":
        if len(columns) < 3:
            raise ValueError("Need at least 3 columns for radar plot")
        fig = radar_plot(df, columns[:5], df[columns[:5]].mean().values.tolist())
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
    return fig, ', '.join(columns[:3])

def histogram(df, col, bins=20):
    fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
    return fig

def boxplot(df, col):
    fig = px.box(df, y=col, title=f"Boxplot of {col}")
    return fig

def scatterplot(df, col1, col2):
    fig = px.scatter(df, x=col1, y=col2, title=f"Scatter Plot: {col1} vs {col2}")
    return fig

def barchart(df, col):
    fig = px.bar(df, x=col, title=f"Distribution of {col}")
    return fig

def linechart(df, col):
    fig = px.line(df, y=col, title=f"Trend of {col}")
    return fig

def piechart(df, col):
    fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
    return fig

def heatmap(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[columns].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title="Correlation Heatmap")
    return fig

def radar_plot(df, categories, values):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Radar Plot"
    )
    return fig

def correlation_matrix(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[columns].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}"
    ))
    fig.update_layout(title="Correlation Matrix")
    return fig

def scatter_matrix(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()[:5]
    fig = px.scatter_matrix(df[columns])
    fig.update_layout(title="Scatter Matrix")
    return fig

def correlation_plot(df, col1, col2):
    fig = px.scatter(df, x=col1, y=col2, trendline="ols",
                     title=f"Correlation: {col1} vs {col2}")
    return fig
