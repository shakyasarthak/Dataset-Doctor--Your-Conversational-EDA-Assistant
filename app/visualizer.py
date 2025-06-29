import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import numpy as np

def extract_columns(df, user_input):
    """Enhanced column extraction with NLP patterns"""
    columns = []
    user_input = user_input.lower()
    
    # Pattern 1: Direct column names
    for col in df.columns:
        col_lower = col.lower()
        if re.search(rf"\b{re.escape(col_lower)}\b", user_input):
            columns.append(col)
    
    # Pattern 2: "of " pattern
    if not columns and " of " in user_input:
        last_part = user_input.split("of")[-1].strip()
        for col in df.columns:
            if col.lower() in last_part:
                columns.append(col)
    
    # Pattern 3: "between X and Y"
    if not columns and " between " in user_input and " and " in user_input:
        parts = user_input.split("between")[-1].split(" and ")
        if len(parts) >= 2:
            col1 = parts[0].strip()
            col2 = parts[1].split()[0].strip()
            if col1 in df.columns and col2 in df.columns:
                columns = [col1, col2]
    
    # Pattern 4: "for X" pattern
    if not columns and " for " in user_input:
        last_part = user_input.split("for")[-1].strip()
        for col in df.columns:
            if col.lower() in last_part:
                columns.append(col)
    
    return columns

def get_visualization(df, user_input, viz_type):
    """Unified visualization handler"""
    columns = extract_columns(df, user_input)
    
    # Handle numeric parameters
    bins = 20
    if "bins" in user_input:
        try:
            bins_match = re.search(r'(\d+)\s+bins', user_input)
            bins = int(bins_match.group(1)) if bins_match else 20
        except:
            pass
    
    # Generate visualization
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

# Visualization functions
def histogram(df, col, bins=20):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[col].dropna(), kde=True, bins=bins)
    ax.set_title(f"Histogram of {col}", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def boxplot(df, col):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=df[col])
    ax.set_title(f"Boxplot of {col}", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def scatterplot(df, col1, col2):
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x=df[col1], y=df[col2])
    ax.set_title(f"Scatter Plot: {col1} vs {col2}", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def barchart(df, col):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=df[col])
    ax.set_title(f"Distribution of {col}", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def linechart(df, col):
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=df, x=df.index, y=col)
    ax.set_title(f"Trend of {col}", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def piechart(df, col):
    plt.figure(figsize=(10, 6))
    ax = df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Pie Chart of {col}", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def heatmap(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm')
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def radar_plot(df, categories, values):
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.plot(angles, values, color='royalblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Plot", fontsize=14)
    plt.tight_layout()
    return plt.gcf()
