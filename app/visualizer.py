import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd

def get_visualization(df, user_input, viz_type):
    """Unified visualization handler with NLP support"""
    # Extract column names using NLP patterns
    columns = []
    
    # Pattern 1: Direct column names in input
    for col in df.columns:
        if re.search(rf"\b{re.escape(col)}\b", user_input, re.IGNORECASE):
            columns.append(col)
    
    # Pattern 2: "of <column>" pattern
    if not columns and " of " in user_input:
        last_part = user_input.split("of")[-1].strip()
        for col in df.columns:
            if col.lower() in last_part.lower():
                columns.append(col)
    
    # Pattern 3: "between X and Y" for scatter plots
    if viz_type == "scatter" and not columns:
        if " between " in user_input and " and " in user_input:
            parts = user_input.split("between")[-1].split(" and ")
            if len(parts) >= 2:
                col1 = parts[0].strip()
                col2 = parts[1].split()[0].strip()  # Take first word after "and"
                if col1 in df.columns and col2 in df.columns:
                    columns = [col1, col2]
    
    # Generate visualization
    if not columns:
        raise ValueError("Couldn't identify columns from your query")
    
    if viz_type == "histogram":
        return histogram(df, columns[0]), columns[0]
    elif viz_type == "boxplot":
        return boxplot(df, columns[0]), columns[0]
    elif viz_type == "scatter" and len(columns) >= 2:
        return scatterplot(df, columns[0], columns[1]), f"{columns[0]} vs {columns[1]}"
    elif viz_type == "bar":
        return barchart(df, columns[0]), columns[0]
    elif viz_type == "line":
        return linechart(df, columns[0]), columns[0]
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")

def histogram(df, col):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[col].dropna(), kde=True)
    ax.set_title(f"Histogram of {col}", fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def boxplot(df, col):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=df[col])
    ax.set_title(f"Boxplot of {col}", fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def scatterplot(df, col1, col2):
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x=df[col1], y=df[col2])
    ax.set_title(f"Scatter Plot: {col1} vs {col2}", fontsize=14)
    ax.set_xlabel(col1, fontsize=12)
    ax.set_ylabel(col2, fontsize=12)
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
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel(col, fontsize=12)
    plt.tight_layout()
    return plt.gcf()
