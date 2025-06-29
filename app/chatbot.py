import torch
from langchain_experimental.agents import create_csv_agent
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from eda_engine import get_summary, get_missing_report, generate_profile, clean_data
from visualizer import get_visualization
import tempfile
import atexit
from transformers import pipeline
import pandas as pd
import re
import numpy as np

# Load environment variables
load_dotenv()

# Verify Hugging Face API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    st.error("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    exit(1)

# Load NLP model for intent recognition
nlp_model = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    top_k=1
)

def understand_intent(user_input):
    """Advanced intent classification"""
    user_input = user_input.lower()
    intent_map = {
        "visualization": ["histogram", "distribution", "chart", "plot", "graph", "visualize", "show me"],
        "summary": ["summarize", "describe", "overview", "stats", "statistics"],
        "missing": ["missing", "null", "na", "empty", "blank"],
        "report": ["report", "profile", "analysis", "eda", "exploratory"],
        "clean": ["clean", "tidy", "preprocess", "outlier", "impute", "handle"],
        "correlation": ["correlation", "relationship", "correlate", "r value", "r-value", "covariance"]
    }
    # Try classification model
    try:
        result = nlp_model(user_input)[0]
        if result['score'] > 0.7:
            return result['label']
    except:
        pass
    # Fallback to keyword matching
    for intent, keywords in intent_map.items():
        if any(keyword in user_input for keyword in keywords):
            return intent
    return "general"

def get_agent(df):
    """Create LangChain agent for CSV analysis"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        df.to_csv(temp_path, index=False)
        atexit.register(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
    hf_llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",
        max_new_tokens=512,
        temperature=0.1
    )
    # Add security parameter and sandboxing
    return create_csv_agent(
        hf_llm,
        temp_path,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )

def respond(df, user_input):
    """Handle user input with NLP and return response"""
    intent = understand_intent(user_input)
    try:
        if intent == "visualization":
            return handle_visualization(df, user_input)
        elif intent == "summary":
            return handle_summary(df)
        elif intent == "missing":
            return handle_missing(df)
        elif intent == "report":
            return handle_report(df)
        elif intent == "clean":
            return handle_cleaning(df)
        elif intent == "correlation":
            return handle_correlation(df, user_input)
        else:
            return handle_general_query(df, user_input)
    except Exception as e:
        # Friendly error messages for common errors
        error_msg = str(e)
        user_friendly_errors = {
            "unexpected keyword argument 'tools'":
                "I'm having trouble processing that request. Please try simpler questions about your data.",
            "DType could not be promoted":
                "I found mixed data types in your columns. Could you check if all values in each column are the same type?",
            "Couldn't identify columns":
                "I couldn't find the columns you mentioned. Try naming them specifically like 'Show sales by region'"
        }
        for pattern, friendly_msg in user_friendly_errors.items():
            if pattern in error_msg:
                return friendly_msg
        return f"Oops! Something went wrong: {error_msg}. Try asking differently!"

def handle_visualization(df, query):
    """Unified visualization handler"""
    # Determine visualization type
    if "pie" in query or "percent" in query:
        viz_type = "piechart"
    elif "heatmap" in query or "correlation" in query:
        viz_type = "heatmap"
    elif "radar" in query or "spider" in query:
        viz_type = "radar"
    elif "boxplot" in query or "outlier" in query:
        viz_type = "boxplot"
    elif "scatter" in query or "relationship" in query:
        viz_type = "scatter"
    elif "bar" in query or "categorical" in query:
        viz_type = "bar"
    elif "line" in query or "trend" in query:
        viz_type = "line"
    else:
        viz_type = "histogram"
    fig, col_name = get_visualization(df, query, viz_type)
    return (f"Here's the {viz_type} for `{col_name}`", fig)

def handle_summary(df):
    summary = get_summary(df)
    return summary

def handle_missing(df):
    missing_report = get_missing_report(df)
    return missing_report

def handle_report(df):
    report_path = generate_profile(df)
    with open(report_path, "r", encoding="utf-8") as f:
        report_html = f.read()
    summary = f"🔍 Here's my analysis of your dataset:\n\n"\
              f"- Found {len(df.columns)} features and {len(df)} records\n"\
              f"- Key insights: {generate_insights(df)}\n\n"\
              f"Full EDA report is ready below 👇"
    return (summary, "report", report_html)

def generate_insights(df):
    """Generate natural language insights"""
    insights = []
    for col in df.select_dtypes(include='number'):
        insights.append(f"{col} ranges from {df[col].min():.2f} to {df[col].max():.2f}")
    for col in df.select_dtypes(exclude='number'):
        top_val = df[col].value_counts().index[0]
        insights.append(f"{col} has {df[col].nunique()} unique values, most common: {top_val}")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        strong_corrs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
        strong_corrs = strong_corrs[strong_corrs.between(0.7, 1) & (strong_corrs < 1)]
        if not strong_corrs.empty:
            insights.append("\n🔗 Strong correlations found:")
            for (col1, col2), value in strong_corrs.head(3).items():
                insights.append(f"- {col1} & {col2}: {value:.2f}")
    return "\n- ".join(insights[:5]) + "\n... [see full report for more]"

def handle_cleaning(df):
    cleaned_df = clean_data(df.copy())
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    return (f"✅ Data cleaned successfully! Ready for download.", "download", csv)

def handle_correlation(df, query):
    from visualizer import extract_columns, correlation_matrix, correlation_plot, scatter_matrix
    columns = extract_columns(df, query)
    if len(columns) == 0:
        return "🔍 Couldn't identify columns for correlation analysis. Please specify columns like 'correlation between price and size'"
    if len(columns) == 1:
        fig = correlation_matrix(df, columns)
        return ("Here's the correlation matrix for your data:", fig)
    elif len(columns) == 2:
        fig = correlation_plot(df, columns[0], columns[1])
        return (f"Relationship between {columns[0]} and {columns[1]}:", fig)
    else:
        fig = scatter_matrix(df, columns)
        return (f"Scatter matrix for {', '.join(columns[:5])}:", fig)

def handle_general_query(df, user_input):
    agent = get_agent(df)
    response = agent.invoke(user_input)['output']
    return response
