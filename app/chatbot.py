import os
from together import Together
import pandas as pd
from eda_engine import get_summary, get_missing_report, generate_profile, clean_data, suggest_features
from visualizer import get_visualization

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def together_chat(prompt, system_message=None):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content.strip()

def respond(df, user_input):
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
        elif intent == "suggestions":
            return handle_suggestions(df)
        else:
            return handle_general_query(df, user_input)
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}. Try asking differently!"

def understand_intent(user_input):
    user_input = user_input.lower()
    intent_map = {
        "visualization": ["histogram", "distribution", "chart", "plot", "graph", "visualize", "show me"],
        "summary": ["summarize", "describe", "overview", "stats", "statistics"],
        "missing": ["missing", "null", "na", "empty", "blank"],
        "report": ["report", "profile", "analysis", "eda", "exploratory"],
        "clean": ["clean", "tidy", "preprocess", "outlier", "impute", "handle"],
        "correlation": ["correlation", "relationship", "correlate", "r value", "r-value", "covariance"],
        "suggestions": ["suggest", "feature", "engineering", "transform", "encode"]
    }
    for intent, keywords in intent_map.items():
        if any(keyword in user_input for keyword in keywords):
            return intent
    return "general"

def handle_visualization(df, query):
    import plotly.express as px
    if "refine" in query.lower():
        if 'last_fig' in st.session_state:
            fig = st.session_state.last_fig
            if "add trendline" in query.lower():
                fig = px.scatter(df, x=fig.data[0].x, y=fig.data[0].y, trendline="ols")
            if "change color" in query.lower():
                color = "red" if "red" in query.lower() else "blue"
                fig.update_traces(marker=dict(color=color))
            st.session_state.last_fig = fig
            return (f"Refined visualization based on your request", fig)
        else:
            return "No previous visualization to refine. Please generate a visualization first."
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
    import streamlit as st
    st.session_state.last_fig = fig
    return (f"Here's the {viz_type} for `{col_name}`", fig)

def handle_summary(df):
    return get_summary(df)

def handle_missing(df):
    return get_missing_report(df)

def handle_report(df):
    report_path = generate_profile(df)
    with open(report_path, "r", encoding="utf-8") as f:
        report_html = f.read()
    summary = f"🔍 Here's my analysis of your dataset:\n\n"\
              f"- Found {len(df.columns)} features and {len(df)} records\n\n"\
              f"Full EDA report is ready below 👇"
    return (summary, "report", report_html)

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

def handle_suggestions(df):
    suggestions = suggest_features(df)
    if suggestions:
        return "Here are some feature engineering suggestions:\n- " + "\n- ".join(suggestions)
    else:
        return "No feature engineering suggestions at this time."

def handle_general_query(df, user_input):
    # Optionally, add a system message for context
    system_message = "You are a data analysis assistant. Answer accurately and step-by-step."
    return together_chat(user_input, system_message)

def generate_dynamic_suggestions(df):
    suggestions = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    if numeric_cols:
        if len(numeric_cols) > 1:
            suggestions.append(f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        suggestions.append(f"Plot distribution of {numeric_cols[0]}")
    if categorical_cols:
        suggestions.append(f"Show bar chart of {categorical_cols[0]}")
    suggestions.append("Generate EDA report")
    suggestions.append("Check for missing values")
    suggestions.append("Suggest feature engineering")
    return suggestions[:5]

def generate_explanation(df, last_query, last_response):
    prompt = (
        f"User asked: {last_query}\n"
        f"Assistant answered: {last_response}\n"
        "Explain, step-by-step, how you arrived at this answer. "
        "Include reasoning, calculations, and any assumptions."
    )
    return together_chat(prompt)

def generate_recommendations(df, last_query, last_response):
    preview = (
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns. "
        f"Columns: {', '.join(df.columns[:8])}."
    )
    prompt = (
        f"User's last question: {last_query}\n"
        f"Assistant's last answer: {last_response}\n"
        f"{preview}\n"
        "Based on this, suggest 2-3 next-step analyses or questions the user might ask next. "
        "Be proactive and context-aware."
    )
    return together_chat(prompt)

