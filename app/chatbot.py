from langchain_experimental.agents import create_csv_agent
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from eda_engine import get_summary, get_missing_report, generate_profile, clean_data
from visualizer import get_visualization
import tempfile
import atexit

# Load environment variables
load_dotenv()

# Verify Hugging Face API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    st.error("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    exit(1)

def get_agent(df):
    """Create LangChain agent for CSV analysis with proper temp file handling"""
    # Create named temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        df.to_csv(temp_path, index=False)
    
    # Auto-delete temp file on exit
    atexit.register(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
    
    # Use Hugging Face hosted model
    hf_llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text-generation",
        max_new_tokens=512,  # Increased for complex queries
        temperature=0.2,
        do_sample=False
    )
    
    return create_csv_agent(
        hf_llm,
        temp_path,
        verbose=True,
        allow_dangerous_code=True,  # Required for code execution
        agent_type="openai-tools"   # Better for code execution
    )

def respond(df, user_input):
    """Handle user input with natural language processing and return response"""
    input_l = user_input.lower()
    
    try:
        # Natural language command handling
        if any(keyword in input_l for keyword in ["histogram", "distribution", "frequency"]):
            fig, col_name = get_visualization(df, user_input, "histogram")
            return (f"Here's the histogram for `{col_name}`", fig)
        
        elif any(keyword in input_l for keyword in ["boxplot", "outliers", "quartiles"]):
            fig, col_name = get_visualization(df, user_input, "boxplot")
            return (f"Here's the boxplot for `{col_name}`", fig)
        
        elif any(keyword in input_l for keyword in ["scatter", "relationship", "correlation"]):
            fig, col_name = get_visualization(df, user_input, "scatter")
            return (f"Here's the scatter plot for `{col_name}`", fig)
        
        elif any(keyword in input_l for keyword in ["bar", "categorical", "count"]):
            fig, col_name = get_visualization(df, user_input, "bar")
            return (f"Here's the bar chart for `{col_name}`", fig)
        
        elif any(keyword in input_l for keyword in ["line", "trend", "time series"]):
            fig, col_name = get_visualization(df, user_input, "line")
            return (f"Here's the line chart for `{col_name}`", fig)
        
        elif any(keyword in input_l for keyword in ["summarize", "describe", "overview"]):
            st.dataframe(get_summary(df))
            return "Here's a statistical summary of your dataset."
        
        elif any(keyword in input_l for keyword in ["missing", "null", "na"]):
            st.write(get_missing_report(df))
            return "Here's a missing value report."
        
        elif any(keyword in input_l for keyword in ["eda report", "profile", "data profile"]):
            report_path = generate_profile(df)
            
            # Display report in-app
            with st.expander("📊 Full EDA Report", expanded=True):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=800, scrolling=True)
            
            # Download option
            with open(report_path, "rb") as f:
                st.download_button("📥 Download Full Report", f, file_name="EDA_Report.html")
            
            return "Here's your interactive EDA report. Expand to view or download."
        
        elif any(keyword in input_l for keyword in ["clean", "cleanse", "tidy"]):
            clean_df = clean_data(df.copy())
            export_path = "cleaned_data.csv"
            clean_df.to_csv(export_path, index=False)
            
            with open(export_path, "rb") as f:
                st.download_button("📥 Download Cleaned Data", f, file_name="cleaned_data.csv")
            
            return "Data cleaned and ready for download"
        
        else:
            # For other queries, use LangChain agent
            agent = get_agent(df)
            response = agent.invoke(user_input)['output']
            return response
            
    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
        st.info("💡 Try rephrasing your query or check column names")
        return "I couldn't process your request. Please try again."
