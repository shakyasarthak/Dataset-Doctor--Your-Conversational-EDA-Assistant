import streamlit as st
import pandas as pd
from chatbot import respond
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Verify Hugging Face API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    st.error("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    st.stop()

st.set_page_config(
    page_title="DataInsight Pro 🤖",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your DataInsight Pro 🤖. Upload a CSV file to start exploring your data!"}]

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} with {len(st.session_state.df)} rows")
        st.caption(f"Columns: {', '.join(st.session_state.df.columns[:5])}...")
    if st.session_state.df is not None:
        with st.expander("📊 Dataset Preview", expanded=True):
            st.dataframe(st.session_state.df.head(3))
            st.caption(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

with col2:
    st.header("Chat with Your Data")
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if isinstance(msg["content"], tuple):
                # Handle different content types
                if len(msg["content"]) == 3 and msg["content"][1] == "report":
                    st.markdown(msg["content"][0])
                    with st.expander("📊 View Full Analysis Report"):
                        st.components.v1.html(msg["content"][2], height=600, scrolling=True)
                elif len(msg["content"]) == 3 and msg["content"][1] == "download":
                    st.markdown(msg["content"][0])
                    st.download_button(
                        label="💾 Download Cleaned Data",
                        data=msg["content"][2],
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.markdown(msg["content"][0])
                    if hasattr(msg["content"][1], 'savefig'):
                        st.pyplot(msg["content"][1])
            elif isinstance(msg["content"], pd.DataFrame):
                st.dataframe(msg["content"])
            else:
                st.markdown(msg["content"])

    if st.session_state.df is not None:
        examples = [
            "Show histogram of Age",
            "Correlation between price and size",
            "Generate EDA report",
            "Clean outliers in Price"
        ]
        prompt = st.chat_input("Ask about your data...", key="chat_input")
        st.caption(f"💡 Try: {', '.join(examples)}")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Analysing..."):
                response = respond(st.session_state.df, prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.info("💡 Upload a CSV file to start chatting with your data")

with st.sidebar:
    st.header("Data Tools")
    if st.session_state.df is not None:
        if st.button("📊 Generate Full EDA Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                from eda_engine import generate_profile
                report_path = generate_profile(st.session_state.df)
                with open(report_path, "rb") as f:
                    st.download_button(
                        "💾 Save Report As...",
                        f,
                        file_name="eda_report.html",
                        key="eda_report"
                    )
        if st.button("🧹 Auto-Clean Data", use_container_width=True):
            from eda_engine import clean_data
            cleaned_df = clean_data(st.session_state.df.copy())
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Save Cleaned Data As...",
                csv,
                file_name="cleaned_data.csv",
                key="cleaned_data"
            )
        if st.button("📈 Show Data Summary", use_container_width=True):
            from eda_engine import get_summary
            st.dataframe(get_summary(st.session_state.df))
    st.header("Conversation Tools")
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset! How can I help?"}]
        st.rerun()
    if st.button("📥 Export Chat", use_container_width=True):
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            "💾 Save Chat History As...",
            chat_history,
            file_name="chat_history.txt",
            key="chat_history"
        )
    st.header("Security Notice")
    st.warning("""
    This application executes Python code for data analysis.
    For your security:
    - Only upload data from trusted sources
    - Review all code suggestions before execution
    - Avoid sensitive data in analyses
    """)
    st.divider()
    feedback = st.radio("Rate this response:", ("👍", "👎"), index=None, key="feedback")
    if feedback:
        st.success("Thanks for your feedback!")
