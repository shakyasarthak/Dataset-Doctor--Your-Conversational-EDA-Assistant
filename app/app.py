import streamlit as st
import pandas as pd
from chatbot import respond, generate_dynamic_suggestions, generate_explanation, generate_recommendations
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Load Together API key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    st.error("Together API key not found. Please set the TOGETHER_API_KEY environment variable.")
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
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""

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

        suggestions = generate_dynamic_suggestions(st.session_state.df)
        with st.expander("💡 Suggested Queries"):
            for suggestion in suggestions:
                st.code(suggestion)

with col2:
    st.header("Chat with Your Data")
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if isinstance(msg["content"], tuple):
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
                    if isinstance(msg["content"][1], plt.Figure):
                        st.pyplot(msg["content"][1])
                    elif hasattr(msg["content"][1], 'to_html'):
                        st.plotly_chart(msg["content"][1])
            elif isinstance(msg["content"], pd.DataFrame):
                st.dataframe(msg["content"])
            else:
                st.markdown(msg["content"])

        if msg["role"] == "assistant" and idx > 0:
            if st.button("Explain this step-by-step", key=f"explain_{idx}"):
                last_query = st.session_state.messages[idx-1]["content"]
                last_response = msg["content"]
                explanation = generate_explanation(
                    st.session_state.df,
                    last_query,
                    last_response
                )
                st.session_state.messages.append({"role": "assistant", "content": explanation})
                st.session_state.last_action = ("explanation", last_query, last_response)
                st.rerun()

        if msg["role"] == "assistant" and idx > 0:
            recommendations = generate_recommendations(
                st.session_state.df,
                st.session_state.messages[idx-1]["content"],
                msg["content"]
            )
            with st.expander("🔎 Proactive Recommendations"):
                for rec in recommendations.split('\n'):
                    if rec.strip():
                        st.markdown(f"- {rec.strip()}")

    if st.session_state.df is not None:
        examples = [
            "Show histogram of Age",
            "Correlation between price and size",
            "Generate EDA report",
            "Clean outliers in Price",
            "Suggest feature engineering"
        ]
        prompt = st.chat_input("Ask about your data...", key="chat_input")
        st.caption(f"💡 Try: {', '.join(examples[:3])}")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.last_user_query = prompt
            with st.spinner("Analysing..."):
                response = respond(st.session_state.df, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_action = ("response", prompt, response)
            st.rerun()
    else:
        st.info("💡 Upload a CSV file to start chatting with your data")

with st.sidebar:
    st.header("Data Tools")
    if st.session_state.df is not None:
        if st.button("📊 Generate Full EDA Report", use_container_width=True):
            from eda_engine import generate_profile
            with st.spinner("Generating comprehensive report..."):
                report_path = generate_profile(st.session_state.df)
                with open(report_path, "r", encoding="utf-8") as f:
                    report_html = f.read()
                summary = (
                    f"🔍 Here's my analysis of your dataset:\n\n"
                    f"- Found {len(st.session_state.df.columns)} features and {len(st.session_state.df)} records\n"
                    f"Full EDA report is ready below 👇"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (summary, "report", report_html)
                })
                st.rerun()

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
            summary = get_summary(st.session_state.df)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.rerun()

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
        st.session_state.feedback = feedback
        st.success("Thanks for your feedback!")
        if "feedback_log" not in st.session_state:
            st.session_state.feedback_log = []
        st.session_state.feedback_log.append({
            "query": st.session_state.messages[-2]["content"] if len(st.session_state.messages) > 1 else "",
            "response": st.session_state.messages[-1]["content"] if st.session_state.messages else "",
            "feedback": feedback
        })
