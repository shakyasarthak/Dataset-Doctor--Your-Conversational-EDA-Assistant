import streamlit as st
import pandas as pd
from chatbot import respond
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify Hugging Face API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    st.error("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    st.stop()

# Premium UI configuration
st.set_page_config(
    page_title="DataInsight Pro 🤖",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --background: #0f172a;
    --card: #1e293b;
    --text: #f1f5f9;
}

.stApp {
    background: linear-gradient(135deg, var(--background) 0%, #020617 100%);
    color: var(--text);
}

.stChatMessage {
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.stButton>button {
    background: linear-gradient(45deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton>button {
    background: linear-gradient(45deg, #10b981 0%, #059669 100%);
}

.stHeader {
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your DataInsight Pro 🤖. Upload a CSV file to start exploring your data!"}]

# Two-column layout
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
    
    # Display chat history
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if isinstance(msg["content"], tuple):
                # Handle different content types
                if len(msg["content"]) == 3 and msg["content"][1] == "report":
                    st.markdown(msg["content"][0])
                    with st.expander("📊 View Full Analysis Report"):
                        st.components.v1.html(msg["content"][2], height=600, scrolling=True)
                else:
                    st.markdown(msg["content"][0])
                    if msg["content"][1]:
                        st.pyplot(msg["content"][1])
            elif isinstance(msg["content"], pd.DataFrame):
                st.dataframe(msg["content"])
            else:
                st.markdown(msg["content"])
    
    # Chat input
    if st.session_state.df is not None:
        examples = [
            "Show histogram of Age",
            "Find missing values",
            "Generate EDA report",
            "Clean outliers in Price"
        ]
        
        prompt = st.chat_input("Ask about your data...", key="chat_input")
        st.caption(f"💡 Try: {', '.join(examples)}")
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("Analysing..."):
                response = respond(st.session_state.df, prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    else:
        st.info("💡 Upload a CSV file to start chatting with your data")

# Sidebar tools
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
        
        if st.button("❓ Help Examples", use_container_width=True):
            st.info("Try these commands:")
            st.code("""
- "Show distribution of Age"
- "Correlation between Height and Weight"
- "Missing values report"
- "Clean outliers in Price"
- "Generate EDA report"
""")
    
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
    
    # Feedback mechanism
    st.divider()
    feedback = st.radio("Rate this response:", ("👍", "👎"), index=None, key="feedback")
    if feedback:
        st.success("Thanks for your feedback!")
