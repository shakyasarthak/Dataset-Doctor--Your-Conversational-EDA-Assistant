# streamlit app entry point

# import streamlit as st
# import pandas as pd
# from chatbot import respond

# from dotenv import load_dotenv
# import os

# load_dotenv()  # Loads variables from .env

# api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if not api_key:
#     print("Hugging Face API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
#     exit(1)

# st.set_page_config(page_title="Dataset Doctor 🤖", layout="wide")
# st.title("🩺 Dataset Doctor - Chat with Your Data")

# # Session State for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Upload CSV
# uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success("✅ File uploaded and loaded.")

#     # Chat interface
#     st.markdown("## 💬 Chat with your dataset")
#     prompt = st.chat_input("Type a command like 'Show histogram of Age'")

#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             # Get response first
#             response = respond(df, prompt)
            
#             # Then handle display
#             if isinstance(response, pd.DataFrame):
#                 st.dataframe(response)
#             elif isinstance(response, str) and "```" in response:
#                 st.code(response.split("```")[1])
#             else:
#                 st.markdown(response)
            
#             # Now append to session state
#             st.session_state.messages.append({"role": "assistant", "content": response})

#     # Display chat history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

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

# Configure page
st.set_page_config(page_title="Dataset Doctor 🤖", layout="wide")
st.title("🩺 Dataset Doctor - Chat with Your Data")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded and loaded.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Handle different content types
        if isinstance(msg["content"], tuple):
            # Tuple format: (text, visualization)
            st.markdown(msg["content"][0])
            if msg["content"][1] is not None:
                st.pyplot(msg["content"][1])
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your data (e.g., 'Show distribution of Age')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if df is not None:
        # Get response and display in chat
        response = respond(df, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response immediately after user input
        with st.chat_message("assistant"):
            if isinstance(response, tuple):
                st.markdown(response[0])
                if response[1] is not None:
                    st.pyplot(response[1])
            else:
                st.markdown(response)
    else:
        st.warning("Please upload a CSV file first")
