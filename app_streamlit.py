import streamlit as st
import requests

st.set_page_config(page_title="QA Chatbot", layout="centered")

st.title("🤖 QA Chatbot")

# HuggingFace API
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Context input (top)
context = st.text_area("📄 Enter context:", height=150)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
question = st.chat_input("Ask a question...")

if question:
    if not context:
        st.warning("⚠️ Please enter context first!")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        # Get answer
        result = query({
            "inputs": {
                "question": question,
                "context": context
            }
        })

        answer = result.get("answer", "❌ No answer found")

        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)
