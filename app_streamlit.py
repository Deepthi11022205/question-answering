import streamlit as st
import requests
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="QA Chatbot", page_icon="🤖")

st.title("🤖 QA Chatbot")

# -------------------------------
# DEFAULT CONTEXT
# -------------------------------
default_context = """India is a country in South Asia.
Its capital is New Delhi.
It is the 7th largest country in the world."""

context = st.text_area("📄 Enter context:", value=default_context, height=150)
question = st.text_input("❓ Ask a question:")

# -------------------------------
# HUGGING FACE API
# -------------------------------
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

def query(payload):
    MAX_RETRIES = 5

    for i in range(MAX_RETRIES):
        response = requests.post(API_URL, json=payload)
        data = response.json()

        # ✅ If model gives answer
        if isinstance(data, dict) and "answer" in data:
            return data

        # ⏳ If model is loading
        if "error" in data:
            time.sleep(3)

    return {"answer": "⚠️ Model took too long to load. Try again."}

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("Get Answer"):
    if context and question:
        with st.spinner("🤖 Thinking... please wait"):
            result = query({
                "inputs": {
                    "question": question,
                    "context": context
                }
            })

        st.success(f"Answer: {result['answer']}")
    else:
        st.warning("Please enter both context and question")
