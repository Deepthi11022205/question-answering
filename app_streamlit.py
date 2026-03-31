import streamlit as st
import requests

st.title("🤖 QA Chatbot")

# Default context
default_context = """India is a country in South Asia. 
Its capital is New Delhi. 
It is the 7th largest country in the world."""

context = st.text_area("📄 Enter context:", value=default_context, height=150)

# API
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

def query(payload):
    response = requests.post(API_URL, json=payload)
    data = response.json()

    if isinstance(data, dict) and "answer" in data:
        return data
    else:
        return {"answer": "⚠️ Model not ready / Try again"}

# Input question
question = st.text_input("❓ Ask a question:").strip().capitalize()

if st.button("Get Answer"):
    if question:
        result = query({
            "inputs": {
                "question": question,
                "context": context
            }
        })

        answer = result.get("answer", "No answer found")

        st.success(f"Answer: {answer}")
    else:
        st.warning("Please enter a question")
