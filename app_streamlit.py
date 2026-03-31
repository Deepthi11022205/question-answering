import streamlit as st
import requests

st.title("Question Answering System")

# HuggingFace API (free)
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

context = st.text_area("Enter context:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        result = query({
            "inputs": {
                "question": question,
                "context": context
            }
        })
        st.write("Answer:", result.get("answer", "No answer found"))
    else:
        st.warning("Please enter both context and question")
