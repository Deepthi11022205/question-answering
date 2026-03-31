import streamlit as st
from transformers import pipeline

st.title("Question Answering System")

@st.cache_resource
def load_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        framework="pt"   # force lightweight mode
    )

qa_pipeline = load_model()

context = st.text_area("Enter context:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.write("Answer:", result["answer"])
    else:
        st.warning("Please enter both context and question")
