import streamlit as st
from transformers import pipeline

st.title("Question Answering System")

qa_pipeline = pipeline("question-answering")

context = st.text_area("Enter context:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.write("Answer:", result["answer"])
    else:
        st.warning("Please enter both context and question")