# ============================================================
# NLP Question Answering System
# Backend: Flask + HuggingFace Transformers (DistilBERT)
# ============================================================

from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import time

app = Flask(__name__)

# ---- Load the NLP model once at startup ----
print("⏳ Loading DistilBERT NLP model... (first time may take a minute)")
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)
print("✅ NLP Model loaded successfully!")

# ---- Default context for general questions ----
DEFAULT_CONTEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
the natural intelligence displayed by animals including humans. AI research has been 
defined as the field of study of intelligent agents, which refers to any system that 
perceives its environment and takes actions that maximize its chance of achieving its goals.

Machine learning is a method of data analysis that automates analytical model building. 
It is based on the idea that systems can learn from data, identify patterns and make 
decisions with minimal human intervention. Deep learning is part of a broader family 
of machine learning methods based on artificial neural networks with representation learning.

Natural Language Processing (NLP) is a subfield of linguistics, computer science, and 
artificial intelligence concerned with the interactions between computers and human language, 
in particular how to program computers to process and analyze large amounts of natural 
language data. Python is a high-level, general-purpose programming language. Its design 
philosophy emphasizes code readability. Python was created by Guido van Rossum and was 
first released in 1991. Flask is a micro web framework written in Python.

The Internet is a global system of interconnected computer networks. It carries a vast 
range of information resources and services. The World Wide Web is an information system 
where documents and other web resources are identified by URLs. Google was founded in 1998 
by Larry Page and Sergey Brin. Facebook was founded by Mark Zuckerberg in 2004.
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    context = data.get("context", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    # Use provided context or fall back to default
    if not context:
        context = DEFAULT_CONTEXT

    try:
        start = time.time()
        result = qa_pipeline(question=question, context=context)
        elapsed = round((time.time() - start) * 1000)  # ms

        return jsonify({
            "answer": result["answer"],
            "confidence": round(result["score"] * 100, 2),
            "time_ms": elapsed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "distilbert-base-cased-distilled-squad"})

if __name__ == "__main__":
    print("\n🚀 NLP Q&A System running at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
