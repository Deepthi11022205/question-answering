================================================
  NLP Question Answering System
  Flask + HuggingFace DistilBERT
================================================

SETUP INSTRUCTIONS (Do this once):
------------------------------------
1. Open VS Code
2. Open Terminal in VS Code (Ctrl + `)
3. Navigate to this folder:
      cd nlp-qa-system

4. Create a virtual environment:
      python -m venv venv

5. Activate it:
   Windows:   venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

6. Install dependencies:
      pip install -r requirements.txt
   
   NOTE: First install may take 5-10 minutes
   (downloads the DistilBERT model ~250MB)

RUN THE PROJECT:
-----------------
1. Make sure venv is activated (see step 5 above)
2. Run:
      python app.py

3. Open browser and go to:
      http://127.0.0.1:5000

HOW TO USE:
------------
MODE 1 - General Q&A:
   - Just type any question and click "Ask Question"
   - Uses built-in general knowledge context

MODE 2 - Context-Based Q&A:
   - Click "Context-Based Q&A" tab
   - Paste any paragraph of text
   - Ask a question about that text
   - Model extracts the answer from your text

EXAMPLE QUESTIONS TO TRY:
--------------------------
General Tab:
   - Who created Python?
   - What is machine learning?
   - Who founded Facebook?
   - What is NLP?

Context Tab:
   1. Paste a Wikipedia paragraph about any topic
   2. Ask: "What is the main purpose?"
   3. Ask: "When was it founded?"
   4. Ask: "Who is the founder?"

PROJECT FILES:
--------------
app.py              → Flask backend + NLP model
templates/index.html → Web UI
requirements.txt    → Dependencies
README.txt          → This file

TECH STACK:
-----------
- Python 3.9+
- Flask 3.0
- HuggingFace Transformers
- DistilBERT (distilbert-base-cased-distilled-squad)
- Trained on SQuAD2 dataset (100k+ Q&A pairs)

================================================
