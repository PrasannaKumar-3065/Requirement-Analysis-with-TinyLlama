Requirement Analysis AI
=================================

This project provides an AI-powered pipeline for requirement analysis using Retrieval-Augmented Generation (RAG) and transformer models. It supports document chunking, semantic search, answer generation, and evaluation with multiple metrics.

Features
--------
- Document chunking and semantic embedding
- Retrieval and reranking of relevant information
- LLM-based answer generation
- Evaluation harness for accuracy, F1, coverage, and hallucination detection
- Results saved for further analysis

Setup
-----
1. **Clone the repository**
2. **Install dependencies**::

    pip install -r requirements.txt

3. **Prepare your documents**
   - Place `.txt` files in ``advanced_RAG/docs/``
   - Optionally, create ``eval_data.json`` for custom evaluation questions

Usage
-----
- **Run evaluation**::

    python advanced_RAG/eval_rag.py

- **View results**
  - Results are saved in ``eval_results_module8.json``
  - Analyze results using Python, pandas, or visualization tools

File Overview
-------------
- ``advanced_RAG/advanced_rag.py``: Main RAG pipeline and document processing
- ``advanced_RAG/eval_rag.py``: Evaluation harness and metrics
- ``advanced_RAG/docs/``: Folder for input documents
- ``eval_data.json``: Optional custom evaluation questions

Customization
-------------
- Update models and parameters in ``advanced_rag.py``
- Add or modify evaluation questions in ``eval_data.json``

Dashboard & Analysis
--------------------
You can use Jupyter Notebook, pandas, or Streamlit to analyze and visualize your model’s performance over time.

License
-------
MIT
