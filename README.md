# Private Document RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload private documents (PDF, TXT, XML, DOCX) and ask questions in natural language — **completely offline**.

> No APIs. No cloud. No data leaves the local machine.

---

## 🚀 Motivation

This project was built to understand how RAG systems work internally without relying on paid APIs.
The goal was to implement the full pipeline locally using embeddings, vector similarity search, and an LLM running through Ollama.

---

## ✨ Features

* Drag & drop document upload
* Supports PDF, TXT, XML, DOCX formats
* One-click knowledge base rebuild
* Conversational chat interface with history
* Download chat option
* Clear chat functionality
* Dark mode Streamlit UI

---

## 🧠 Tech Stack

* **LLM:** Ollama with Llama 3.2 (local)
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit
* **Embeddings:** Ollama Embeddings
* **Language:** Python

---

## ⚙️ How It Works (RAG Pipeline)

1. Documents are loaded and split into chunks with overlap
2. Text chunks are converted into embeddings
3. Embeddings are stored in Chroma vector database
4. User query is embedded and matched using semantic similarity search
5. Relevant chunks are passed to the LLM as context
6. LLM generates a context-aware answer

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure **Ollama** is running with the Llama model pulled before starting the app.

---

## 📚 Key Learnings

* Practical understanding of how **embeddings** and **vector similarity search** work
* Impact of **chunk size and overlap** on retrieval quality
* Internals of ChromaDB and why top-k retrieval (k=5) is effective
* Handling Windows file locking issues while rebuilding vector stores
* Importance of filtering empty documents to avoid embedding errors
* Building a clean Streamlit app around a complex RAG pipeline

---

## ✅ Outcome

A fully functional, local, private AI chatbot that demonstrates real-world implementation of:

**RAG | Embeddings | Vector Databases | LLM Integration | Semantic Search | Streamlit Deployment**

---

### Next Step

Fine-tuning and extending the system for domain-specific knowledge.


