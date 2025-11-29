# Private-Document-Rag-Chatbot
# Private Document AI Chatbot

A 100% local RAG application that lets you upload PDFs, TXT, XML, DOCX and ask questions in natural language.

Completely private — no data ever leaves my laptop.

## Why I built this
I wanted to understand Retrieval-Augmented Generation from scratch without using any paid API. After a lot of trial and error with Windows file locks, Chroma database issues, and Ollama setup, I finally got a clean, dark-mode web app that actually works reliably.

## Features
- Drag & drop file upload
- Supports PDF, TXT, XML, DOCX
- Rebuild knowledge base with one click
- Chat history + download button
- Clear chat button
- Dark mode 

## Tech stack
- Ollama + llama3.2 (runs locally)
- LangChain
- Chroma vector database
- Streamlit frontend

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py

## What I actually learned 

- How embeddings really work under the hood — not just calling `OllamaEmbeddings()`  
- Why chunk size + overlap matters (tried 500/100 → 1000/200 → 1500/300 and saw real differences in answers)  
- How vector similarity search works inside Chroma (and why `k=5` is usually enough)  
- Windows file locking hell: why `shutil.rmtree` fails and how to fix it with `ignore_errors=True` + killing python.exe  
- The difference between `DirectoryLoader` and manual loader per file type  
- Why you must filter out empty documents or Chroma throws `ValueError: Expected non-empty list`  
- How to make Streamlit not hang forever when Ollama isn’t running  
- Turning a 200-line messy script into a clean, reusable, dark-mode web app

Phase 1 complete. Next: fine-tuning my own model.
