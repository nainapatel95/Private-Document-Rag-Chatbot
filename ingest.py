from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

print("Loading ALL files from 'data' folder (PDF, TXT, XML, DOCX, etc.)...")

# Correct DirectoryLoader usage – loader_cls must be a SINGLE class, not a dict
loader = DirectoryLoader(
    'data/',
    glob="**/*.*",
    show_progress=True,
    silent_errors=True
)

docs = loader.load()
print(f"Successfully loaded {len(docs)} documents")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

print("Building vector database (this may take 20–60 seconds)...")
embeddings = OllamaEmbeddings(model="llama3.2")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore"
)

print("SUCCESS! Your knowledge base now supports PDF, TXT, XML, DOCX, and HTML!")
print("Run → python query.py")
