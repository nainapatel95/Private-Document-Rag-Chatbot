import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredXMLLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from datetime import datetime

# Add this function at the top
def check_ollama():
    try:
        from ollama import Client
        Client().list()  # quick ping
        return True
    except:
        return False

# Then in sidebar, right after the Rebuild button:
if not check_ollama():
    st.error("Ollama is not running! Open the Ollama app first.")
    st.stop()
# --------------------------- UI SETUP --------------------------
st.set_page_config(page_title="My Private Document AI", page_icon="robot", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: white;}
    .stButton>button {background-color: #ff4b4b; color: white; border-radius: 8px;}
    .stDownloadButton>button {background-color: #1f77b4; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("Private Document AI Chatbot")
st.markdown("Chat history downloadable • 100% local with Ollama + LangChain + Streamlit")

DATA_FOLDER = "data"
DB_FOLDER = "vectorstore"

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Drop PDFs, TXT, XML, DOCX...",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'xml', 'docx']
    )

    if st.button("Save Uploaded Files", type="primary"):
        if uploaded_files:
            os.makedirs(DATA_FOLDER, exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to data folder")
        else:
            st.warning("Please upload at least one file")

    if st.button("Rebuild Knowledge Base"):
        if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
            st.error("No files in data folder!")
        else:
            with st.spinner("Loading & processing documents..."):
                # Auto-select loader based on file extension
                def get_loader(filepath):
                    ext = os.path.splitext(filepath)[1].lower()
                    if ext == ".pdf": return PyPDFLoader(filepath)
                    if ext == ".txt": return TextLoader(filepath)
                    if ext in [".xml", ".html"]: return UnstructuredXMLLoader(filepath)
                    if ext in [".docx", ".doc"]: return UnstructuredWordDocumentLoader(filepath)
                    return None

                docs = []
                for file in os.listdir(DATA_FOLDER):
                    path = os.path.join(DATA_FOLDER, file)
                    loader = get_loader(path)
                    if loader:
                        docs.extend(loader.load())

                # Remove empty documents
                docs = [d for d in docs if d.page_content.strip()]

                if not docs:
                    st.error("No text found in any file. Try simpler PDFs/TXT.")
                    st.stop()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                if not chunks:
                    st.error("Failed to create chunks.")
                    st.stop()

                # Create fresh vector DB
                if os.path.exists(DB_FOLDER):
                    shutil.rmtree(DB_FOLDER, ignore_errors=True)  # Safe delete on Windows

                embeddings = OllamaEmbeddings(model="llama3.2")
                Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)

            st.success(f"Knowledge base ready! {len(docs)} docs → {len(chunks)} chunks")

# --------------------------- CHAT INTERFACE ---------------------------
if os.path.exists(DB_FOLDER) and os.listdir(DB_FOLDER):
    embeddings = OllamaEmbeddings(model="llama3.2")
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    template = """Answer using only this context:\n\n{context}\n\nQuestion: {question}\nAnswer clearly:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2", temperature=0.1)

    chain = ({"context": retriever, "question": RunnablePassthrough()}
             | prompt | llm | StrOutputParser())

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            st.download_button(
                "Download Chat",
                chat_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # User input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Upload files → Save → Rebuild Knowledge Base → Start chatting!")

st.caption("|Ollama + LangChain + Streamlit ")