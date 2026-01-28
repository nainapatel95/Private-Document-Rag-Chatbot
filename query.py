from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the vector database
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Prompt template
template = """
You are an assistant that answers based ONLY on the given context.
If the answer is not in the context, say: 'Information not found in documents.'

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Load local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
)

# Full RAG chain
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("\nYour RAG Chatbot is Ready! Ask anything about your documents.")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower().strip() == "exit":
        print("Goodbye!")
        break

    print("Thinking...")
    response = chain.invoke(query)
    print(f"Bot: {response}\n")
