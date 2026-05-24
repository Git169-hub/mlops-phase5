from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
import time
import mlflow

app = FastAPI()

# Sample documents to build index from
docs = [
    Document(page_content="MLOps is the practice of deploying and maintaining ML models in production."),
    Document(page_content="FastAPI is a modern web framework for building APIs with Python."),
    Document(page_content="Docker packages applications and their dependencies into containers."),
    Document(page_content="FAISS is a library for efficient similarity search on dense vectors."),
    Document(page_content="RAG stands for Retrieval Augmented Generation, combining search with LLMs."),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
If you don't know, say "I don't know".

Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QuestionRequest(BaseModel):
    question: str

mlflow.set_experiment("rag-pipeline")

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    with mlflow.start_run():
        # Log inputs
        mlflow.log_param("model", "llama-3.3-70b-versatile")
        mlflow.log_param("top_k", 3)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("question", request.question)

        # Measure latency
        start = time.time()
        answer = chain.invoke(request.question)
        latency = time.time() - start

        # Log metrics
        mlflow.log_metric("response_latency_sec", latency)
        mlflow.log_metric("answer_length_chars", len(answer))

    return {"question": request.question, "answer": answer}