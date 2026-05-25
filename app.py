from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time

app = FastAPI()

# Simple document store - no heavy ML model needed
docs = [
    "MLOps is the practice of deploying and maintaining ML models in production.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Docker packages applications and their dependencies into containers.",
    "FAISS is a library for efficient similarity search on dense vectors.",
    "RAG stands for Retrieval Augmented Generation, combining search with LLMs.",
]

def simple_retrieve(question: str, k: int = 3):
    question_words = question.lower().split()
    scored = []
    for doc in docs:
        score = sum(1 for word in question_words if word in doc.lower())
        scored.append((score, doc))
    scored.sort(reverse=True)
    return [doc for _, doc in scored[:k]]

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    context = "\n\n".join(simple_retrieve(request.question))
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": request.question})
    latency = time.time() - start

    return {
        "question": request.question,
        "answer": answer.content,
        "latency_sec": round(latency, 2)
    }