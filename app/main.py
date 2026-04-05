from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import run_rag

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask(query: Query):
    return run_rag(query.question)