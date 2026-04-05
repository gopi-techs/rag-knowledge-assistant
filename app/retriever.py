import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
    return db.as_retriever(search_kwargs={"k": 6})