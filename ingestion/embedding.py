
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")