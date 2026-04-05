from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")