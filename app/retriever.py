from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    return db.as_retriever(search_kwargs={"k": 4})