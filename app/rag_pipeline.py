from app.retriever import get_retriever
from app.generator import get_llm

retriever = get_retriever()
llm = get_llm()

def run_rag(query: str):
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer strictly from the context below.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in docs]
    }