from app.retriever import get_retriever
from app.generator import get_llm

retriever = get_retriever()
llm = get_llm()

def run_rag(query: str):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the context below to answer the question.
    If partial information is available, try to infer the best possible answer.
    Only say "I don't know" if nothing relevant is found.

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