from ingestion.loader import load_documents
from ingestion.chunking import split_documents
from ingestion.embedding import create_vector_store

def main():
    print("📥 Loading documents...")
    docs = load_documents("data/sample.pdf")

    print("✂️ Splitting documents...")
    chunks = split_documents(docs)

    print("🧠 Creating embeddings and FAISS index...")
    create_vector_store(chunks)

    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    main()