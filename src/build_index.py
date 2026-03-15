import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Imports des fonctions déjà écrites
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_dataset
from build_documents import build_documents, split_documents

load_dotenv()

FAISS_INDEX_PATH = "D:/RAG_vectorstore/faiss_index"


def build_faiss_index(chunks):
    """Crée les embeddings et construit l'index FAISS."""
    print("[4/4] Création des embeddings et construction de l'index FAISS...")
    print(f"      Modèle : text-embedding-3-small | Chunks : {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"      Index sauvegardé → {FAISS_INDEX_PATH}")
    return vectorstore


if __name__ == "__main__":
    print("[1/4] Chargement du dataset...")
    df = load_dataset()
    print(f"      {len(df)} CVs chargés")

    print("[2/4] Construction des documents LangChain...")
    documents = build_documents(df)
    print(f"      {len(documents)} documents créés")

    print("[3/4] Chunking...")
    chunks = split_documents(documents)
    print(f"      {len(chunks)} chunks produits")

    vectorstore = build_faiss_index(chunks)
    print("\nIndex FAISS prêt.")
