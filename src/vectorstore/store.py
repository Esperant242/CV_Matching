from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.vectorstore.embedder import get_embeddings
from config import FAISS_INDEX_PATH


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Construit un index FAISS depuis une liste de chunks."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[store] Index FAISS construit ({len(chunks)} chunks)")
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """Sauvegarde l'index FAISS sur disque."""
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[store] Index sauvegardé → {FAISS_INDEX_PATH}")


def load_vectorstore() -> FAISS:
    """Charge un index FAISS existant depuis le disque."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[store] Index FAISS chargé depuis {FAISS_INDEX_PATH}")
    return vectorstore
