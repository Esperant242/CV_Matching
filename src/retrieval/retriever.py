from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import TOP_K


def retrieve(vectorstore: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    """Recherche les k chunks les plus proches de la requête.

    Args:
        vectorstore: index FAISS chargé
        query: texte de l'offre d'emploi
        k: nombre de résultats à retourner
    """
    results = vectorstore.similarity_search(query, k=k)
    print(f"[retriever] {len(results)} chunks récupérés pour la requête")
    return results


def retrieve_with_scores(vectorstore: FAISS, query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Retourne les chunks avec leur score de similarité cosinus."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    for doc, score in results:
        print(f"[retriever] score={score:.4f} | {doc.metadata}")
    return results
