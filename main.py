"""
Pipeline RAG CV Matching
------------------------
Usage:
    python main.py --build          # Indexer les CVs
    python main.py --query "..."    # Matcher une offre d'emploi
"""

import argparse
from src.ingestion.loader import load_cvs_from_csv, dataframe_to_documents
from src.ingestion.cleaner import clean_dataframe
from src.chunking.splitter import split_documents
from src.vectorstore.store import build_vectorstore, save_vectorstore, load_vectorstore
from src.retrieval.retriever import retrieve_with_scores
from src.ranking.ranker import rank_all


CV_FILE = "cvs.csv"          # fichier dans data/raw/
CV_TEXT_COL = "cv_text"      # colonne texte du CSV
CV_META_COLS = ["name", "email", "position"]  # colonnes métadonnées


def build_index():
    """Charge, nettoie, chunke et indexe les CVs."""
    df = load_cvs_from_csv(CV_FILE)
    df = clean_dataframe(df, text_col=CV_TEXT_COL)
    documents = dataframe_to_documents(df, text_col=CV_TEXT_COL, metadata_cols=CV_META_COLS)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)
    print("\n[main] Index construit et sauvegardé.")


def match_offer(job_offer: str):
    """Charge l'index et trouve les meilleurs CVs pour une offre."""
    vectorstore = load_vectorstore()
    chunks = [doc for doc, _ in retrieve_with_scores(vectorstore, job_offer)]
    results = rank_all(job_offer, chunks)

    print("\n=== RESULTATS ===")
    for r in results:
        print(f"\nCandidat : {r['metadata']}")
        print(r["llm_response"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CV Matching")
    parser.add_argument("--build", action="store_true", help="Construire l'index FAISS")
    parser.add_argument("--query", type=str, help="Offre d'emploi à matcher")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        match_offer(args.query)
    else:
        parser.print_help()
