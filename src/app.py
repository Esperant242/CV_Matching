import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rank_with_llm import rank_candidates_with_llm


def display_results(ranking: list[dict]) -> None:
    """Affiche le classement final des candidats."""
    print("\n" + "=" * 60)
    print(f"  TOP {len(ranking)} CANDIDATS")
    print("=" * 60)

    for rank, candidate in enumerate(ranking, start=1):
        print(f"\n#{rank}  ID: {candidate['candidate_id']}  |  {candidate['category']}")
        print(f"    Score      : {candidate['score_sur_20']} / 20")
        print(f"    Décision   : {candidate['decision']}")
        print(f"    Résumé     : {candidate['justification']}")
        print("-" * 60)


def run(job_description: str, top_k: int = 5) -> None:
    """Pipeline complet : retrieval → scoring LLM → affichage."""
    print(f"\nJob description reçue ({len(job_description)} chars)")
    print(f"Recherche des {top_k} meilleurs profils...\n")

    ranking = rank_candidates_with_llm(job_description, k=top_k)
    display_results(ranking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CV Matching — MVP")
    parser.add_argument("--query", type=str, help="Job description à matcher")
    parser.add_argument("--k", type=int, default=5, help="Nombre de candidats (défaut: 5)")
    args = parser.parse_args()

    if args.query:
        job_description = args.query
    else:
        # Job description par défaut pour test rapide
        job_description = """
        We are looking for a Senior Python Developer with experience in backend development,
        REST APIs, and cloud infrastructure (AWS or GCP). The candidate should be proficient
        in Django or FastAPI, SQL databases, and have a good understanding of CI/CD pipelines.
        Experience with machine learning or data engineering is a plus.
        """

    run(job_description, top_k=args.k)
