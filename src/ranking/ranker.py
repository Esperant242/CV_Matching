from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage
from config import OPENAI_API_KEY, OPENAI_LLM_MODEL

RANKING_PROMPT = """Tu es un expert en recrutement.

Voici une offre d'emploi :
{job_offer}

Voici un extrait de CV :
{cv_chunk}

Évalue la compatibilité de ce CV avec l'offre.
Réponds en JSON avec ce format exact :
{{
  "score": <entier de 0 à 10>,
  "points_forts": "<liste des compétences qui matchent>",
  "points_faibles": "<ce qui manque>",
  "verdict": "<Excellent match | Bon match | Match partiel | Peu adapté>"
}}
"""


def rank_cv(job_offer: str, cv_chunk: Document) -> dict:
    """Demande au LLM de scorer un chunk de CV par rapport à une offre."""
    llm = ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )
    prompt = RANKING_PROMPT.format(
        job_offer=job_offer,
        cv_chunk=cv_chunk.page_content,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"metadata": cv_chunk.metadata, "llm_response": response.content}


def rank_all(job_offer: str, chunks: list[Document]) -> list[dict]:
    """Score tous les chunks récupérés et retourne les résultats triés."""
    results = [rank_cv(job_offer, chunk) for chunk in chunks]
    print(f"[ranker] {len(results)} CVs scorés")
    return results
