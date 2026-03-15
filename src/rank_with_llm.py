import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from retrieve import retrieve_top_matches

load_dotenv()

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0


PROMPT_TEMPLATE = """
You are an expert HR recruiter. Evaluate the following candidate CV extract against the job description.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV (resume_id: {resume_id}, category: {category}):
{cv_text}

SCORING RULES — you MUST follow these intervals strictly:

| Score  | Decision         | Criteria                                                                 |
|--------|------------------|--------------------------------------------------------------------------|
| 17-20  | Excellent match  | Meets almost all requirements. Strong skills, relevant experience, right domain. |
| 13-16  | Good match       | Meets most requirements. Some gaps on secondary skills or experience level.      |
| 8-12   | Partial match    | Meets some requirements. Noticeable gaps on key skills or domain.                |
| 0-7    | Not suitable     | Does not meet the main requirements. Wrong domain or missing core skills.        |

Rules:
- Never score above 16 unless the candidate clearly meets ALL major requirements.
- Never score below 8 unless the candidate is clearly in the wrong domain.
- The decision label MUST match the score interval above. No exceptions.
- Be consistent: two similar CVs for the same job must receive similar scores.

Return ONLY a raw JSON object. No markdown, no code block, no explanation. Just the JSON.
{{
  "candidate_id": {resume_id},
  "category": "{category}",
  "score_sur_20": <integer from 0 to 20>,
  "decision": "<Excellent match | Good match | Partial match | Not suitable>",
  "justification": "<2-3 sentences explaining the score based on the criteria above>"
}}
"""


def retrieve_candidates(job_description: str, k: int = 5) -> dict:
    """Récupère les top k chunks et les groupe par resume_id."""
    chunks = retrieve_top_matches(job_description, k=k)

    candidates = {}
    for chunk in chunks:
        rid = chunk.metadata["resume_id"]
        if rid not in candidates:
            candidates[rid] = {
                "resume_id": rid,
                "category": chunk.metadata["category"],
                "cv_text": chunk.page_content,
            }
        else:
            candidates[rid]["cv_text"] += " " + chunk.page_content

    # Tri déterministe par resume_id pour stabiliser l'ordre d'appel LLM
    return dict(sorted(candidates.items()))



def build_prompt(job_description: str, candidate: dict) -> str:
    """Construit le prompt pour un candidat donné."""
    return PROMPT_TEMPLATE.format(
        job_description=job_description.strip(),
        resume_id=candidate["resume_id"],
        category=candidate["category"],
        cv_text=candidate["cv_text"][:2000],
    )


def rank_candidates_with_llm(job_description: str, k: int = 5) -> list[dict]:
    """Récupère les candidats, les score via LLM et retourne un classement."""
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=42,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    candidates = retrieve_candidates(job_description, k=k)
    results = []

    for candidate in candidates.values():
        prompt = build_prompt(job_description, candidate)
        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            # Nettoie les balises markdown que le LLM ajoute parfois (```json ... ```)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scored = json.loads(raw.strip())
        except json.JSONDecodeError:
            print(f"[warn] JSON invalide pour resume_id={candidate['resume_id']}: {response.content[:100]}")
            scored = {
                "candidate_id": candidate["resume_id"],
                "category": candidate["category"],
                "score_sur_20": 0,
                "decision": "Parse error",
                "justification": response.content,
            }

        results.append(scored)

    results.sort(key=lambda x: x.get("score_sur_20", 0), reverse=True)
    return results


if __name__ == "__main__":
    job_description = """
    We are looking for an HR Recruiter / Talent Acquisition Specialist.
    The candidate should have experience in full-cycle recruiting, sourcing candidates,
    conducting interviews, and managing job postings on LinkedIn and other platforms.
    Strong skills in candidate screening, onboarding, HR policies, and ATS tools required.
    Experience in employer branding and workforce planning is a plus.
    """

    print("Scoring LLM en cours...\n")
    ranking = rank_candidates_with_llm(job_description, k=5)

    print("=" * 60)
    print("CLASSEMENT FINAL")
    print("=" * 60)
    for rank, candidate in enumerate(ranking, start=1):
        print(f"\n#{rank} — resume_id: {candidate['candidate_id']} | {candidate['category']}")
        print(f"Score       : {candidate['score_sur_20']} / 20")
        print(f"Décision    : {candidate['decision']}")
        print(f"Justification : {candidate['justification']}")
