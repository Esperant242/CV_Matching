import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_PATH = "D:/RAG_vectorstore/faiss_index"


def retrieve_top_matches(query: str, k: int = 5) -> list:
    """Charge l'index FAISS et retourne les k chunks les plus proches de la query."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vectorstore.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    job_description = """
    We are looking for an HR Recruiter / Talent Acquisition Specialist.
    The candidate should have experience in full-cycle recruiting, sourcing candidates,
    conducting interviews, and managing job postings on LinkedIn and other platforms.
    Strong skills in candidate screening, onboarding, HR policies, and ATS tools required.
    Experience in employer branding and workforce planning is a plus.
    """

    print("Recherche des CVs les plus proches...\n")
    results = retrieve_top_matches(job_description, k=5)

    for i, doc in enumerate(results, start=1):
        print(f"--- Résultat {i} ---")
        print(f"resume_id : {doc.metadata['resume_id']}")
        print(f"category  : {doc.metadata['category']}")
        print(f"extrait   : {doc.page_content[:500]}")
        print()
