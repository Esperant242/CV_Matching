import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_data import load_dataset

# --- Paramètres de chunking ---
# Basés sur l'analyse du dataset :
#   - longueur moyenne d'un CV : ~3 986 chars
#   - objectif : 3 à 5 chunks par CV pour capturer chaque section
#   - chunk_size = 1000 → ~250 tokens, optimal pour OpenAI embeddings
#   - chunk_overlap = 150 → 15% de recouvrement pour ne pas couper le contexte
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Convertit un DataFrame de CVs en liste de Documents LangChain.

    Chaque document contient :
      - page_content : le texte brut du CV
      - metadata     : resume_id et category
    Les lignes avec Text vide sont ignorées.
    """
    documents = []

    for idx, row in df.iterrows():
        text = str(row["Text"]).strip()
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "resume_id": idx,
                "category": row["Category"],
            },
        )
        documents.append(doc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Découpe une liste de Documents en chunks.

    Utilise RecursiveCharacterTextSplitter qui essaie de couper
    sur les séparateurs naturels (espaces, ponctuation) avant
    de couper arbitrairement.

    Les metadata d'origine (resume_id, category) sont conservées
    sur chaque chunk produit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Chargement
    df = load_dataset()

    # Étape 1 : construction des documents
    documents = build_documents(df)
    print(f"Documents avant chunking : {len(documents)}")

    # Étape 2 : chunking
    chunks = split_documents(documents)
    print(f"Chunks après chunking    : {len(chunks)}")

    # Taille moyenne
    taille_moy = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"Taille moyenne par chunk : ~{taille_moy} chars")

    # Exemple de chunk
    print("\n--- Exemple de chunk (400 premiers chars) ---")
    print(chunks[10].page_content[:400])

    print("\n--- Metadata du chunk ---")
    print(chunks[10].metadata)
