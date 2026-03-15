import pandas as pd
from langchain.schema import Document
from config import DATA_RAW_DIR
import os


def load_cvs_from_csv(filename: str) -> pd.DataFrame:
    """Charge un fichier CSV de CVs depuis data/raw/."""
    path = os.path.join(DATA_RAW_DIR, filename)
    df = pd.read_csv(path)
    print(f"[loader] {len(df)} CVs chargés depuis {path}")
    return df


def dataframe_to_documents(df: pd.DataFrame, text_col: str, metadata_cols: list[str]) -> list[Document]:
    """Convertit un DataFrame en liste de Documents LangChain.

    Args:
        df: DataFrame source
        text_col: colonne contenant le texte principal du CV
        metadata_cols: colonnes à inclure en métadonnées (ex: nom, poste, email)
    """
    documents = []
    for _, row in df.iterrows():
        metadata = {col: row[col] for col in metadata_cols if col in row}
        doc = Document(page_content=str(row[text_col]), metadata=metadata)
        documents.append(doc)
    print(f"[loader] {len(documents)} documents créés")
    return documents
