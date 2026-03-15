import re
import pandas as pd


def clean_text(text: str) -> str:
    """Nettoie un texte brut : espaces, caractères spéciaux, lignes vides."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)          # espaces multiples
    text = re.sub(r'[^\w\s\.,;:()\-@]', '', text)  # caractères parasites
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Applique le nettoyage sur la colonne texte du DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df = df[df[text_col].str.len() > 50]  # supprime les entrées trop courtes
    df = df.drop_duplicates(subset=[text_col])
    print(f"[cleaner] {len(df)} CVs après nettoyage")
    return df
