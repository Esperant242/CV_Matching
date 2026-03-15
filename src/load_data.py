import pandas as pd
from pathlib import Path

# Chemin absolu basé sur l'emplacement de ce fichier
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "Preprocessed_Data.txt"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le fichier CSV et retourne un DataFrame."""
    df = pd.read_csv(path)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Affiche les informations clés du dataset."""

    print("=" * 60)
    print("SHAPE")
    print(f"  {df.shape[0]} lignes  |  {df.shape[1]} colonnes")

    print("\n" + "=" * 60)
    print("COLONNES")
    print(f"  {list(df.columns)}")

    print("\n" + "=" * 60)
    print("5 PREMIERES LIGNES")
    print(df.head())

    print("\n" + "=" * 60)
    print("VALEURS MANQUANTES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  Aucune valeur manquante.")
    else:
        print(missing[missing > 0].to_string())

    print("\n" + "=" * 60)
    print("CATEGORIES UNIQUES")
    n_categories = df["Category"].nunique()
    print(f"  {n_categories} catégories distinctes")

    print("\n" + "=" * 60)
    print("DISTRIBUTION DES CATEGORIES")
    distribution = df["Category"].value_counts()
    print(distribution.to_string())

    print("\n" + "=" * 60)
    print("3 EXEMPLES DE CV (tronqués à 500 caractères)")
    samples = df[["Category", "Text"]].sample(n=min(3, len(df)), random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), start=1):
        print(f"\n--- Exemple {i} | Catégorie : {row['Category']} ---")
        print(row["Text"][:500])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    df = load_dataset()
    inspect_dataset(df)
