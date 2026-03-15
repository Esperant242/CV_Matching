from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list[Document]) -> list[Document]:
    """Découpe les documents en chunks avec chevauchement.

    Utilise RecursiveCharacterTextSplitter pour respecter les séparateurs
    naturels (paragraphes, phrases) avant de couper arbitrairement.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[splitter] {len(documents)} documents → {len(chunks)} chunks")
    return chunks
