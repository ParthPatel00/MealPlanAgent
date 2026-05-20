"""
Build and persist a ChromaDB-backed LlamaIndex vector store from recipe documents.

Run once after data/processed/recipes_clean.json is generated:
    python -m src.rag.indexer
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.data.preprocessor import load_documents

load_dotenv()

CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
COLLECTION_NAME = "recipes"
# Local model, no API key required, ~90 MB download on first use
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


DEFAULT_EMBED_MODEL = EMBED_MODEL


def _get_embed_model(model_name: str = EMBED_MODEL) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name)


def build_index(
    chroma_path: Path = CHROMA_PATH,
    embed_model: str = EMBED_MODEL,
    collection_name: str = COLLECTION_NAME,
    documents=None,
) -> VectorStoreIndex:
    """Build and persist a VectorStoreIndex from the processed recipe documents."""
    if documents is None:
        print("Loading documents...")
        documents = load_documents()
        print(f"  {len(documents)} documents loaded.")

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Building index with {embed_model} (this may take a few minutes)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_get_embed_model(embed_model),
        show_progress=True,
    )
    print(f"Index built and persisted at {chroma_path}")
    return index


def load_index(
    chroma_path: Path = CHROMA_PATH,
    embed_model: str = EMBED_MODEL,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """Load an existing persisted ChromaDB index."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=_get_embed_model(embed_model),
    )


if __name__ == "__main__":
    build_index()
