"""
Hybrid retriever: BM25 keyword search + vector cosine similarity.

Merges both result sets (Reciprocal Rank Fusion) and returns top-k
recipes with citation metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]

from src.rag.indexer import load_index

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "10"))


@dataclass
class RecipeHit:
    recipe_id: int
    name: str
    score: float
    minutes: int
    tags: list[str]
    ingredients: list[str]
    nutrition: dict
    text: str


def _reciprocal_rank_fusion(
    vector_hits: list, bm25_hits: list, k: int = 60
) -> list[tuple[str, float]]:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, node in enumerate(vector_hits):
        doc_id = node.node.node_id
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, node in enumerate(bm25_hits):
        doc_id = node.node.node_id
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """BM25 + vector retrieval with RRF merging."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._index = None
        self._vector_retriever = None
        self._bm25_retriever = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        self._index = load_index()
        self._vector_retriever = VectorIndexRetriever(
            index=self._index, similarity_top_k=self.top_k * 2
        )
        # BM25 needs text nodes. ChromaDB only stores embeddings, not the original
        # text, so we rebuild nodes directly from the processed JSON file.
        from llama_index.core.schema import TextNode
        from src.data.preprocessor import load_documents
        documents = load_documents()
        bm25_nodes = [
            TextNode(text=doc.text, id_=doc.id_, metadata=doc.metadata)
            for doc in documents
        ]
        self._bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes, similarity_top_k=self.top_k * 2
        )

    def retrieve(self, query: str) -> list[RecipeHit]:
        """Return top-k RecipeHit objects for the given query."""
        self._ensure_loaded()

        query_bundle = QueryBundle(query_str=query)
        vector_hits = self._vector_retriever.retrieve(query_bundle)
        bm25_hits = self._bm25_retriever.retrieve(query_bundle)

        ranked = _reciprocal_rank_fusion(vector_hits, bm25_hits)

        # Build a node_id -> node map from both result sets
        node_map = {n.node.node_id: n.node for n in vector_hits + bm25_hits}

        results: list[RecipeHit] = []
        for node_id, score in ranked[: self.top_k]:
            node = node_map.get(node_id)
            if node is None:
                continue
            meta = node.metadata
            # Deserialize JSON-encoded list/dict fields stored for ChromaDB compatibility
            tags = json.loads(meta["tags_json"]) if "tags_json" in meta else meta.get("tags", [])
            ingredients = json.loads(meta["ingredients_json"]) if "ingredients_json" in meta else meta.get("ingredients", [])
            nutrition = json.loads(meta["nutrition_json"]) if "nutrition_json" in meta else meta.get("nutrition", {})
            results.append(
                RecipeHit(
                    recipe_id=meta.get("recipe_id", -1),
                    name=meta.get("name", "Unknown"),
                    score=score,
                    minutes=meta.get("minutes", 0),
                    tags=tags,
                    ingredients=ingredients,
                    nutrition=nutrition,
                    text=node.text,
                )
            )
        return results


# Singleton for use across the app
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
