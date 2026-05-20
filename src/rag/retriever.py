"""
Hybrid retriever: BM25 keyword search + vector cosine similarity + optional
knowledge graph re-ranking (3-way fusion).

Merges both result sets (Reciprocal Rank Fusion) and returns top-k
recipes with citation metadata.

Supports configurable component selection for ablation studies:
- use_vector: enable/disable vector similarity retrieval
- use_bm25: enable/disable BM25 keyword retrieval
- use_kg: enable/disable knowledge graph re-ranking
- rrf_k: RRF fusion constant (higher = more uniform weighting)
- kg_boost_weight: how much graph proximity boosts scores
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]

from src.rag.indexer import load_index, DEFAULT_EMBED_MODEL
from src.rag.knowledge_graph import graph_rerank, load_graph

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
    source: str = ""


@dataclass
class RetrievalTrace:
    """Records which components contributed each result for analysis."""
    query: str = ""
    vector_ids: list[int] = field(default_factory=list)
    bm25_ids: list[int] = field(default_factory=list)
    fused_ids: list[int] = field(default_factory=list)
    final_ids: list[int] = field(default_factory=list)
    kg_applied: bool = False


def _reciprocal_rank_fusion(
    ranked_lists: list[list], k: int = 60
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, node in enumerate(ranked_list):
            doc_id = node.node.node_id
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Configurable retriever supporting ablation over vector, BM25, and KG components."""

    def __init__(
        self,
        top_k: int = TOP_K,
        use_vector: bool = True,
        use_bm25: bool = True,
        use_kg: bool = True,
        rrf_k: int = 60,
        kg_boost_weight: float = 0.1,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chroma_path: str | None = None,
        collection_name: str | None = None,
    ):
        self.top_k = top_k
        self.use_vector = use_vector
        self.use_bm25 = use_bm25
        self.use_kg = use_kg
        self.rrf_k = rrf_k
        self.kg_boost_weight = kg_boost_weight
        self.embed_model = embed_model
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._index = None
        self._vector_retriever = None
        self._bm25_retriever = None
        self._bm25_nodes = None
        self._graph = load_graph() if use_kg else None

    def _ensure_loaded(self) -> None:
        if self._index is not None and self._bm25_nodes is not None:
            return

        load_kwargs = {}
        if self.chroma_path:
            load_kwargs["chroma_path"] = Path(self.chroma_path)
        if self.embed_model:
            load_kwargs["embed_model"] = self.embed_model
        if self.collection_name:
            load_kwargs["collection_name"] = self.collection_name

        if self.use_vector and self._index is None:
            self._index = load_index(**load_kwargs)
            self._vector_retriever = VectorIndexRetriever(
                index=self._index, similarity_top_k=self.top_k * 2
            )

        if self.use_bm25 and self._bm25_nodes is None:
            from llama_index.core.schema import TextNode
            from src.data.preprocessor import load_documents
            documents = load_documents()
            self._bm25_nodes = [
                TextNode(text=doc.text, id_=doc.id_, metadata=doc.metadata)
                for doc in documents
            ]
            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=self._bm25_nodes, similarity_top_k=self.top_k * 2
            )

    def retrieve(
        self,
        query: str,
        preferred_ingredients: list[str] | None = None,
        preferred_tags: list[str] | None = None,
        return_trace: bool = False,
    ) -> list[RecipeHit] | tuple[list[RecipeHit], RetrievalTrace]:
        """Return top-k RecipeHit objects using configured retrieval components."""
        self._ensure_loaded()

        trace = RetrievalTrace(query=query)
        query_bundle = QueryBundle(query_str=query)

        ranked_lists = []
        node_map = {}

        if self.use_vector and self._vector_retriever:
            vector_hits = self._vector_retriever.retrieve(query_bundle)
            ranked_lists.append(vector_hits)
            for n in vector_hits:
                node_map[n.node.node_id] = n.node
            trace.vector_ids = [
                n.node.metadata.get("recipe_id", -1) for n in vector_hits
            ]

        if self.use_bm25 and self._bm25_retriever:
            bm25_hits = self._bm25_retriever.retrieve(query_bundle)
            ranked_lists.append(bm25_hits)
            for n in bm25_hits:
                node_map[n.node.node_id] = n.node
            trace.bm25_ids = [
                n.node.metadata.get("recipe_id", -1) for n in bm25_hits
            ]

        if not ranked_lists:
            if return_trace:
                return [], trace
            return []

        if len(ranked_lists) == 1:
            ranked = [
                (n.node.node_id, n.score if hasattr(n, "score") else 0.0)
                for n in ranked_lists[0]
            ]
        else:
            ranked = _reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        results: list[RecipeHit] = []
        for node_id, score in ranked[: self.top_k * 2]:
            node = node_map.get(node_id)
            if node is None:
                continue
            meta = node.metadata
            tags = json.loads(meta["tags_json"]) if "tags_json" in meta else meta.get("tags", [])
            ingredients = json.loads(meta["ingredients_json"]) if "ingredients_json" in meta else meta.get("ingredients", [])
            nutrition = json.loads(meta["nutrition_json"]) if "nutrition_json" in meta else meta.get("nutrition", {})

            source_parts = []
            rid = meta.get("recipe_id", -1)
            if rid in set(trace.vector_ids):
                source_parts.append("vector")
            if rid in set(trace.bm25_ids):
                source_parts.append("bm25")

            results.append(
                RecipeHit(
                    recipe_id=rid,
                    name=meta.get("name", "Unknown"),
                    score=score,
                    minutes=meta.get("minutes", 0),
                    tags=tags,
                    ingredients=ingredients,
                    nutrition=nutrition,
                    text=node.text,
                    source="+".join(source_parts) if source_parts else "unknown",
                )
            )

        trace.fused_ids = [h.recipe_id for h in results]

        if self.use_kg and self._graph is not None:
            results = graph_rerank(
                self._graph,
                results,
                preferred_ingredients=preferred_ingredients,
                preferred_tags=preferred_tags,
                boost_weight=self.kg_boost_weight,
            )
            trace.kg_applied = True

        results = results[: self.top_k]
        trace.final_ids = [h.recipe_id for h in results]

        if return_trace:
            return results, trace
        return results

    def config_dict(self) -> dict:
        """Return the current configuration as a serializable dict."""
        return {
            "top_k": self.top_k,
            "use_vector": self.use_vector,
            "use_bm25": self.use_bm25,
            "use_kg": self.use_kg,
            "rrf_k": self.rrf_k,
            "kg_boost_weight": self.kg_boost_weight,
            "embed_model": self.embed_model,
        }


# Singleton for use across the app
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
