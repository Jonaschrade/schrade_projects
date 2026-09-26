"""
MemoryStore: one isolated ChromaDB collection per agent.

MEMORY_ENABLED = False (the ablation arm) makes ``retrieve`` return nothing while
``add`` keeps writing; reflection, which reads ``all_recent``, is gated by the
main loops.
"""

import re
import time
import unicodedata
import uuid

import chromadb

from config import MAX_MEMORIES_RETRIEVE, MEMORY_DIR, MEMORY_ENABLED, MEMORY_PERSIST
from memory.scoring import score_memory


def _collection_name(agent_name: str) -> str:
    """ChromaDB-safe collection name for an agent."""
    name = unicodedata.normalize("NFKD", agent_name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-") or "agent"
    return f"mem_{name}"


class MemoryStore:
    def __init__(self, agent_name: str, persist: bool = MEMORY_PERSIST):
        """Create or reopen agent_name's collection, on disk (MEMORY_DIR) if persist."""
        if persist:
            client = chromadb.PersistentClient(path=MEMORY_DIR)
        else:
            client = chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(_collection_name(agent_name))

    # write ------------------------------------------------------------------

    def add(
        self,
        content: str,
        mem_type: str,
        importance: float,
        embedding: list,
        stance: int = 0,
    ) -> None:
        """Store one memory.

        Args:
            mem_type:   'interaction' or 'reflection'.
            importance: LLM-rated significance, 1-10.
            stance:     +1, -1, or 0 (neutral); biases later retrieval.
        """
        self.col.add(
            documents  = [content],
            embeddings = [embedding],
            metadatas  = [{
                "type": mem_type,
                "importance": importance,
                "ts": time.time(),
                "stance": stance,
            }],
            ids        = [str(uuid.uuid4())],
        )

    # read -------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: list,
        n: int = MAX_MEMORIES_RETRIEVE,
        query_stance: int | None = None,
    ) -> list[dict]:
        """Top-n memories by composite score (memory/scoring.py) over the whole collection.

        ``query_stance`` biases toward congruent memories; None leaves retrieval
        stance-unbiased. Empty when MEMORY_ENABLED is False.

        Returns:
            [{"content", "type", "score"}], best first.
        """
        if not MEMORY_ENABLED or self.col.count() == 0:
            return []

        results = self.col.query(
            query_embeddings = [query_embedding],
            n_results        = self.col.count(),
            include          = ["documents", "metadatas", "distances"],
        )

        scored = [
            {
                "content": doc,
                "type": meta["type"],
                "score": score_memory(meta, dist, query_stance),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:n]

    def all_recent(self, limit: int) -> list[str]:
        """The ``limit`` most recent memories, for seeding reflection."""
        if self.col.count() == 0:
            return []
        # Sorted by timestamp: col.get(limit=n) follows ID order, not insertion order.
        results = self.col.get(include=["documents", "metadatas"])
        pairs = list(zip(results["documents"], results["metadatas"]))
        pairs.sort(key=lambda x: x[1].get("ts", 0.0), reverse=True)
        return [doc for doc, _ in pairs[:limit]]
