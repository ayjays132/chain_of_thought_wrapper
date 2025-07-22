from __future__ import annotations

import math
from typing import List, Tuple

class SimpleRAG:
    """A minimal retrieval-augmented generation helper."""

    def __init__(self, documents: List[str] | None = None) -> None:
        self.documents = documents or []

    def add_document(self, text: str) -> None:
        """Store a document for later retrieval."""
        self.documents.append(text)

    def clear(self) -> None:
        self.documents = []

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return the most relevant documents using naive TF scoring."""
        scores: List[Tuple[str, float]] = []
        q_tokens = query.lower().split()
        for doc in self.documents:
            d_tokens = doc.lower().split()
            # simple term frequency based score
            score = sum(d_tokens.count(t) for t in q_tokens) / (len(d_tokens) or 1)
            if score > 0:
                scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

