from __future__ import annotations


class FirstContextGenerator:
    """Deterministic no-network sanity-check generator.

    It returns the first retrieved context. This is not a research RAG generator; it exists
    so the pipeline can be tested end-to-end without external APIs or large checkpoints.
    """

    def generate(self, question: str, contexts: list[str]) -> str:
        return contexts[0] if contexts else ""
