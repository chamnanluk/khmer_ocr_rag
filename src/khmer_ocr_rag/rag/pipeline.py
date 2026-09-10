from __future__ import annotations

from .base import RAGAnswer


class RAGPipeline:
    def __init__(self, retriever, generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retriever.search(question, self.top_k)
        contexts = [h.text for h in hits]
        answer = self.generator.generate(question, contexts)
        return RAGAnswer(
            answer=answer,
            retrieved_passage_ids=[h.passage_id for h in hits],
            context=contexts,
            metadata={"top_k": self.top_k},
        )
