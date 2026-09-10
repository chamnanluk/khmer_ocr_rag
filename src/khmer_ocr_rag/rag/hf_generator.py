from __future__ import annotations

DEFAULT_PROMPT = """You are answering a question using only the provided Khmer evidence.
If the evidence is insufficient, say that the evidence is insufficient. Do not invent facts.

Question:
{question}

Evidence:
{contexts}

Answer in Khmer:"""


class HFGenerator:
    """Pinned Hugging Face causal-LM generator adapter for controlled RAG experiments."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device_map: str | None = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install RAG extras: pip install -e '.[rag]'") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, device_map=device_map)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.revision = revision

    def generate(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(f"[{i+1}] {x}" for i, x in enumerate(contexts))
        prompt = self.prompt_template.format(question=question, contexts=context_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
