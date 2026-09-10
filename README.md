# Khmer OCR → Retrieval → RAG Research Framework

This repository is an implementation scaffold for the research proposal:

**Beyond CER: Evaluating the Downstream Impact of Joint Khmer Text Recognition and Word Segmentation on Retrieval and RAG**

The project is organized around four research questions:

- **RQ1:** joint recognition-segmentation vs. sequential OCR→segmentation on recognition, boundary accuracy, latency, and throughput.
- **RQ2:** effects of OCR and boundary errors on sparse, dense, and hybrid retrieval.
- **RQ3:** which Khmer-specific error types cause the greatest retrieval damage at matched error rates.
- **RQ4:** how upstream errors affect retrieved context and downstream RAG answer quality.

## Design principles

1. **Tier A** is controlled modern Khmer: clean/synthetic or printed text with verified transcription and word boundaries.
2. **Tier B** is real modern Khmer: document/scene/handwriting OCR benchmarks plus a separate modern Khmer document corpus for retrieval/QA.
3. Keep the **RAG configuration fixed** when studying upstream error propagation.
4. Use **oracle conditions** to isolate recognition vs. segmentation damage.
5. Report conventional OCR metrics, but treat retrieval/RAG outcomes as first-class endpoints.

## Repository layout

```text
configs/                 Reproducible YAML experiment specifications
src/khmer_ocr_rag/       Library code
  data/                   Manifests, Unicode normalization, Khmer cluster handling
  ocr/                    Joint CTC model and OCR interfaces
  segmentation/           Sequential Khmer word-segmentation baseline
  perturbations/          Controlled Khmer-specific error injection
  retrieval/              BM25, dense, and hybrid retrieval
  rag/                    Fixed RAG interfaces and extractive demo generator
  metrics/                OCR, segmentation, retrieval, and QA metrics
  evaluation/             Bootstrap/statistical helpers
  experiments/            E1–E5 experiment runners
scripts/                  Data preparation and runnable demos
tests/                    Unit tests for research-critical logic
docs/                     Codex plan, data contract, implementation roadmap
artifacts/                 Checkpoints, indexes, results, reports (not committed)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/demo_retrieval.py
```

For GPU OCR training and dense retrieval:

```bash
pip install -e ".[ocr,dense,stats]"
```

## Experiment conditions

- **P0 Gold:** gold transcription + gold segmentation
- **P1 OCR-only:** predicted transcription, no explicit segmentation
- **P2 Sequential:** OCR → external segmenter
- **P3 Joint:** joint OCR + boundary prediction
- **P4 Oracle-OCR:** gold transcription + predicted segmentation
- **P5 Oracle-Seg:** predicted OCR + gold segmentation

The files under `configs/experiments/` map directly to the proposal's experimental matrix.

## Important implementation boundary

The repository includes a trainable *research implementation* of a joint CTC recognizer with a boundary token and FiLM-style boundary conditioning. It is aligned to the proposal and the described KTRWS mechanism, but it is **not claimed to be an exact reproduction of an unpublished/review-gated repository**. When official code/checkpoints become available, add an adapter under `src/khmer_ocr_rag/ocr/adapters/` or replace the research model behind the same interface.

## First milestone for Codex

Open `CODEX.md` and implement tasks in order. The codebase is deliberately modular so Codex can work one issue/branch at a time without changing the experimental contract.
