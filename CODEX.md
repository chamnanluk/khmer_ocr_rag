# Codex Implementation Guide

## Mission

Implement the research proposal without changing its four research questions or introducing extra objectives. The code must preserve causal comparability across P0–P5 and Tier A/Tier B.

## Non-negotiable research controls

- Joint and sequential OCR comparisons must use matched preprocessing, tokenizer/vocabulary, encoder capacity where feasible, data splits, and hardware timing protocol.
- RAG experiments must freeze chunking, retriever checkpoint, index parameters, top-k, prompt, generator checkpoint, and decoding settings within a comparison.
- Never treat automatically produced word boundaries as gold. Gold evaluation boundaries must be human-verified or come from a licensed manually annotated corpus.
- Scene/handwriting datasets may be used for OCR/error validation even if they are not suitable for passage-level RAG.
- Historical/palm-leaf data are out of scope for the main experiments.

## Implementation phases

### Phase 0 — contracts and tests
1. Keep `data/schemas.py`, `data/manifests.py`, and YAML configs backward compatible.
2. Add unit tests before modifying tokenization, perturbation, metric, or ranking logic.
3. All result rows must contain `run_id`, `seed`, `tier`, `dataset`, `condition`, and `config_hash`.

### Phase 1 — Tier A data
1. Implement renderer for 10–15 legally usable Khmer fonts supplied by the researcher.
2. Add blur, compression, contrast, noise, and perspective augmentations with stored parameters.
3. Produce train/val/test manifests with no text leakage when repeated source sentences are rendered in variants.
4. Build a human-verification workflow for at least 1,000 word-boundary evaluation lines.

### Phase 2 — OCR + segmentation (RQ1)
1. Train `JointCTCRecognizer` with and without the U+200B boundary token.
2. Train a sequential segmentation baseline over the same OCR tokenization.
3. Add adapters for official KTRWS/UKTR checkpoints if/when available.
4. Report CER, normalized CER, KCC error rate, boundary precision/recall/F1, latency, throughput.
5. Timing protocol: warmup, synchronized GPU timing, fixed batch sizes, repeated trials, mean/median/p95.

### Phase 3 — retrieval (RQ2)
1. Build a Tier B-RAG corpus with passage IDs and source-document IDs.
2. Implement BM25 baseline, dense retriever adapter, and hybrid rank fusion.
3. Recommended first dense checkpoint: BGE-M3, because recent Khmer RAG work found it strongest among tested dense retrievers; keep checkpoint configurable.
4. Evaluate P0–P5 using Recall@1/3/5/10, Precision@k, HitRate@k, MRR, nDCG.

### Phase 4 — controlled error propagation (RQ3)
1. Estimate natural confusion frequencies from Tier B OCR outputs.
2. Parameterize Tier A corruption using those distributions.
3. Run error-rate curves at 0/1/2/5/10/20%.
4. Run matched-type experiments near 5% for base, subscript, vowel, diacritic, KCC, Unicode, boundary insertion, and boundary deletion.
5. Preserve corruption logs so every modified span is auditable.

### Phase 5 — RAG (RQ4)
1. Freeze retrieval + generation configuration before comparing upstream conditions.
2. Evaluate gold, sequential, joint, oracle, and selected 2/5/10% noise conditions.
3. Record supporting passage IDs, generated answer, exact model/checkpoint, prompt hash, and retrieved context.
4. Use automated faithfulness/correctness metrics only as diagnostics; validate a stratified subset with Khmer-proficient annotators.

### Phase 6 — statistics and paper tables
1. Add paired bootstrap confidence intervals over the same query set.
2. Add non-parametric paired significance tests with Holm correction where appropriate.
3. Add effect sizes, not only p-values.
4. Export IEEE-ready CSV/LaTeX tables from immutable result files.

## Coding standards

- Prefer pure functions for metrics and perturbations.
- Every random operation accepts an explicit `numpy.random.Generator` or seed.
- Never hard-code dataset paths or model IDs in library code; use YAML.
- No hidden network calls in tests.
- Add provenance metadata to every derived artifact.
- Store raw data separately from derived data; do not commit copyrighted datasets.

## Definition of done for the first reproducible benchmark

A single command should reproduce a small-scale E2/E3 run from a manifest:

```bash
python -m khmer_ocr_rag.cli run configs/experiments/e3_rq3_rate_curve.yaml
```

and write:

```text
artifacts/results/<run_id>/config.yaml
artifacts/results/<run_id>/metrics.csv
artifacts/results/<run_id>/predictions.jsonl
artifacts/results/<run_id>/provenance.json
```
