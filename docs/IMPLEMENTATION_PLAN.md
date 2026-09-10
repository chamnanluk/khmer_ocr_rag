# Implementation Plan Mapped to Research Questions

## RQ1 / E1 — Upstream comparison

Inputs: Tier A + Tier B-OCR manifests.

Pipelines:

- P2: OCR (boundary flag off) → sequential Khmer segmenter.
- P3: same OCR family with boundary flag on → direct segmented output.

Outputs: raw predictions, normalized predictions, KCC sequences, boundary positions, timing traces.

## RQ2 / E2 — Retrieval sensitivity

Index each of P0–P5 as separate corpus variants so query relevance judgments remain identical. Run sparse, dense, and hybrid retrieval against exactly the same query set.

The core causal unit is a query: compare its rank under P0 vs P1–P5.

## RQ3 / E3–E4 — Error curves and matched error families

Use Tier A clean corpus as the controlled source. Perturb only one error family at a time. Log every edit. At equal target error rate, compare change in Recall@k, MRR and nDCG.

Natural Tier B errors are used as an external-validity check, not as the only source of causal attribution.

## RQ4 / E5 — RAG effect

Select a fixed retrieval/generation configuration after validation. Do not retune it per OCR condition. Preserve the retrieved evidence passed to the generator so answer failures can be separated into retrieval failure and generation failure.

## Recommended minimum viable sequence

1. Validate manifests and Khmer normalization.
2. Run BM25-only E2 on gold vs simulated Tier A corruption.
3. Run E3 error curves.
4. Add dense/hybrid retrieval.
5. Integrate OCR predictions.
6. Add joint/sequential OCR training.
7. Add fixed RAG generation and human validation.

This order produces publishable diagnostic results early while the expensive OCR/RAG components are still under development.
