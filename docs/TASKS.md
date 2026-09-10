# Codex Task Backlog

Use one branch/PR per task. Do not combine research-contract changes with model tuning.

## P0 — required before experiments

- [ ] `DATA-01`: ingest Tier A clean text and create source-level train/val/test split.
- [ ] `DATA-02`: document word-boundary annotation guideline and adjudication protocol.
- [ ] `DATA-03`: create Tier B-OCR manifests without modifying official splits.
- [ ] `DATA-04`: curate Tier B-RAG documents, passages, QA, and relevance IDs.
- [ ] `KCC-01`: replace baseline KCC tokenizer with exact published/official KCC rules when available; add regression tests.
- [ ] `PROV-01`: store dataset license/provenance and SHA256 for each immutable manifest.

## RQ1

- [ ] `OCR-01`: image dataset/dataloader with aspect-preserving padding.
- [ ] `OCR-02`: train joint CTC `b=0/b=1` model.
- [ ] `OCR-03`: train sequential KCC boundary segmenter.
- [ ] `OCR-04`: add greedy CTC decoder and optionally beam-search sensitivity analysis.
- [ ] `OCR-05`: implement exact timing protocol: warmup, CUDA synchronization, mean/median/p95, memory.
- [ ] `OCR-06`: official KTRWS/UKTR adapters if checkpoints/code are accessible.
- [ ] `EVAL-01`: aligned boundary evaluation when OCR text is imperfect; document alignment method.

## RQ2

- [ ] `IR-01`: BM25 segmented-word baseline.
- [ ] `IR-02`: sparse KCC/character-n-gram robustness ablation.
- [ ] `IR-03`: BGE-M3 dense retriever with pinned revision and embedding cache.
- [ ] `IR-04`: RRF hybrid retriever; tune only on validation.
- [ ] `IR-05`: deterministic passage chunking with source IDs shared across P0–P5.
- [ ] `IR-06`: paired query-level metric export.

## RQ3

- [ ] `ERR-01`: derive Tier B natural confusion matrix by Khmer error family.
- [ ] `ERR-02`: replace generic deletion fallback with confusion-driven substitutions.
- [ ] `ERR-03`: calibrated 0/1/2/5/10/20% rate curves.
- [ ] `ERR-04`: matched ~5% corruption experiment and edit audit trail.
- [ ] `STAT-01`: paired bootstrap CIs and effect sizes.
- [ ] `STAT-02`: error-type × retriever regression/mixed model.

## RQ4

- [ ] `RAG-01`: pin primary open-weight Khmer-capable generator after validation-only selection.
- [ ] `RAG-02`: grounded/abstaining prompt and prompt hash.
- [ ] `RAG-03`: save context, passage IDs, answers, and generator metadata for every query.
- [ ] `RAG-04`: automated diagnostic metrics with exact judge model/version.
- [ ] `HUM-01`: human annotation sheet for 100–150 stratified answers.
- [ ] `HUM-02`: inter-annotator agreement and adjudication.

## Paper outputs

- [ ] `PAPER-01`: immutable final result CSVs.
- [ ] `PAPER-02`: IEEE table export with confidence intervals.
- [ ] `PAPER-03`: error curves and trade-off figures generated only from result files.
- [ ] `PAPER-04`: experiment ledger mapping every table/figure to run IDs and git commits.
