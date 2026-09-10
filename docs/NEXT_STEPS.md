# Next Steps Checklist

This checklist translates `CODEX.md` and `docs/TASKS.md` into the next executable
milestones. It does not change RQ1-RQ4, Tier A/Tier B, P0-P5, or the frozen-downstream
comparison rule.

## Current verified baseline

- [x] Python 3.11 is pinned in `.python-version`.
- [x] Dependencies are reproducibly locked with `uv.lock`.
- [x] CUDA 12.6 PyTorch runs on the NVIDIA RTX 4060.
- [x] Development, OCR, dense, FAISS, statistics, and RAG extras import successfully.
- [x] The example manifest validates.
- [x] The BM25 demonstration runs.
- [x] The initial test suite passes: 9 tests.
- [x] DATA-01 preparation has 16 focused unit and CLI tests.
- [x] The complete repository suite passes: 25 tests.
- [x] A local fixture completes preparation and manifest validation end to end.
- [ ] Real Tier A data and images are available locally.
- [ ] A trainable `data/manifests/tier_a.jsonl` exists.

## Step 1 - Confirm the DATA-01 inputs

Do not download or redistribute a dataset until its license explicitly permits the intended
research use.

- [ ] Identify the local source-text file or corpus directory.
- [ ] Record the dataset/corpus name and original source URL or citation.
- [ ] Record the exact license and redistribution restrictions.
- [ ] Confirm whether derived manifests may be committed.
- [ ] Identify a stable source grouping field; use `document_id` for source-level splitting.
- [ ] Identify 10-15 legally usable Khmer fonts supplied by the researcher.
- [ ] Keep raw inputs under `data/raw/`; this directory is ignored by Git.

Required decision gate: do not ingest real data until the license and provenance fields above
are known.

## Step 2 - Specify DATA-01 behavior with tests

Add `tests/test_tier_a_split.py` before implementing the splitter.

- [x] Test that a fixed seed produces identical splits.
- [x] Test that every variant of one `document_id` remains in one split.
- [x] Test that NFC-equivalent or duplicate text cannot cross splits.
- [x] Test that every `sample_id` is unique.
- [x] Test rejection of missing dataset/license provenance.
- [x] Test that automatic boundaries cannot be labeled `human` or `licensed_manual_corpus`.
- [x] Test required manifest fields and valid `train`, `val`, and `test` values.
- [x] Test that provenance contains the seed, config hash, input hash, and output hash.

Run the new tests while implementing:

```powershell
uv run --locked --extra dev pytest tests/test_tier_a_split.py -q
```

## Step 3 - Implement DATA-01 without network access

Suggested files:

```text
src/khmer_ocr_rag/data/splitting.py
src/khmer_ocr_rag/data/provenance.py
scripts/prepare_tier_a.py
tests/test_tier_a_split.py
```

- [x] Read researcher-supplied JSONL or CSV input.
- [x] Normalize Khmer text to NFC without changing the raw source transcription.
- [x] Group by `document_id` before splitting.
- [x] Use an explicit seed and configured split ratios.
- [x] Detect and reject cross-split normalized-text leakage.
- [x] Preserve source IDs, licenses, and provenance in every output record.
- [x] Write the manifest deterministically.
- [x] Write a split report and immutable provenance record.
- [x] Make the CLI fail clearly when an input path, license, or grouping field is missing.
- [x] Document all assumptions next to the implementation and in the split report.

Target command:

```powershell
uv run --locked python scripts/prepare_tier_a.py `
  --input data/raw/tier_a/source.jsonl `
  --output data/manifests/tier_a.jsonl `
  --dataset DATASET_NAME `
  --license CONFIRMED_LICENSE `
  --seed 42
```

Implementation status: the offline DATA-01 preparation tooling is complete. Step 1 and the
real-data checks in Step 4 remain open until a licensed source corpus and its provenance are
provided locally.

## Step 4 - Validate the DATA-01 output

- [ ] Validate the generated manifest.
- [ ] Confirm there are non-empty train, validation, and test splits.
- [ ] Confirm source IDs and normalized text do not leak across splits.
- [ ] Review the split and provenance reports manually.
- [ ] Run the complete unit-test and lint baseline.

```powershell
uv run --locked python scripts/validate_manifest.py data/manifests/tier_a.jsonl
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check src tests scripts
```

Expected derived outputs:

```text
data/manifests/tier_a.jsonl
artifacts/reports/tier_a_split_report.json
artifacts/reports/tier_a_provenance.json
```

## Step 5 - Complete KCC-01

Do not claim exact KTRWS/UKTR equivalence unless authoritative rules or official code are
available and licensed for use.

- [ ] Collect authoritative KCC rules or retain the current implementation as a named baseline.
- [ ] Add regression cases for COENG/subscripts.
- [ ] Add dependent-vowel and Khmer-sign cases.
- [ ] Add punctuation, numerals, whitespace, and mixed-script cases.
- [ ] Add NFC-equivalent Unicode cases.
- [ ] Add U+200B preservation/removal cases.
- [ ] Preserve the existing tokenizer API for later official adapters.
- [ ] Document the rule source, version, limitations, and deviations.
- [ ] Run all tests and lint checks before committing.

## Step 6 - Complete IR-01

- [ ] Separate the segmented-word BM25 behavior from the character-bigram fallback in tests.
- [ ] Test deterministic ranking and passage-ID tie breaking.
- [ ] Test empty queries and empty corpora.
- [ ] Test repeated terms and document-length normalization.
- [ ] Preserve `passage_id`, `document_id`, and condition metadata.
- [ ] Keep identical queries and relevance judgments across P0-P5.
- [ ] Record BM25 parameters, tokenizer policy, seed, dataset, and config hash.
- [ ] Run a small BM25-only P0 versus controlled-corruption benchmark.

## Step 7 - First reproducible benchmark gate

The first benchmark is complete only when one command executes real BM25 E2/E3 work rather
than writing a `scaffold-created` result.

- [ ] The run consumes a validated manifest and fixed query/relevance files.
- [ ] It writes `config.yaml`, `metrics.csv`, `predictions.jsonl`, and `provenance.json`.
- [ ] Every result row includes `run_id`, `seed`, `tier`, `dataset`, `condition`, and
      `config_hash`.
- [ ] Every controlled edit is auditable.
- [ ] Re-running with the same inputs and seed gives identical non-timing outputs.
- [ ] No hidden network calls occur during tests or the BM25 benchmark.

## Commit workflow

Keep each task focused and independently reviewable:

```powershell
git status --short
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check src tests scripts
git add <task-specific-files>
git commit -m "Implement DATA-01 Tier A source-level splitting"
git push -u origin data-01-tier-a-ingestion
```

Do not commit `.env`, `.venv`, raw/licensed data, checkpoints, indexes, generated results, or
generated reports.
