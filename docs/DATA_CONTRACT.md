# Data Contract

## Text-line manifest

JSONL/CSV records should contain:

- `sample_id`: stable unique identifier
- `tier`: `A` or `B`
- `dataset`: source dataset name
- `split`: `train`, `val`, or `test`
- `modality`: `document`, `scene`, or `handwritten`
- `image_path`: relative path or dataset-native locator
- `text_gold`: Unicode transcription
- `text_gold_nfc`: NFC-normalized transcription
- `text_gold_segmented`: transcription containing U+200B at gold word boundaries, nullable when not human verified
- `boundary_source`: `human`, `licensed_manual_corpus`, `auto_silver`, or `missing`
- `document_id`: nullable parent document identifier
- `license`: dataset license/provenance note

### DATA-01 source preparation assumptions

- `document_id` is the indivisible source-level split group.
- Documents containing identical NFC-normalized text are joined into one split component to
  prevent repeated text from leaking across train, validation, and test.
- Split ratios target source components rather than exact row counts. Small datasets or large
  duplicate components can therefore produce different realized row ratios.
- Before image rendering, a prepared source row may have an empty `image_path`. Such a manifest
  is valid for text preparation and segmentation work but is not trainable by the OCR pipeline.
- Image rendering must populate `image_path` before OCR training.
- If boundaries were created automatically, `metadata.boundary_generation` must identify the
  automatic method and `boundary_source` must not claim `human` or `licensed_manual_corpus`.
- DATA-01 consumes only local researcher-supplied or explicitly licensed JSONL/CSV inputs; it
  performs no dataset downloads.

## Retrieval corpus

Each passage record should contain:

- `passage_id`
- `document_id`
- `text_gold`
- `text_condition` (P0–P5 or controlled corruption)
- `condition`
- `source_span`
- `provenance`

## QA file

Each QA record should contain:

- `query_id`
- `question_km`
- `answer_gold`
- `relevant_passage_ids` (one or more)
- `supporting_passage_ids`
- `answer_type`
- `annotator_status`

## Corruption log

Every controlled edit must be auditable:

- `sample_id`
- `error_family`
- `start_char`, `end_char`
- `before`, `after`
- `seed`
- `target_rate`
- `realized_rate`
