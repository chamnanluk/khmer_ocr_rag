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
