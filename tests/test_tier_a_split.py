import json
import subprocess
import sys

import pytest

from khmer_ocr_rag.data.provenance import build_dataset_provenance, sha256_file
from khmer_ocr_rag.data.sources import read_tier_a_sources
from khmer_ocr_rag.data.splitting import prepare_tier_a_records


def _source(
    sample_id: str,
    document_id: str,
    text: str,
    *,
    license_name: str = "CC-BY-4.0",
    **extra,
) -> dict:
    return {
        "sample_id": sample_id,
        "document_id": document_id,
        "text_gold": text,
        "license": license_name,
        **extra,
    }


def _prepare(records: list[dict], seed: int = 42):
    return prepare_tier_a_records(
        records,
        dataset="licensed-test-fixture",
        default_license=None,
        seed=seed,
        split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
    )


def test_split_is_deterministic_and_source_grouped():
    records = [
        _source("a-1", "doc-a", "កម្ពុជា"),
        _source("a-2", "doc-a", "ភ្នំពេញ"),
        _source("b-1", "doc-b", "សៀមរាប"),
        _source("c-1", "doc-c", "បាត់ដំបង"),
        _source("d-1", "doc-d", "កំពត"),
        _source("e-1", "doc-e", "កណ្ដាល"),
    ]

    first, first_report = _prepare(records)
    second, second_report = _prepare(list(reversed(records)))

    assert first == second
    assert first_report == second_report
    doc_a_splits = {row["split"] for row in first if row["document_id"] == "doc-a"}
    assert doc_a_splits == {next(row["split"] for row in first if row["sample_id"] == "a-1")}
    assert set(first_report["counts_by_split"]) == {"train", "val", "test"}


def test_nfc_equivalent_text_cannot_cross_splits():
    records = [
        _source("a", "doc-a", "é កម្ពុជា"),
        _source("b", "doc-b", "e\u0301 កម្ពុជា"),
        _source("c", "doc-c", "ភ្នំពេញ"),
        _source("d", "doc-d", "សៀមរាប"),
    ]

    prepared, report = _prepare(records)
    by_id = {row["sample_id"]: row for row in prepared}

    assert by_id["a"]["text_gold_nfc"] == by_id["b"]["text_gold_nfc"]
    assert by_id["a"]["split"] == by_id["b"]["split"]
    assert report["cross_split_text_leaks"] == 0


def test_duplicate_sample_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        _prepare(
            [
                _source("duplicate", "doc-a", "កម្ពុជា"),
                _source("duplicate", "doc-b", "ភ្នំពេញ"),
            ]
        )


def test_missing_license_is_rejected():
    with pytest.raises(ValueError, match="license"):
        _prepare([_source("a", "doc-a", "កម្ពុជា", license_name="")])


def test_automatic_boundaries_cannot_be_claimed_as_human():
    with pytest.raises(ValueError, match="Automatic boundaries"):
        _prepare(
            [
                _source(
                    "a",
                    "doc-a",
                    "កម្ពុជាមាន",
                    text_gold_segmented="កម្ពុជា\u200bមាន",
                    boundary_source="human",
                    metadata={"boundary_generation": "automatic"},
                )
            ]
        )


def test_prepared_rows_follow_textline_contract():
    prepared, _ = _prepare([_source("a", "doc-a", "កម្ពុជា")])

    assert prepared == [
        {
            "sample_id": "a",
            "tier": "A",
            "dataset": "licensed-test-fixture",
            "split": "train",
            "modality": "document",
            "image_path": "",
            "text_gold": "កម្ពុជា",
            "text_gold_nfc": "កម្ពុជា",
            "text_gold_segmented": None,
            "boundary_source": "missing",
            "document_id": "doc-a",
            "license": "CC-BY-4.0",
            "metadata": {},
        }
    ]


def test_positive_splits_receive_a_source_when_enough_sources_exist():
    records = [
        _source("a", "doc-a", "កម្ពុជា"),
        _source("b", "doc-b", "ភ្នំពេញ"),
        _source("c", "doc-c", "សៀមរាប"),
    ]

    _, report = prepare_tier_a_records(
        records,
        dataset="licensed-test-fixture",
        default_license=None,
        seed=42,
    )

    assert report["source_components_by_split"] == {"train": 1, "val": 1, "test": 1}


def test_empty_source_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        _prepare([])


def test_transcription_whitespace_is_preserved():
    prepared, _ = _prepare([_source("a", "doc-a", "  កម្ពុជា  ")])

    assert prepared[0]["text_gold"] == "  កម្ពុជា  "
    assert prepared[0]["text_gold_nfc"] == "  កម្ពុជា  "


def test_null_required_identifier_is_rejected():
    with pytest.raises(ValueError, match="sample_id"):
        _prepare([_source(None, "doc-a", "កម្ពុជា")])


def test_provenance_contains_reproducibility_fields(tmp_path):
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "tier_a.jsonl"
    input_path.write_text('{"sample_id":"a"}\n', encoding="utf-8")
    output_path.write_text('{"sample_id":"a","split":"train"}\n', encoding="utf-8")
    config = {
        "dataset": "licensed-test-fixture",
        "seed": 42,
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    provenance = build_dataset_provenance(
        input_path=input_path,
        output_path=output_path,
        config=config,
        counts_by_split={"train": 1, "val": 0, "test": 0},
    )

    assert provenance["seed"] == 42
    assert provenance["dataset"] == "licensed-test-fixture"
    assert len(provenance["config_hash"]) == 12
    assert provenance["input_sha256"] == sha256_file(input_path)
    assert provenance["output_sha256"] == sha256_file(output_path)
    json.dumps(provenance)


def test_csv_source_metadata_is_decoded(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        'sample_id,document_id,text_gold,license,metadata\n'
        'a,doc-a,កម្ពុជា,CC-BY-4.0,"{""source_page"": 7}"\n',
        encoding="utf-8",
    )

    records = read_tier_a_sources(source)

    assert records[0]["metadata"] == {"source_page": 7}


def test_invalid_csv_metadata_is_rejected_with_line_number(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "sample_id,document_id,text_gold,license,metadata\n"
        'a,doc-a,កម្ពុជា,CC-BY-4.0,"{invalid}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"source\.csv:2"):
        read_tier_a_sources(source)


def test_prepare_cli_writes_manifest_report_and_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    manifest = tmp_path / "tier_a.jsonl"
    report = tmp_path / "split_report.json"
    provenance = tmp_path / "provenance.json"
    records = [
        _source("a", "doc-a", "កម្ពុជា"),
        _source("b", "doc-b", "ភ្នំពេញ"),
        _source("c", "doc-c", "សៀមរាប"),
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_tier_a.py",
            "--input",
            str(source),
            "--output",
            str(manifest),
            "--dataset",
            "licensed-test-fixture",
            "--seed",
            "42",
            "--report",
            str(report),
            "--provenance",
            str(provenance),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert manifest.is_file()
    assert report.is_file()
    assert provenance.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["cross_split_text_leaks"] == 0
    assert json.loads(provenance.read_text(encoding="utf-8"))["output_sha256"] == sha256_file(
        manifest
    )


def test_prepare_cli_reports_validation_error_without_traceback(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"sample_id": "a", "document_id": "doc-a", "text_gold": "កម្ពុជា"})
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_tier_a.py",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "tier_a.jsonl"),
            "--dataset",
            "licensed-test-fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires license provenance" in result.stderr
    assert "Traceback" not in result.stderr


def test_prepare_cli_reports_missing_input_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_tier_a.py",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "tier_a.jsonl"),
            "--dataset",
            "licensed-test-fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing.jsonl" in result.stderr
    assert "Traceback" not in result.stderr
