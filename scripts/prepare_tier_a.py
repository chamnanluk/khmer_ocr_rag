from __future__ import annotations

import argparse
import json
from pathlib import Path

from khmer_ocr_rag.data.manifests import write_jsonl
from khmer_ocr_rag.data.provenance import build_dataset_provenance
from khmer_ocr_rag.data.sources import read_tier_a_sources
from khmer_ocr_rag.data.splitting import prepare_tier_a_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic Tier A source-level manifest without network access."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--license", dest="default_license")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/tier_a_split_report.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("artifacts/reports/tier_a_provenance.json"),
    )
    args = parser.parse_args()

    config = {
        "dataset": args.dataset,
        "default_license": args.default_license,
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "unicode_normalization": "NFC",
        "split_group": "document_id",
        "duplicate_text_policy": "join_source_components",
    }
    try:
        records = read_tier_a_sources(args.input)
        prepared, report = prepare_tier_a_records(
            records,
            dataset=args.dataset,
            default_license=args.default_license,
            seed=args.seed,
            split_ratios=config["split_ratios"],
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_jsonl(prepared, args.output)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = build_dataset_provenance(
        input_path=args.input,
        output_path=args.output,
        config=config,
        counts_by_split=report["counts_by_split"],
    )
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"prepared records: {report['record_count']}")
    print("splits:", report["counts_by_split"])
    print(f"manifest: {args.output}")
    print(f"report: {args.report}")
    print(f"provenance: {args.provenance}")


if __name__ == "__main__":
    main()
