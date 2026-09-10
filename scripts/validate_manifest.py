from __future__ import annotations

import argparse
from collections import Counter

from khmer_ocr_rag.data.manifests import load_textline_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    rows = load_textline_manifest(args.manifest)
    print(f"valid records: {len(rows)}")
    print("tiers:", Counter(r.tier for r in rows))
    print("splits:", Counter(r.split for r in rows))
    print("modalities:", Counter(r.modality for r in rows))
    print("boundary sources:", Counter(r.boundary_source for r in rows))


if __name__ == "__main__":
    main()
