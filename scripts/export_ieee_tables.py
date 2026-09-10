from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Export result CSV as a simple IEEE/LaTeX table body.")
    p.add_argument("csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    Path(args.out).write_text(df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
