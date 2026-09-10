from __future__ import annotations

import argparse

from khmer_ocr_rag.experiments.runner import run_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="khmer-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Create and execute a configured research run")
    run.add_argument("config")
    args = parser.parse_args()
    if args.command == "run":
        out = run_config(args.config)
        print(out)


if __name__ == "__main__":
    main()
