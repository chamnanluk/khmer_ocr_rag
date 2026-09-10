from __future__ import annotations

import argparse
from pathlib import Path


def main():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install OCR extras: pip install -e '.[ocr]'") from exc

    from khmer_ocr_rag.data.manifests import load_textline_manifest
    from khmer_ocr_rag.ocr.dataset import TextLineDataset, collate_textlines
    from khmer_ocr_rag.ocr.joint_ctc import JointCTCRecognizer
    from khmer_ocr_rag.ocr.train import train_epoch
    from khmer_ocr_rag.ocr.vocab import KCCVocabulary

    p = argparse.ArgumentParser(description="Train research joint CTC baseline")
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/checkpoints/joint_ctc.pt")
    p.add_argument("--emit-boundaries", action="store_true")
    args = p.parse_args()

    rows = [r for r in load_textline_manifest(args.manifest) if r.split == "train"]
    texts = [r.text_gold_segmented if args.emit_boundaries else r.text_gold for r in rows]
    vocab = KCCVocabulary.from_texts([t for t in texts if t is not None])
    ds = TextLineDataset(rows, vocab, emit_boundaries=args.emit_boundaries)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_textlines)
    recognizer = JointCTCRecognizer(len(vocab.tokens))
    optimizer = torch.optim.AdamW(recognizer.model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        result = train_epoch(recognizer, loader, optimizer, args.device, boundary_flag=int(args.emit_boundaries))
        print(f"epoch={epoch} loss={result.loss:.5f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": recognizer.model.state_dict(), "vocab": vocab.tokens, "args": vars(args)}, out)
    print(out)


if __name__ == "__main__":
    main()
