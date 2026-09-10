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
    from khmer_ocr_rag.ocr.vocab import KCCVocabulary
    from khmer_ocr_rag.segmentation.dataset import BoundaryDataset, collate_boundaries
    from khmer_ocr_rag.segmentation.neural import NeuralBoundarySegmenter
    from khmer_ocr_rag.segmentation.train import train_boundary_epoch

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/checkpoints/segmenter.pt")
    args = p.parse_args()

    rows = [r for r in load_textline_manifest(args.manifest) if r.split == "train" and r.text_gold_segmented]
    texts = [r.text_gold_segmented for r in rows]
    vocab = KCCVocabulary.from_texts(texts)
    ds = BoundaryDataset(texts, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_boundaries)
    segmenter = NeuralBoundarySegmenter(len(vocab.tokens))
    optimizer = torch.optim.AdamW(segmenter.model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        loss = train_boundary_epoch(segmenter, loader, optimizer, args.device)
        print(f"epoch={epoch} loss={loss:.5f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": segmenter.model.state_dict(), "vocab": vocab.tokens, "args": vars(args)}, out)
    print(out)


if __name__ == "__main__":
    main()
