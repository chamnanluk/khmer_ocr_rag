from __future__ import annotations

from pathlib import Path


class TextLineDataset:
    """PyTorch dataset for manifest-backed line images.

    Images are resized to a fixed height while preserving aspect ratio; width padding is
    handled by `collate_textlines` so long Khmer lines are not squeezed to a fixed width.
    """

    def __init__(self, records, vocab, image_height: int = 32, emit_boundaries: bool = False):
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Install OCR extras: pip install -e '.[ocr]'") from exc
        self.Image = Image
        self.torch = torch
        self.records = records
        self.vocab = vocab
        self.image_height = image_height
        self.emit_boundaries = emit_boundaries

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = self.Image.open(Path(rec.image_path)).convert("RGB")
        w, h = image.size
        new_w = max(1, round(w * self.image_height / max(1, h)))
        image = image.resize((new_w, self.image_height), self.Image.Resampling.BILINEAR)
        import numpy as np
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = self.torch.from_numpy(arr).permute(2, 0, 1)
        target_text = rec.text_gold_segmented if self.emit_boundaries else rec.text_gold
        if target_text is None:
            raise ValueError(f"Missing target for {rec.sample_id}")
        y = self.torch.tensor(self.vocab.encode(target_text), dtype=self.torch.long)
        return {"sample_id": rec.sample_id, "image": x, "target": y, "target_text": target_text}


def collate_textlines(batch):
    import torch
    max_w = max(item["image"].shape[-1] for item in batch)
    b, c, h = len(batch), batch[0]["image"].shape[0], batch[0]["image"].shape[1]
    images = torch.ones((b, c, h, max_w), dtype=batch[0]["image"].dtype)
    widths = []
    targets = []
    target_lengths = []
    for i, item in enumerate(batch):
        w = item["image"].shape[-1]
        images[i, :, :, :w] = item["image"]
        widths.append(w)
        targets.append(item["target"])
        target_lengths.append(len(item["target"]))
    return {
        "sample_ids": [x["sample_id"] for x in batch],
        "images": images,
        "pixel_widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "target_texts": [x["target_text"] for x in batch],
    }
