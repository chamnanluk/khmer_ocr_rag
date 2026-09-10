from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    except ImportError as exc:
        raise SystemExit("Install OCR extras: pip install -e '.[ocr]'") from exc

    parser = argparse.ArgumentParser(description="Render Tier A Khmer text lines from researcher-supplied fonts.")
    parser.add_argument("--input", required=True, help="UTF-8 JSONL with sample_id and text_gold")
    parser.add_argument("--font", action="append", required=True, help="Path to a legally usable Khmer font; repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--font-size", type=int, default=38)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)
    fonts = [ImageFont.truetype(p, args.font_size) for p in args.font]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = []
    for row in rows:
        text = row["text_gold"]
        font_idx = int(rng.integers(0, len(fonts)))
        font = fonts[font_idx]
        bbox = font.getbbox(text)
        width = max(128, bbox[2] - bbox[0] + 32)
        image = Image.new("RGB", (width, args.height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((16, max(0, (args.height - args.font_size)//2 - 4)), text, fill="black", font=font)
        # Controlled visual corruption parameters are stored for reproducibility.
        blur = float(rng.choice([0.0, 0.0, 0.5, 1.0]))
        contrast = float(rng.choice([0.8, 1.0, 1.0, 1.2]))
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        image_path = out / f"{row['sample_id']}.png"
        image.save(image_path)
        manifest.append({**row, "image_path": str(image_path), "render": {"font": args.font[font_idx], "font_size": args.font_size, "blur": blur, "contrast": contrast, "seed": args.seed}})
    (out / "render_manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in manifest)+"\n", encoding="utf-8")
    print(f"rendered {len(manifest)} lines -> {out}")


if __name__ == "__main__":
    main()
