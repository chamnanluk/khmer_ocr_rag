from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationSpec:
    gaussian_blur: float = 0.0
    contrast: float = 1.0
    jpeg_quality: int = 100
    perspective: float = 0.0
    speckle_std: float = 0.0
    occlusion_fraction: float = 0.0

    def to_dict(self):
        return asdict(self)


def sample_degradation(rng: np.random.Generator, severity: int = 1) -> DegradationSpec:
    severity = max(0, min(3, int(severity)))
    if severity == 0:
        return DegradationSpec()
    return DegradationSpec(
        gaussian_blur=float(rng.uniform(0.0, [0, 0.6, 1.2, 2.0][severity])),
        contrast=float(rng.uniform([1, 0.85, 0.65, 0.45][severity], 1.1)),
        jpeg_quality=int(rng.integers([100, 75, 55, 35][severity], 101)),
        perspective=float(rng.uniform(0.0, [0, 0.02, 0.05, 0.08][severity])),
        speckle_std=float(rng.uniform(0.0, [0, 4, 9, 15][severity])),
        occlusion_fraction=float(rng.uniform(0.0, [0, 0.01, 0.03, 0.06][severity])),
    )


def apply_degradation(image, spec: DegradationSpec, rng: np.random.Generator):
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise ImportError("Install OCR extras: pip install -e '.[ocr]'") from exc

    out = image.convert("RGB")
    if spec.gaussian_blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(spec.gaussian_blur))
    if spec.contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(spec.contrast)
    if spec.perspective > 0:
        w, h = out.size
        d = spec.perspective * min(w, h)
        coeffs = (
            1, float(rng.uniform(-d, d))/max(1,h), float(rng.uniform(-d, d)),
            float(rng.uniform(-d, d))/max(1,w), 1, float(rng.uniform(-d, d)),
            0, 0,
        )
        out = out.transform(out.size, Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
    if spec.speckle_std > 0:
        arr = np.asarray(out).astype(np.float32)
        arr += rng.normal(0, spec.speckle_std, size=arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if spec.occlusion_fraction > 0:
        w, h = out.size
        area = max(1, int(w * h * spec.occlusion_fraction))
        ow = max(1, int(np.sqrt(area * max(1, w/h))))
        oh = max(1, area // ow)
        x0 = int(rng.integers(0, max(1, w-ow+1)))
        y0 = int(rng.integers(0, max(1, h-oh+1)))
        ImageDraw.Draw(out).rectangle((x0, y0, min(w, x0+ow), min(h, y0+oh)), fill="white")
    return out
