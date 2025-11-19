#!/usr/bin/env python3
"""
Recolor an RGB PNG to a blue-only palette.

Steps:
- Read the input PNG
- Convert to luminance (grayscale)
- Normalize per-image to [0, 1] (min–max)
- Map to Matplotlib 'Blues' colormap
- Save PNG next to the input with a suffix

Usage:
  python publication_code/recolor_png_to_blue.py \
    publication_code/figures/.../multiwindow_bin50ms_original_plain_sanqin.png \
    publication_code/figures/.../multiwindow_bin50ms_compensated_plain_sanqin.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from PIL import Image


def recolor_one(src: Path, out_suffix: str = "_blue.png", pmin: float = 1.0, pmax: float = 99.0, gamma: float = 0.8, reverse: bool = False) -> Path:
    img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    # Luminance (perceptual)
    gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    lo = float(np.percentile(gray, pmin))
    hi = float(np.percentile(gray, pmax))
    if not np.isfinite(hi - lo) or abs(hi - lo) < 1e-12:
        norm = np.zeros_like(gray, dtype=np.float32)
    else:
        norm = np.clip((gray - lo) / max(1e-12, (hi - lo)), 0.0, 1.0)
    if gamma is not None and gamma > 0:
        norm = np.power(norm, gamma).astype(np.float32)
    cmap = cmaps["Blues_r"] if reverse else cmaps["Blues"]
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    out = src.with_name(src.stem + out_suffix)
    Image.fromarray(rgb).save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Recolor PNGs to blue-only palette")
    ap.add_argument("inputs", nargs="+", type=Path, help="Input PNG files")
    ap.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for stretch (default 1)")
    ap.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for stretch (default 99)")
    ap.add_argument("--gamma", type=float, default=0.8, help="Gamma for contrast (default 0.8)")
    ap.add_argument("--reverse", action="store_true", help="Use Blues_r (white high, deep blue low)")
    args = ap.parse_args()
    for p in args.inputs:
        out = recolor_one(p, pmin=args.pmin, pmax=args.pmax, gamma=args.gamma, reverse=args.reverse)
        print("saved:", out)


if __name__ == "__main__":
    main()
