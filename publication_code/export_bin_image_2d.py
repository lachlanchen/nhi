#!/usr/bin/env python3
"""
Export a single time-bin frame (original/compensated) as plain 2D images.

Usage examples:

1) Provide a segment; script finds the latest trainer NPZ next to it:
   ./publication_code/export_bin_image_2d.py \
     scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/ \
     --bin-index 18 --out-dir publication_code/figures

2) Provide a direct *_all_time_bins_data_*.npz:
   ./publication_code/export_bin_image_2d.py \
     --bins-npz <path/to/..._all_time_bins_data_*.npz> \
     --bin-index 18 --out-dir publication_code/figures

Saves two files (by default, both):
  - bin18_original.{png,pdf}
  - bin18_compensated.{png,pdf}

They share the same vmin/vmax so the colormap matches between the pair.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_bins_npz_from_segment(segment_or_dir: Path) -> Path:
    p = segment_or_dir
    seg_npz: Path | None = None
    if p.is_dir():
        # Search for a forward segment within this dataset dir
        cands = sorted(p.rglob("*_segments/Scan_*_Forward_events.npz"), key=lambda q: q.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"No forward segment NPZ found under {p}")
        seg_npz = cands[-1]
    else:
        seg_npz = p
    tb_dir = seg_npz.parent / "time_binned_frames"
    cands = sorted(tb_dir.glob(f"{seg_npz.stem}_chunked_processing_all_time_bins_data_*.npz"), key=lambda q: q.stat().st_mtime)
    if not cands:
        # Fallback: any *_all_time_bins_data_*.npz
        cands = sorted(tb_dir.glob("*_all_time_bins_data_*.npz"), key=lambda q: q.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No time-binned NPZ found in {tb_dir}")
    return cands[-1]


def save_img_pair(npz_path: Path, bin_index: int, out_dir: Path, cmap: str = "magma", dpi_png: int = 300, also_pdf: bool = True) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as d:
        ob_key = f"original_bin_{bin_index}"
        cb_key = f"compensated_bin_{bin_index}"
        if ob_key not in d or cb_key not in d:
            raise KeyError(f"Keys not found in {npz_path.name}: {ob_key}, {cb_key}")
        ob = d[ob_key].astype(np.float32)
        cb = d[cb_key].astype(np.float32)
    vmin = float(min(ob.min(), cb.min()))
    vmax = float(max(ob.max(), cb.max()))

    def _save_one(img: np.ndarray, stem: str) -> Path:
        plt.figure(figsize=(6, 3))
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        plt.axis("off")
        png_path = out_dir / f"{stem}.png"
        plt.savefig(png_path, dpi=dpi_png, bbox_inches="tight", pad_inches=0)
        if also_pdf:
            pdf_path = out_dir / f"{stem}.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        return png_path

    stem_base = f"bin{bin_index:02d}"
    png_o = _save_one(ob, f"{stem_base}_original")
    _ = _save_one(cb, f"{stem_base}_compensated")
    return png_o, out_dir / f"{stem_base}_compensated.png"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export plain 2D bin images (original/compensated) from time-binned NPZ")
    ap.add_argument("segment_or_dir", nargs="?", type=Path, help="Segment NPZ or dataset directory (to search for segment)")
    ap.add_argument("--bins-npz", type=Path, default=None, help="Direct path to *_all_time_bins_data_*.npz (takes precedence)")
    ap.add_argument("--bin-index", type=int, default=18, help="Bin index to export (default: 18)")
    ap.add_argument("--out-dir", type=Path, default=Path("publication_code/figures"), help="Output directory")
    args = ap.parse_args()

    if args.bins_npz is not None:
        npz_path = args.bins_npz
    else:
        if args.segment_or_dir is None:
            raise SystemExit("Provide either --bins-npz or a segment/dataset path")
        npz_path = find_bins_npz_from_segment(args.segment_or_dir)

    out_dir = args.out_dir / f"bin_images_{args.bin_index:02d}"
    png_o, png_c = save_img_pair(npz_path, args.bin_index, out_dir)
    print(f"Saved 2D bin images to {out_dir}\n  - {png_o.name}\n  - {png_c.name}")


if __name__ == "__main__":
    main()

