#!/usr/bin/env python3
"""
Save Figure 3(c)-style bin images as two separate figures (original / compensated),
using the same data source and normalization logic as the existing multi-window
figure pipeline.

Inputs
- A forward segment NPZ (Scan_*_Forward_events.npz) or a dataset directory to search in
- Or a direct path to the trainer-saved time-binned NPZ under time_binned_frames/

Behavior
- Loads the *_all_time_bins_data_*.npz produced by training (50 ms bins in our runs)
- Selects a bin (explicit index or "best" by max variance drop)
- Computes shared vmin/vmax across the pair
- Saves two independent figures, each with axes and an individual colorbar

Example (sanqin; choose best bin; save under publication_code/figures):
  ~/miniconda3/envs/nhi_test/bin/python \
    publication_code/multiwindow_compensation_figure_save_separate_figure.py \
    scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/ \
    --select best --outdir publication_code/figures

Example (explicit bin 18):
  ~/miniconda3/envs/nhi_test/bin/python \
    publication_code/multiwindow_compensation_figure_save_separate_figure.py \
    scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/ \
    --bin-index 18 --outdir publication_code/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def find_bins_npz_from_input(path: Path) -> Path:
    p = path.resolve()
    if p.is_file() and p.suffix.lower() == ".npz":
        return p
    # Treat as dataset directory; look for forward segment
    cands = sorted(p.rglob("*_segments/Scan_*_Forward_events.npz"), key=lambda q: q.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No forward segment NPZ found under {p}")
    seg_npz = cands[-1]
    tb_dir = seg_npz.parent / "time_binned_frames"
    npz_cands = sorted(tb_dir.glob(f"{seg_npz.stem}_chunked_processing_all_time_bins_data_*.npz"), key=lambda q: q.stat().st_mtime)
    if not npz_cands:
        npz_cands = sorted(tb_dir.glob("*_all_time_bins_data_*.npz"), key=lambda q: q.stat().st_mtime)
    if not npz_cands:
        raise FileNotFoundError(f"No *_all_time_bins_data_*.npz found in {tb_dir}")
    return npz_cands[-1]


def select_best_bin(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=False) as d:
        ok = sorted(k for k in d.keys() if k.startswith("original_bin_"))
        ck = sorted(k for k in d.keys() if k.startswith("compensated_bin_"))
        if not ok or len(ok) != len(ck):
            raise RuntimeError("Unexpected NPZ structure for best-bin selection")
        var_o = np.array([np.var(d[k].astype(np.float32)) for k in ok], dtype=np.float32)
        var_c = np.array([np.var(d[k].astype(np.float32)) for k in ck], dtype=np.float32)
        diff = var_o - var_c
        idx = int(np.argmax(diff)) if float(diff.max()) > 0 else int(np.argmax(var_o))
        return idx


def save_one(img: np.ndarray, stem: str, outdir: Path, vmin: float, vmax: float, *, title: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    im = ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    label_font, tick_font = 12, 10
    ax.set_title(title, fontsize=18, pad=8)
    ax.set_xlabel("X (px)", fontsize=label_font)
    ax.set_ylabel("Y (px)", fontsize=label_font)
    ax.tick_params(axis='both', labelsize=tick_font)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Slim colorbar on the right
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="2.5%", height="85%", loc="center right", borderpad=1.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=tick_font)
    cb.set_label("Value", rotation=90, fontsize=label_font)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.18)
    fig.savefig(outdir / f"{stem}.pdf", dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_combined(ob: np.ndarray, cb: np.ndarray, bin_idx: int, outdir: Path, vmin: float, vmax: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.4), sharey=True)
    label_font, tick_font, title_font = 12, 10, 18
    im1 = ax1.imshow(ob, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    ax1.set_title(f"Original – Bin {bin_idx}", fontsize=title_font, pad=8)
    ax1.set_xlabel("X (px)", fontsize=label_font)
    ax1.set_ylabel("Y (px)", fontsize=label_font)
    ax1.tick_params(axis='both', labelsize=tick_font)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    im2 = ax2.imshow(cb, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
    ax2.set_title(f"Compensated – Bin {bin_idx}", fontsize=title_font, pad=8)
    ax2.set_xlabel("X (px)", fontsize=label_font)
    ax2.tick_params(axis='both', labelsize=tick_font)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Shared colorbar to the right
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax2, width="2.5%", height="85%", loc="center right", borderpad=1.3)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=tick_font)
    cb.set_label("Value", rotation=90, fontsize=label_font)

    fig.subplots_adjust(left=0.07, right=0.88, top=0.80, bottom=0.18, wspace=0.08)
    fig.savefig(outdir / f"figure03_bin{bin_idx:02d}_combined.pdf", dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(outdir / f"figure03_bin{bin_idx:02d}_combined.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description="Save Figure 3(c) as two separate figures (original/compensated)")
    ap.add_argument("input", type=Path, help="Segment NPZ, dataset directory, or direct *_all_time_bins_data_*.npz")
    ap.add_argument("--bin-index", type=int, default=None, help="Explicit bin index (if omitted, select best by variance drop)")
    ap.add_argument("--outdir", type=Path, default=Path("publication_code/figures"))
    ap.add_argument("--also-combined", action="store_true", help="Also save the legacy combined two-panel figure (panel c style)")
    args = ap.parse_args()

    bins_npz = find_bins_npz_from_input(args.input)
    bin_idx = args.bin_index if args.bin_index is not None else select_best_bin(bins_npz)

    with np.load(bins_npz, allow_pickle=False) as d:
        ob = d[f"original_bin_{bin_idx}"].astype(np.float32)
        cb = d[f"compensated_bin_{bin_idx}"].astype(np.float32)

    vmin = float(min(ob.min(), cb.min()))
    vmax = float(max(ob.max(), cb.max()))

    out = args.outdir / f"figure03_separate_bin{bin_idx:02d}"
    save_one(ob, f"figure03_bin{bin_idx:02d}_original", out, vmin, vmax, title=f"Original – Bin {bin_idx}")
    save_one(cb, f"figure03_bin{bin_idx:02d}_compensated", out, vmin, vmax, title=f"Compensated – Bin {bin_idx}")
    if args.also_combined:
        save_combined(ob, cb, bin_idx, out, vmin, vmax)
    print(f"Saved separate bin figures to {out} (bin {bin_idx})\nSource NPZ: {bins_npz}\nAlso combined: {args.also_combined}")


if __name__ == "__main__":
    main()
