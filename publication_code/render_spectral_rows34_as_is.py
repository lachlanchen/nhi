#!/usr/bin/env python3
"""Render spectral reconstruction figure using existing frame folders (no re-rotation)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose spectral reconstruction figure using existing PNGs.")
    parser.add_argument("--orig-dir", type=Path, required=True, help="Directory with original frames (row 1)")
    parser.add_argument("--comp-dir", type=Path, required=True, help="Directory with compensated frames (row 2)")
    parser.add_argument("--gradient-dir", type=Path, required=True, help="Directory with gradient PNGs (row 3)")
    parser.add_argument("--ref-dir", type=Path, required=True, help="Directory with reference PNGs (row 4)")
    parser.add_argument("--wavelengths", type=float, nargs="*", default=None, help="List of wavelengths to display (default: derived from orig-dir)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder for PDF/PNG")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG beside PDF")
    parser.add_argument("--title", default="Spectral Reconstruction", help="Figure title")
    parser.add_argument("--no-colorbar", action="store_true", help="Skip wavelength color bar at the bottom")
    parser.add_argument("--crop-json", type=Path, help="Crop metadata JSON with ref/template boxes")
    parser.add_argument("--save-crops", action="store_true", help="Save processed crops to output_dir/crops/<row>")
    parser.add_argument("--flip-row12", action="store_true", help="Flip rows 1 & 2 vertically before cropping")
    parser.add_argument("--flip-row34", action="store_true", help="Flip rows 3 & 4 vertically before cropping")
    return parser.parse_args()


PAT_NM = re.compile(r"_(\d+(?:\.\d+)?)nm", re.IGNORECASE)
PAT_BIN = re.compile(r"_(\d+(?:\.\d+)?)[^_]*to(\d+(?:\.\d+)?)", re.IGNORECASE)


def build_map(folder: Path, allow_bins: bool = False) -> Dict[float, Path]:
    mapping: Dict[float, Path] = {}
    for path in sorted(folder.glob("*.png")):
        nm_val: Optional[float] = None
        m = PAT_NM.search(path.stem)
        if m:
            nm_val = float(m.group(1))
        elif allow_bins:
            mb = PAT_BIN.search(path.stem)
            if mb:
                start = float(mb.group(1))
                end = float(mb.group(2))
                nm_val = 0.5 * (start + end)
        if nm_val is None:
            continue
        mapping[nm_val] = path
    return mapping


def choose(mapping: Dict[float, Path], target_nm: float) -> Optional[Tuple[float, Path]]:
    if not mapping:
        return None
    keys = np.array(list(mapping.keys()), dtype=np.float32)
    idx = int(np.argmin(np.abs(keys - target_nm)))
    nm_sel = float(keys[idx])
    return nm_sel, mapping[nm_sel]


def nm_to_rgb(nm: float) -> Tuple[float, float, float]:
    wl = float(np.clip(nm, 380.0, 780.0))
    if wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    if wl < 380 or wl > 780:
        factor = 0.0
    elif wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    return (r * factor, g * factor, b * factor)


def add_colorbar(ax: plt.Axes, wavelengths: List[float]) -> None:
    if len(wavelengths) < 2:
        ax.axis("off")
        return
    wl_min = min(wavelengths)
    wl_max = max(wavelengths)
    samples = np.linspace(wl_min, wl_max, 600)
    colors = np.array([nm_to_rgb(w) for w in samples])
    bar_img = np.tile(colors[None, :, :], (20, 1, 1))
    ax.imshow(bar_img, aspect="auto")
    ax.set_yticks([])
    ticks = np.linspace(0, samples.size - 1, len(wavelengths))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{w:.0f}" for w in wavelengths], fontsize=8)
    ax.set_xlabel("Wavelength (nm)", fontsize=9)


def load_image(path: Path, flip: bool = False) -> np.ndarray:
    img = plt.imread(path)
    if flip:
        img = np.flipud(img)
    return img


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_map = build_map(args.orig_dir.resolve())
    comp_map = build_map(args.comp_dir.resolve())
    grad_map = build_map(args.gradient_dir.resolve(), allow_bins=True)
    ref_map = build_map(args.ref_dir.resolve())

    if not orig_map:
        raise FileNotFoundError(f"No wavelength-tagged files found in {args.orig_dir}")

    if args.wavelengths:
        wl_targets = list(args.wavelengths)
    else:
        wl_targets = list(np.linspace(400, 700, 5))

    ref_bbox = None
    tpl_bbox = None
    if args.crop_json and Path(args.crop_json).exists():
        meta = json.loads(Path(args.crop_json).read_text())
        ref_meta = meta.get("ref_crop")
        tpl_meta = meta.get("template_crop")
        if ref_meta:
            ref_bbox = (int(ref_meta["x0"]), int(ref_meta["y0"]), int(ref_meta["x1"]), int(ref_meta["y1"]))
        if tpl_meta:
            tpl_bbox = (int(tpl_meta["x0"]), int(tpl_meta["y0"]), int(tpl_meta["x1"]), int(tpl_meta["y1"]))

    row_config = {
        "orig": {"flip": args.flip_row12, "crop": ref_bbox},
        "comp": {"flip": args.flip_row12, "crop": ref_bbox},
        "grad": {"flip": args.flip_row34, "crop": tpl_bbox},
        "ref": {"flip": args.flip_row34, "crop": tpl_bbox},
    }

    columns = []
    for target in wl_targets:
        col = {"target": target}
        for key, mapping in [("orig", orig_map), ("comp", comp_map), ("grad", grad_map), ("ref", ref_map)]:
            col[key] = choose(mapping, target)
        columns.append(col)

    ncols = len(columns)
    row_labels = ["Original", "Compensated", "Gradient", "Reference"]
    fig = plt.figure(figsize=(1.4 * ncols + 0.6, 5.6))
    height_ratios = [1, 1, 1, 1, 0.15]
    gs = fig.add_gridspec(len(row_labels) + 1, ncols + 1, wspace=0.08, hspace=0.08, width_ratios=[0.2] + [1] * ncols, height_ratios=height_ratios)

    for row_idx, key in enumerate(["orig", "comp", "grad", "ref"]):
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, row_labels[row_idx], rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
        cfg = row_config.get(key, {"flip": False, "crop": None})
        crop_dir = None
        if args.save_crops:
            crop_dir = output_dir / "crops" / key
            crop_dir.mkdir(parents=True, exist_ok=True)
        for col_idx, col in enumerate(columns):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.axis("off")
            entry = col.get(key)
            if not entry:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="red")
                continue
            nm_sel, path = entry
            img = load_image(path, flip=cfg.get("flip", False))
            crop_bbox = cfg.get("crop")
            if crop_bbox:
                x0, y0, x1, y1 = [int(v) for v in crop_bbox]
                x0 = max(0, min(img.shape[1], x0))
                x1 = max(0, min(img.shape[1], x1))
                y0 = max(0, min(img.shape[0], y0))
                y1 = max(0, min(img.shape[0], y1))
                if x0 < x1 and y0 < y1:
                    img = img[y0:y1, x0:x1]
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(f"{nm_sel:.0f} nm", fontsize=8)
            if crop_dir is not None:
                save_path = crop_dir / Path(path).name
                plt.imsave(save_path, img, cmap="gray" if img.ndim == 2 else None)

    if args.no_colorbar:
        fig.delaxes(fig.axes[-1])
    else:
        colorbar_ax = fig.add_subplot(gs[-1, 1:])
        add_colorbar(colorbar_ax, wl_targets)
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()

    pdf_path = output_dir / "spectral_reconstruction_rows34_rotated.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    if args.save_png:
        fig.savefig(output_dir / "spectral_reconstruction_rows34_rotated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
