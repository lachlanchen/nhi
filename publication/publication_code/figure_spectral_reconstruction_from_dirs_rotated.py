#!/usr/bin/env python3
"""Create a four-row spectral reconstruction panel from prepared frame folders."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble spectral reconstruction figure from cropped frames.")
    parser.add_argument("--orig-dir", type=Path, required=True, help="Directory with original ROI frames (*_roi)")
    parser.add_argument("--comp-dir", type=Path, default=None, help="Directory with compensated ROI frames (defaults to orig-dir)")
    parser.add_argument("--gradient-dir", type=Path, required=True, help="Directory with gradient PNGs (e.g., *_gradient_20nm)")
    parser.add_argument("--ref-dir", type=Path, default=None, help="Directory with reference PNGs (optional)")
    parser.add_argument("--wavelengths", type=float, nargs="*", default=None, help="Target wavelengths (nm); defaults to sorted wavelengths available in --orig-dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder for the figure")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG beside PDF")
    parser.add_argument("--title", default="Spectral Reconstruction (ROI)", help="Figure title")
    return parser.parse_args()


def list_images_with_nm(folder: Optional[Path]) -> Dict[float, Path]:
    if folder is None or not folder.exists():
        return {}
    pattern_nm = re.compile(r"_(\d+(?:\.\d+)?)(?:[^/]*)nm", re.IGNORECASE)
    pattern_bin = re.compile(r"_(\d+(?:\.\d+)?)[^_]*to(\d+(?:\.\d+)?)", re.IGNORECASE)
    mapping: Dict[float, Path] = {}
    for path in sorted(folder.glob("*.png")):
        nm_val: Optional[float] = None
        match_nm = pattern_nm.search(path.stem)
        if match_nm:
            nm_val = float(match_nm.group(1))
        else:
            match_bin = pattern_bin.search(path.stem)
            if match_bin:
                start = float(match_bin.group(1))
                end = float(match_bin.group(2))
                nm_val = 0.5 * (start + end)
        if nm_val is None:
            continue
        mapping[nm_val] = path
    return mapping


def nearest_image(mapping: Dict[float, Path], target_nm: float) -> Optional[Tuple[float, Path]]:
    if not mapping:
        return None
    nms = np.array(list(mapping.keys()), dtype=np.float32)
    idx = int(np.argmin(np.abs(nms - target_nm)))
    nm_sel = float(nms[idx])
    return nm_sel, mapping[nm_sel]


def load_image(path: Path) -> np.ndarray:
    arr = plt.imread(path)
    if arr.ndim == 2:
        return arr
    return arr[..., :3]


def nm_to_rgb(nm: float) -> Tuple[float, float, float]:
    # Rough approximation of visible spectrum
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


def build_row_axes(fig, gs, row_idx: int, num_cols: int, label: str) -> List[plt.Axes]:
    label_ax = fig.add_subplot(gs[row_idx, 0])
    label_ax.axis("off")
    label_ax.text(0.5, 0.5, label, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    axes: List[plt.Axes] = []
    for col in range(num_cols):
        ax = fig.add_subplot(gs[row_idx, col + 1])
        ax.axis("off")
        axes.append(ax)
    return axes


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_map = list_images_with_nm(args.orig_dir.resolve())
    comp_map = list_images_with_nm(args.comp_dir.resolve() if args.comp_dir else args.orig_dir.resolve())
    grad_map = list_images_with_nm(args.gradient_dir.resolve())
    ref_map = list_images_with_nm(args.ref_dir.resolve()) if args.ref_dir else {}

    if not orig_map:
        raise FileNotFoundError(f"No wavelength-tagged frames found in {args.orig_dir}")
    if args.wavelengths:
        wavelength_targets = list(args.wavelengths)
    else:
        wavelength_targets = sorted(orig_map.keys())

    columns: List[dict] = []
    for target in wavelength_targets:
        col = {"target": target}
        for key, mapping in [("orig", orig_map), ("comp", comp_map), ("grad", grad_map), ("ref", ref_map)]:
            sel = nearest_image(mapping, target)
            col[key] = sel
        columns.append(col)

    num_cols = len(columns)
    row_labels = ["Original", "Compensated", "Gradient", "Reference"]
    fig = plt.figure(figsize=(1.4 * num_cols + 0.6, 5.6))
    height_ratios = [1, 1, 1, 1, 0.15]
    gs = fig.add_gridspec(len(row_labels) + 1, num_cols + 1, wspace=0.08, hspace=0.08, width_ratios=[0.2] + [1] * num_cols, height_ratios=height_ratios)

    def get_paths(key):
        selected_paths = []
        labels = []
        for col in columns:
            sel = col.get(key)
            if sel:
                nm_sel, path = sel
                selected_paths.append(path)
                labels.append(f"{nm_sel:.0f} nm")
            else:
                selected_paths.append(None)
                labels.append("")
        return selected_paths, labels

    for row_idx, key in enumerate(["orig", "comp", "grad", "ref"]):
        axes = build_row_axes(fig, gs, row_idx, num_cols, row_labels[row_idx])
        paths, labels = get_paths(key)
        for ax, lbl, path in zip(axes, labels, paths):
            if path is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7, color="red")
                continue
            img = load_image(path)
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(lbl, fontsize=7)

    bar_ax = fig.add_subplot(gs[-1, 1:])
    add_colorbar(bar_ax, wavelength_targets)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    pdf_path = output_dir / "spectral_reconstruction_roi.pdf"
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")
    if args.save_png:
        fig.savefig(output_dir / "spectral_reconstruction_roi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
