#!/usr/bin/env python3
"""
Compute gradient along wavelength for a stack of ROI frames saved per wavelength.

Input directory is expected to contain images named with an "nm" suffix, e.g.:
  band_012_488nm.png, band_013_492nm.png, ...

Steps:
 1) Parse wavelength from filenames; sort ascending
 2) Load images (grayscale or convert RGB->gray via luminance)
 3) Compute dI/dλ per pixel via finite differences along the wavelength axis:
    - central difference for interior: (I[i+1]-I[i-1]) / (λ[i+1]-λ[i-1])
    - forward/backward difference at edges
 4) Save gradient frames to a sibling folder next to input, named <name>_gradient
    - For each wavelength index i, save NPZ (float32), PDF and optional PNG

Usage:
  python scripts/hs_gradient_wavelength.py --frames-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched --save-png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gradient along wavelength for ROI frames")
    ap.add_argument("--frames-dir", type=Path, required=True, help="Directory of ROI frames with *_<nm>nm.png naming")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output directory; defaults to sibling '<name>_gradient'")
    ap.add_argument("--bin-nm", type=float, default=None, help="If set, compute one gradient per non-overlapping bin of this nm width (difference across bin endpoints)")
    return ap.parse_args()


def list_frames_with_nm(frames_dir: Path) -> List[Tuple[float, Path]]:
    pat = re.compile(r".*_(\d+(?:\.\d+)?)nm(?:_[^.]*)?\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    items: List[Tuple[float, Path]] = []
    for p in sorted(frames_dir.glob("*.png")):
        m = pat.match(p.name)
        if m:
            try:
                nm = float(m.group(1))
                items.append((nm, p))
            except Exception:
                continue
    # Also try jpg/jpeg if no pngs
    if not items:
        for p in sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg")):
            m = pat.match(p.name)
            if m:
                try:
                    nm = float(m.group(1))
                    items.append((nm, p))
                except Exception:
                    continue
    items.sort(key=lambda x: x[0])
    return items


def imread_gray(path: Path) -> np.ndarray:
    arr = plt.imread(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Assume RGB[A]; convert to luminance
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return gray.astype(np.float32)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def save_frame(stem: Path, frame: np.ndarray, save_png: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), frame=frame.astype(np.float32))
    data = frame.astype(np.float32)
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(stem.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    if save_png:
        normed = (data - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)
        plt.imsave(stem.with_suffix(".png"), normed, cmap="gray", vmin=0.0, vmax=1.0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    items = list_frames_with_nm(frames_dir)
    if not items and frames_dir.name.endswith("_roi_crops"):
        parent = frames_dir.parent / frames_dir.name.replace("_roi_crops", "")
        if parent.exists():
            items = list_frames_with_nm(parent)
    if not items:
        raise FileNotFoundError(f"No wavelength-tagged frames found in {frames_dir}")

    nms = [nm for nm, _ in items]
    paths = [p for _, p in items]
    # Load all images into a 3D stack [N,H,W]
    imgs: List[np.ndarray] = [imread_gray(p) for p in paths]
    H, W = imgs[0].shape
    stack = np.stack(imgs, axis=0).astype(np.float32)

    wl = np.array(nms, dtype=np.float32)

    # Decide output directory(s)
    base_out = args.output_dir.resolve() if args.output_dir is not None else frames_dir.parent / (frames_dir.name.rstrip('/') + "_gradient")
    base_out.mkdir(parents=True, exist_ok=True)

    if args.bin_nm is None:
        # Mode A: per-index finite-difference gradient along wavelength
        grad = np.zeros_like(stack, dtype=np.float32)
        if len(wl) >= 2:
            grad[0] = (stack[1] - stack[0]) / max(wl[1] - wl[0], 1e-6)
            grad[-1] = (stack[-1] - stack[-2]) / max(wl[-1] - wl[-2], 1e-6)
        if len(wl) >= 3:
            denom = (wl[2:] - wl[:-2]).reshape(-1, 1, 1)
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            grad[1:-1] = (stack[2:] - stack[:-2]) / denom

        out_dir = base_out
        # Save one frame per wavelength position
        for i, nm in enumerate(nms):
            stem = out_dir / f"grad_{i:04d}_{nm:.1f}nm"
            save_frame(stem, grad[i], args.save_png)

        mode_desc = "finite_difference_central (edges forward/backward)"
        extra = {"num_frames": int(len(nms))}
    else:
        # Mode B: gradient per non-overlapping bins of width bin_nm
        bin_nm = float(args.bin_nm)
        wl_min = float(wl.min())
        wl_max = float(wl.max())
        # Start bins at the first wavelength rounded down to nearest multiple of bin_nm
        start = bin_nm * np.floor(wl_min / bin_nm)
        edges = []
        v = start
        while v < wl_max:
            edges.append(v)
            v += bin_nm
        edges.append(v)

        out_dir = base_out.parent / (base_out.name + f"_{int(bin_nm)}nm")
        out_dir.mkdir(parents=True, exist_ok=True)

        # For each bin [e_k, e_{k+1}], pick nearest frames to boundaries and compute gradient
        count = 0
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            # nearest indices to lo and hi
            i_lo = int(np.argmin(np.abs(wl - lo)))
            i_hi = int(np.argmin(np.abs(wl - hi)))
            if i_lo == i_hi:
                continue
            dlam = float(wl[i_hi] - wl[i_lo])
            if abs(dlam) < 1e-6:
                continue
            g = (stack[i_hi] - stack[i_lo]) / dlam
            stem = out_dir / f"grad_bin_{k:03d}_{wl[i_lo]:.1f}to{wl[i_hi]:.1f}nm"
            save_frame(stem, g, args.save_png)
            count += 1

        mode_desc = f"bin_endpoints_difference ({int(bin_nm)} nm bins)"
        extra = {"num_bins": int(count), "bin_nm": bin_nm}

    # Save a quick JSON metadata
    meta = {
        "source_dir": str(frames_dir),
        "output_dir": str(out_dir),
        "wavelength_nm": [float(x) for x in nms],
        "shape": [int(H), int(W)],
        "mode": mode_desc,
    }
    meta.update(extra)
    import json
    with (out_dir / "gradient_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print("Saved gradient frames to:", out_dir)


if __name__ == "__main__":
    main()
