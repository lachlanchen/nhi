#!/usr/bin/env python3
"""
Export each band (frame) of an ENVI cube to PNGs, optionally cropping by a
square ROI from roi.json.

Examples
--------
  # Export all bands with ROI crop into a folder next to the data
  python scripts/envi_export_frames.py \
      --hdr hyperspectral_data_sanqin_gt/test300.hdr \
      --data hyperspectral_data_sanqin_gt/test300.spe \
      --roi-json hyperspectral_data_sanqin_gt/roi_test300_full/roi.json \
      --out-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames

  # Export a wavelength range only
  python scripts/envi_export_frames.py --hdr ... --data ... \
      --wl-min 400 --wl-max 700 --out-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export per-band frames from ENVI cube to PNGs")
    ap.add_argument("--hdr", type=Path, required=True, help="Path to ENVI .hdr")
    ap.add_argument("--data", type=Path, default=None, help="Path to ENVI data file (e.g., .spe)")
    ap.add_argument("--roi-json", type=Path, default=None, help="ROI JSON with bbox_square_xyxy/bbox_xyxy to crop")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for frames")
    ap.add_argument("--wl-min", type=float, default=None, help="Minimum wavelength (nm) to export")
    ap.add_argument("--wl-max", type=float, default=None, help="Maximum wavelength (nm) to export")
    ap.add_argument("--start-band", type=int, default=None, help="Start band index (inclusive)")
    ap.add_argument("--end-band", type=int, default=None, help="End band index (inclusive)")
    ap.add_argument("--p-low", type=float, default=1.0, help="Lower percentile for scaling (default 1)")
    ap.add_argument("--p-high", type=float, default=99.0, help="Upper percentile for scaling (default 99)")
    ap.add_argument("--global-scale", action="store_true", help="Use global vmin/vmax across selected bands")
    return ap.parse_args()


def load_envi(hdr: Path, data: Path | None):
    from spectral.io import envi
    img = envi.open(str(hdr), str(data)) if data else envi.open(str(hdr))
    cube = img.load()  # [lines, samples, bands]
    meta = dict(img.metadata)
    return cube, meta


def get_roi_bounds(meta_json: Path | None, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    H, W = shape
    if not meta_json:
        return 0, 0, W - 1, H - 1
    d = json.loads(meta_json.read_text(encoding="utf-8"))
    xyxy = d.get("bbox_square_xyxy") or d.get("bbox_xyxy")
    if xyxy:
        x0, y0, x1, y1 = [int(v) for v in xyxy]
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        return x0, y0, x1, y1
    rel = d.get("bbox_square_rel") or d.get("bbox_rel")
    if rel:
        x0r, y0r, x1r, y1r = [float(v) for v in rel]
        x0 = int(round(x0r * W)); x1 = int(round(x1r * W))
        y0 = int(round(y0r * H)); y1 = int(round(y1r * H))
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        return x0, y0, x1, y1
    return 0, 0, W - 1, H - 1


def select_bands(meta: dict, B: int, wl_min: float | None, wl_max: float | None, start: int | None, end: int | None) -> Tuple[List[int], List[float] | None]:
    wl = meta.get("wavelength")
    if wl is not None:
        wavelengths = [float(w) for w in wl]
    else:
        wavelengths = None

    idxs = list(range(B))
    if start is not None or end is not None:
        s = start if start is not None else 0
        e = end if end is not None else B - 1
        idxs = [i for i in idxs if s <= i <= e]
    if wavelengths is not None and (wl_min is not None or wl_max is not None):
        wmin = wl_min if wl_min is not None else -np.inf
        wmax = wl_max if wl_max is not None else np.inf
        idxs = [i for i in idxs if wmin <= wavelengths[i] <= wmax]
    return idxs, wavelengths


def compute_global_limits(cube_crop: np.ndarray, band_idxs: List[int], p_low: float, p_high: float) -> Tuple[float, float]:
    vals = []
    for i in band_idxs:
        img = np.asarray(cube_crop[:, :, i], dtype=np.float32)
        vals.append(np.percentile(img, p_low))
        vals.append(np.percentile(img, p_high))
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-3
    return vmin, vmax


def save_frame(img: np.ndarray, out: Path, vmin: float | None, vmax: float | None, p_low: float, p_high: float) -> None:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if vmin is None or vmax is None:
        vmin = float(np.percentile(arr, p_low))
        vmax = float(np.percentile(arr, p_high))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-3
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out), norm, cmap="gray")


def main() -> None:
    args = parse_args()
    cube, meta = load_envi(args.hdr.resolve(), args.data.resolve() if args.data else None)
    H, W, B = cube.shape
    x0, y0, x1, y1 = get_roi_bounds(args.roi_json.resolve() if args.roi_json else None, (H, W))
    cube_crop = cube[y0 : y1 + 1, x0 : x1 + 1, :]
    band_idxs, wavelengths = select_bands(meta, B, args.wl_min, args.wl_max, args.start_band, args.end_band)

    vmin_global = vmax_global = None
    if args.global_scale:
        vmin_global, vmax_global = compute_global_limits(cube_crop, band_idxs, args.p_low, args.p_high)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in band_idxs:
        frame = cube_crop[:, :, i]
        if wavelengths is not None:
            out_path = out_dir / f"band_{i:03d}_{wavelengths[i]:.0f}nm.png"
        else:
            out_path = out_dir / f"band_{i:03d}.png"
        save_frame(frame, out_path, vmin_global, vmax_global, args.p_low, args.p_high)
    print(f"Saved {len(band_idxs)} frame(s) to: {out_dir}")


if __name__ == "__main__":
    main()
