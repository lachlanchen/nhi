#!/usr/bin/env python3
"""Render rotated hyperspectral frames (PNG stack) to an RGB image."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.hs_to_rgb import load_cie_cmf, xyz_to_srgb  # type: ignore


def parse_hdr_wavelengths(hdr_path: Path) -> List[float]:
    meta: dict[str, str] = {}
    with hdr_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip().lower()] = value.strip()
    wls = meta.get("wavelength")
    if not wls:
        raise ValueError("Header missing wavelength list, required for colorimetric RGB.")
    stripped = wls.strip("{}")
    return [float(item.strip()) for item in stripped.split(",") if item.strip()]


def load_frames(frames_dir: Path) -> Tuple[np.ndarray, List[int]]:
    pngs = sorted(frames_dir.glob("band_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No band_*.png files found in {frames_dir}")

    band_indices: List[int] = []
    frames: List[np.ndarray] = []
    pattern = re.compile(r"band_(\d+)_?")
    ref_shape = None
    for path in pngs:
        match = pattern.match(path.stem)
        if not match:
            continue
        idx = int(match.group(1))
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim != 2:
            raise ValueError(f"Expected grayscale frame at {path}, got shape {img.shape}")
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError("Rotated frames have inconsistent shapes; re-run rotation without cropping.")
        band_indices.append(idx)
        frames.append(img.astype(np.float32))

    stack = np.stack(frames, axis=-1)  # H x W x B_used
    return stack, band_indices


def render_colorimetric(stack: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    wl_cmf, xbar, ybar, zbar = load_cie_cmf(REPO_ROOT)
    x_interp = np.interp(wavelengths, wl_cmf, xbar)
    y_interp = np.interp(wavelengths, wl_cmf, ybar)
    z_interp = np.interp(wavelengths, wl_cmf, zbar)
    weights = np.ones_like(wavelengths, dtype=np.float32)
    spectra = stack.reshape(-1, stack.shape[2])
    X = spectra @ (x_interp * weights)
    Y = spectra @ (y_interp * weights)
    Z = spectra @ (z_interp * weights)
    XYZ = np.stack([X, Y, Z], axis=-1)
    y_scale = np.percentile(Y, 99.0)
    if y_scale <= 0:
        y_scale = 1.0
    XYZ /= y_scale
    rgb = xyz_to_srgb(XYZ).reshape(stack.shape[0], stack.shape[1], 3)
    return rgb


def render_three_band(stack: np.ndarray, wavelengths: np.ndarray, r_nm: float, g_nm: float, b_nm: float) -> np.ndarray:
    def nearest(target):
        return int(np.argmin(np.abs(wavelengths - target)))
    idx_r = nearest(r_nm)
    idx_g = nearest(g_nm)
    idx_b = nearest(b_nm)
    R = stack[:, :, idx_r]
    G = stack[:, :, idx_g]
    B = stack[:, :, idx_b]
    def scale99(ch):
        p = np.percentile(ch, 99.0)
        if p <= 0:
            p = 1.0
        return np.clip(ch / p, 0.0, 1.0)
    rgb = np.stack([scale99(R), scale99(G), scale99(B)], axis=-1)
    return (rgb * 255 + 0.5).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert rotated PNG frames into an RGB image.")
    ap.add_argument("--frames-dir", type=Path, required=True, help="Directory containing band_*.png outputs")
    ap.add_argument("--hdr", type=Path, required=True, help="Original ENVI .hdr (for wavelength lookup)")
    ap.add_argument("--output", type=Path, required=True, help="Output RGB file (.png)")
    ap.add_argument("--method", choices=["colorimetric", "3band"], default="colorimetric")
    ap.add_argument("--r", type=float, default=610.0, help="R wavelength for 3-band mode")
    ap.add_argument("--g", type=float, default=550.0, help="G wavelength for 3-band mode")
    ap.add_argument("--b", type=float, default=450.0, help="B wavelength for 3-band mode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stack, band_indices = load_frames(args.frames_dir.resolve())
    hdr_wavelengths = parse_hdr_wavelengths(args.hdr.resolve())
    if len(hdr_wavelengths) <= max(band_indices):
        raise ValueError("Wavelength list shorter than band indices present in frames.")

    band_indices = sorted(band_indices)
    # Reorder stack to match sorted indices
    stack = stack[:, :, np.argsort(band_indices)]
    wavelengths = np.array([hdr_wavelengths[i] for i in sorted(band_indices)], dtype=np.float32)

    if args.method == "colorimetric":
        rgb = render_colorimetric(stack, wavelengths)
    else:
        rgb = render_three_band(stack, wavelengths, args.r, args.g, args.b)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    metadata = {
        "frames_dir": str(args.frames_dir),
        "hdr": str(args.hdr),
        "method": args.method,
        "rgb_file": str(args.output),
        "band_indices": band_indices,
        "r_nm": args.r,
        "g_nm": args.g,
        "b_nm": args.b,
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved RGB image to {args.output}")


if __name__ == "__main__":
    main()
