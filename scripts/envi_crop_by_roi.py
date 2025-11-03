#!/usr/bin/env python3
"""
Crop an ENVI cube using a square ROI from roi.json and save the cropped cube
back to the same folder (and optionally a colorimetric RGB quicklook).

Example:
  python scripts/envi_crop_by_roi.py \
      --hdr hyperspectral_data_sanqin_gt/test300.hdr \
      --data hyperspectral_data_sanqin_gt/test300.spe \
      --roi-json hyperspectral_data_sanqin_gt/roi_test300_full/roi.json \
      --out-prefix hyperspectral_data_sanqin_gt/test300_roi_square \
      --save-rgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Crop ENVI cube using ROI JSON (square bbox)")
    ap.add_argument("--hdr", type=Path, required=True, help="Path to ENVI .hdr")
    ap.add_argument("--data", type=Path, default=None, help="Path to ENVI data file (e.g., .spe); if omitted, spectral will infer")
    ap.add_argument("--roi-json", type=Path, required=True, help="ROI JSON containing bbox_square_xyxy or bbox_xyxy")
    ap.add_argument("--out-prefix", type=Path, required=True, help="Output path prefix for cropped .hdr/.spe")
    ap.add_argument("--ext", default=".spe", help="Extension for cropped data file (default .spe)")
    ap.add_argument("--save-rgb", action="store_true", help="Also save a colorimetric RGB quicklook of the cropped cube")
    return ap.parse_args()


def load_envi(hdr: Path, data: Path | None):
    from spectral.io import envi
    if data is None:
        img = envi.open(str(hdr))
    else:
        img = envi.open(str(hdr), str(data))
    cube = img.load()  # memmap [lines, samples, bands]
    meta = dict(img.metadata)
    return cube, meta


def save_envi(prefix: Path, arr: np.ndarray, meta: dict, ext: str) -> Tuple[Path, Path]:
    from spectral.io import envi
    meta2 = dict(meta)
    lines, samples, bands = arr.shape
    meta2["lines"] = int(lines)
    meta2["samples"] = int(samples)
    meta2["bands"] = int(bands)
    interleave = meta.get("interleave", "bil").lower()
    interleave = "bil" if interleave not in ("bil", "bsq", "bip") else interleave
    dtype = arr.dtype
    hdr_path = Path(str(prefix) + ".hdr").resolve()
    # spectral always writes .img by default; override with ext
    envi.save_image(str(hdr_path), arr, dtype=dtype, interleave=interleave, metadata=meta2, ext=ext, force=True)
    data_path = Path(str(prefix) + ext).resolve()
    return hdr_path, data_path


def _xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float32,
    )
    rgb_lin = np.tensordot(xyz, M.T, axes=1)
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    a = 0.055
    thr = 0.0031308
    rgb = np.where(rgb_lin <= thr, 12.92 * rgb_lin, (1 + a) * np.power(rgb_lin, 1 / 2.4) - a)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255 + 0.5).astype(np.uint8)


def render_rgb_colorimetric(arr: np.ndarray, meta: dict, out_png: Path) -> None:
    # arr: [lines, samples, bands]
    repo_root = Path(__file__).resolve().parents[1]
    # Load CMF file directly (ciexyz31*.txt in repo root)
    cmf_path = None
    for name in ("ciexyz31_1.txt", "ciexyz31.txt"):
        path = repo_root / name
        if path.exists():
            cmf_path = path
            break
    if cmf_path is None:
        return
    first = cmf_path.read_text().splitlines()[0]
    delim = "," if "," in first else None
    data = np.loadtxt(cmf_path, dtype=np.float32, delimiter=delim)
    wl_cmf, xbar, ybar, zbar = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    wl = meta.get("wavelength")
    if wl is None:
        return
    wavelengths = np.array([float(w) for w in wl], dtype=np.float32)
    # Convert to [H,W,B]
    cube = np.asarray(arr, dtype=np.float32)
    xk = np.interp(wavelengths, wl_cmf, xbar).astype(np.float32)
    yk = np.interp(wavelengths, wl_cmf, ybar).astype(np.float32)
    zk = np.interp(wavelengths, wl_cmf, zbar).astype(np.float32)
    H, W, B = cube.shape
    spectra = cube.reshape(-1, B)
    X = spectra @ xk
    Y = spectra @ yk
    Z = spectra @ zk
    XYZ = np.stack([X, Y, Z], axis=-1).reshape(H, W, 3)
    y_scale = float(np.percentile(XYZ[..., 1], 99.0))
    if y_scale <= 0:
        y_scale = 1.0
    XYZ /= y_scale
    rgb = _xyz_to_srgb(XYZ)
    import matplotlib.pyplot as plt
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_png), rgb)


def main() -> None:
    args = parse_args()
    hdr = args.hdr.resolve()
    data = args.data.resolve() if args.data else None
    cube, meta = load_envi(hdr, data)
    H, W, B = cube.shape
    roi = json.loads(args.roi_json.read_text(encoding="utf-8"))
    xyxy = roi.get("bbox_square_xyxy") or roi.get("bbox_xyxy")
    if not xyxy:
        raise SystemExit("ROI JSON missing bbox_square_xyxy/bbox_xyxy")
    x0, y0, x1, y1 = [int(v) for v in xyxy]
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    cropped = cube[y0 : y1 + 1, x0 : x1 + 1, :]
    hdr_out, data_out = save_envi(args.out_prefix, cropped, meta, args.ext)
    print(f"Saved cropped ENVI: {hdr_out} and {data_out}")
    if args.save_rgb:
        out_png = Path(str(args.out_prefix) + "_rgb.png")
        render_rgb_colorimetric(cropped, meta, out_png)
        print(f"Saved RGB quicklook: {out_png}")


if __name__ == "__main__":
    main()
