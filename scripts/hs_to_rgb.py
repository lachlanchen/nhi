#!/usr/bin/env python3
"""
ENVI hyperspectral to RGB renderer.

Supports two mappings:
  1) colorimetric: CIE 1931 2° CMFs + (optional) illuminant → XYZ → sRGB (D65)
  2) 3-band: pick nearest bands to R/G/B target wavelengths

Usage examples:
  # Install dependencies in your conda env
  #   conda activate nhi_test
  #   pip install spectral numpy matplotlib

  # Colorimetric (recommended when wavelengths are present in header)
  python scripts/hs_to_rgb.py hyperspectral_data_sanqin_gt/test1.hdr \
         --method colorimetric --out rgb_colorimetric.png

  # 3-band quick mapping (choose target wavelengths in nm)
  python scripts/hs_to_rgb.py hyperspectral_data_sanqin_gt/test1.hdr \
         --method 3band --r 610 --g 550 --b 450 --out rgb_3band.png

Notes
-----
- This script expects ENVI .hdr with 'wavelength' metadata. If absent, 3-band
  still works by nearest indices (but colorimetric needs wavelengths).
- CMFs are loaded from repo files 'ciexyz31.txt' (or '_1'); ensure present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from spectral import open_image
    from spectral.io import envi
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: spectral. Install with `pip install spectral`."
    ) from e


def load_envi_cube(hdr_path: Path):
    # Try simple open first
    try:
        img = open_image(str(hdr_path))
    except Exception:
        # Fallback: explicit data file alongside HDR (e.g., .spe)
        candidate = hdr_path.with_suffix('.spe')
        if candidate.exists():
            img = envi.open(str(hdr_path), str(candidate))
        else:
            raise
    cube = img.load()  # memmap (H, W, B)
    meta = img.metadata or {}
    wls = meta.get("wavelength")
    if wls is not None:
        try:
            wavelengths = np.array([float(w) for w in wls], dtype=np.float32)
        except Exception:
            wavelengths = None
    else:
        wavelengths = None
    return cube, wavelengths


def load_cie_cmf(repo_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIE 1931 2° CMFs (wavelength nm, xbar, ybar, zbar).

    Tries 'ciexyz31_1.txt' then 'ciexyz31.txt'. Files are in repo root.
    """
    for name in ("ciexyz31_1.txt", "ciexyz31.txt"):
        path = repo_root / name
        if path.exists():
            # Detect delimiter (comma vs whitespace)
            first = path.read_text().splitlines()[0]
            delim = ',' if ',' in first else None
            data = np.loadtxt(path, dtype=np.float32, delimiter=delim)
            # Heuristic: columns are [lambda, x, y, z]
            if data.shape[1] < 4:
                raise RuntimeError(f"CMF file format unexpected: {path}")
            wl, xbar, ybar, zbar = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            return wl, xbar, ybar, zbar
    raise FileNotFoundError("ciexyz31*.txt not found in repo root.")


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ (D65) to sRGB with sRGB transfer function.

    xyz: [..., 3] array
    Returns: [..., 3] uint8 image in [0,255]
    """
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float32,
    )
    rgb_lin = np.tensordot(xyz, M.T, axes=1)
    # Clip negatives before EOTF
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    # sRGB EOTF
    a = 0.055
    thr = 0.0031308
    rgb = np.where(
        rgb_lin <= thr,
        12.92 * rgb_lin,
        (1 + a) * np.power(rgb_lin, 1 / 2.4) - a,
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255 + 0.5).astype(np.uint8)


def render_colorimetric(cube: np.memmap, wavelengths: np.ndarray, repo_root: Path) -> np.ndarray:
    """Render hyperspectral cube to sRGB via CIE 1931 colorimetry.

    - Assumes cube contains radiance/reflectance-like spectra per pixel.
    - Uses equal-energy illuminant if none provided.
    """
    H, W, B = cube.shape
    wl_cmf, xbar, ybar, zbar = load_cie_cmf(repo_root)

    if wavelengths is None or len(wavelengths) != B:
        raise ValueError("Colorimetric method requires wavelengths in ENVI header.")

    # Interpolate CMFs to cube bands
    x_interp = np.interp(wavelengths, wl_cmf, xbar)
    y_interp = np.interp(wavelengths, wl_cmf, ybar)
    z_interp = np.interp(wavelengths, wl_cmf, zbar)

    # Equal-energy illuminant (weight 1.0); scale CMFs to avoid tiny numbers
    w = np.ones_like(wavelengths, dtype=np.float32)
    x_w = (x_interp * w).astype(np.float32)
    y_w = (y_interp * w).astype(np.float32)
    z_w = (z_interp * w).astype(np.float32)

    # Flatten pixels for efficient dot-product
    spectra = np.asarray(cube, dtype=np.float32).reshape(-1, B)
    X = spectra @ x_w
    Y = spectra @ y_w
    Z = spectra @ z_w
    XYZ = np.stack([X, Y, Z], axis=-1)

    # Simple normalization: scale so that 99th percentile Y maps near white
    y_scale = np.percentile(Y, 99.0)
    if y_scale <= 0:
        y_scale = 1.0
    XYZ /= y_scale

    rgb = xyz_to_srgb(XYZ).reshape(H, W, 3)
    return rgb


def render_3band(cube: np.memmap, wavelengths: np.ndarray | None, r_nm: float, g_nm: float, b_nm: float) -> np.ndarray:
    """Quick 3-band mapping using nearest bands to given R/G/B targets (nm).
    Scales each channel to its 99th percentile for contrast.
    """
    H, W, B = cube.shape
    if wavelengths is None:
        # Fall back to index-based: spread roughly across bands
        idx_r = int(0.8 * (B - 1))
        idx_g = int(0.5 * (B - 1))
        idx_b = int(0.2 * (B - 1))
    else:
        def nearest_idx(target):
            return int(np.argmin(np.abs(wavelengths - target)))

        idx_r = nearest_idx(r_nm)
        idx_g = nearest_idx(g_nm)
        idx_b = nearest_idx(b_nm)

    R = np.asarray(cube[:, :, idx_r], dtype=np.float32)
    G = np.asarray(cube[:, :, idx_g], dtype=np.float32)
    B = np.asarray(cube[:, :, idx_b], dtype=np.float32)

    def scale99(x):
        p99 = np.percentile(x, 99.0)
        if p99 <= 0:
            p99 = 1.0
        y = np.clip(x / p99, 0.0, 1.0)
        return (y * 255 + 0.5).astype(np.uint8)

    rgb = np.stack([scale99(R), scale99(G), scale99(B)], axis=-1)
    return rgb


def main():
    ap = argparse.ArgumentParser(description="Render ENVI cube to RGB (colorimetric or 3-band)")
    ap.add_argument("hdr", type=Path, help="Path to ENVI .hdr file")
    ap.add_argument("--method", choices=["colorimetric", "3band"], default="colorimetric")
    ap.add_argument("--r", type=float, default=610.0, help="R target wavelength (nm) for 3-band")
    ap.add_argument("--g", type=float, default=550.0, help="G target wavelength (nm) for 3-band")
    ap.add_argument("--b", type=float, default=450.0, help="B target wavelength (nm) for 3-band")
    ap.add_argument("--out", type=Path, default=Path("hs_rgb.png"))
    args = ap.parse_args()

    hdr_path = args.hdr.resolve()
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    cube, wavelengths = load_envi_cube(hdr_path)
    repo_root = Path(__file__).resolve().parents[1]

    if args.method == "colorimetric":
        if wavelengths is None:
            raise SystemExit("Colorimetric method requires wavelengths in header; try --method 3band.")
        rgb = render_colorimetric(cube, wavelengths, repo_root)
    else:
        rgb = render_3band(cube, wavelengths, args.r, args.g, args.b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(args.out), rgb)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
