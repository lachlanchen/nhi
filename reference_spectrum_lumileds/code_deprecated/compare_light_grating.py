#!/usr/bin/env python3
"""Plot light source SPD, grating efficiency, and their product.

The script loads ``light_spd.csv`` (lamp spectrum) and ``diffraction_grating.csv``
(multiple groove densities) from the repository root, interpolates them onto a
shared wavelength axis, and visualises both the original curves and their
pointwise product. The product approximates the light transmitted through the
selected grating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIGHT_FILE = REPO_ROOT / "light_spd.csv"
GRATING_FILE = REPO_ROOT / "diffraction_grating.csv"
OUTPUT_DIR = REPO_ROOT / "groundtruth_spectrum" / "plots"


class GratingDataError(RuntimeError):
    pass


def load_light_spd(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Light SPD file not found: {path}")
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Unexpected light SPD format; expected two columns")
    wavelengths = data[:, 0]
    intensity = data[:, 1]
    mask = np.isfinite(wavelengths) & np.isfinite(intensity)
    return wavelengths[mask], intensity[mask]


def load_grating_efficiencies(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Grating efficiency file not found: {path}")
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    # Columns come in wavelength/eff pairs for each groove density.
    groove_map = {600: (0, 1), 300: (2, 3), 830: (4, 5), 1200: (6, 7)}
    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for groove, (wl_idx, eff_idx) in groove_map.items():
        if raw.shape[1] <= eff_idx:
            continue
        wl = raw[:, wl_idx]
        eff = raw[:, eff_idx]
        mask = np.isfinite(wl) & np.isfinite(eff) & (eff >= 0)
        wl = wl[mask]
        eff = eff[mask]
        if wl.size == 0:
            continue
        order = np.argsort(wl)
        result[groove] = (wl[order], eff[order] / 100.0)
    if not result:
        raise GratingDataError("No grating curves were parsed from the CSV")
    return result


def build_product(
    light_wl: np.ndarray,
    light_intensity: np.ndarray,
    grating_wl: np.ndarray,
    grating_eff: np.ndarray,
    num_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    overlap_min = max(light_wl.min(), grating_wl.min())
    overlap_max = min(light_wl.max(), grating_wl.max())
    if overlap_min >= overlap_max:
        raise ValueError("No wavelength overlap between light SPD and grating efficiency")
    wl_common = np.linspace(overlap_min, overlap_max, num_points)
    light_interp = np.interp(wl_common, light_wl, light_intensity)
    grating_interp = np.interp(wl_common, grating_wl, grating_eff)
    product = light_interp * grating_interp
    return wl_common, light_interp, grating_interp, product


def plot_curves(
    wl_light: np.ndarray,
    light: np.ndarray,
    wl_grating: np.ndarray,
    grating: np.ndarray,
    wl_common: np.ndarray,
    product: np.ndarray,
    groove: int,
    output_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(wl_light, light, color="tab:blue", label="Light SPD")
    axes[0].set_ylabel("Relative intensity")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Light SPD vs. {groove} grooves/mm grating efficiency")

    axes_twin = axes[0].twinx()
    axes_twin.plot(wl_grating, grating * 100.0, color="tab:orange", label="Grating efficiency")
    axes_twin.set_ylabel("Efficiency (%)", color="tab:orange")
    axes_twin.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = axes_twin.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    axes[1].plot(wl_common, product, color="tab:purple", label="SPD × efficiency")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Relative throughput")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    print(f"Saved plot → {output_path}")

    backend = plt.get_backend().lower()
    if "agg" in backend:
        return
    try:
        plt.show()
    except Exception as exc:
        print(f"Skipping interactive display (backend={backend}): {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot light SPD, grating efficiency, and their product")
    parser.add_argument(
        "--grooves",
        type=int,
        default=600,
        choices=(300, 600, 830, 1200),
        help="Select grating groove density (default 600)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2000,
        help="Number of samples for the interpolated product curve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    light_wl, light_intensity = load_light_spd(LIGHT_FILE)
    grating_curves = load_grating_efficiencies(GRATING_FILE)
    if args.grooves not in grating_curves:
        raise GratingDataError(f"Groove density {args.grooves} not found in grating CSV")
    grating_wl, grating_eff = grating_curves[args.grooves]

    wl_common, light_interp, grating_interp, product = build_product(
        light_wl, light_intensity, grating_wl, grating_eff, args.points
    )

    out_name = f"light_grating_product_{args.grooves}grooves.png"
    out_path = OUTPUT_DIR / out_name

    plot_curves(
        light_wl,
        light_intensity,
        grating_wl,
        grating_eff,
        wl_common,
        product,
        args.grooves,
        out_path,
    )


if __name__ == "__main__":
    main()
