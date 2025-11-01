#!/usr/bin/env python3
"""
Align rescaled background spectrum to spectrometer ground-truth.

Given the fine-resolution background series exported by
`figure04_rescaled.py` (figure04_rescaled_bg_series.npz) this script:
  1. Loads two ground-truth spectrometer curves from the provided directory.
  2. Detects the active wavelength band from the ground-truth pair.
  3. Uses the same plateau-alignment logic as the publication overlay utility
     to derive a linear wavelength ↔ time mapping for the rescaled cumulative
     background.
  4. Saves a JSON summary containing the slope/intercept, wavelength range,
     and per-bin wavelength correspondence (50 ms bins) for later reuse in
     weighted accumulations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse publication helpers
from groundtruth_spectrum_2835.compare_publication_cumulative import (  # noqa: E402
    detect_visible_edges,
    load_gt_curves,
)
from groundtruth_spectrum.compare_reconstruction_to_gt import (  # noqa: E402
    moving_average,
    normalise_curve,
    detect_active_region,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align rescaled background cumulant to spectrometer ground-truth.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        required=True,
        help="Directory produced by figure04_rescaled.py containing figure04_rescaled_bg_series.npz and weights JSON.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=REPO_ROOT / "groundtruth_spectrum_2835",
        help="Directory containing spectrometer *.txt exports.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit path for the alignment JSON (defaults to figure dir).",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Emit PNG alongside PDF outputs.",
    )
    return parser.parse_args()


def load_background_series(figure_dir: Path) -> Dict[str, np.ndarray]:
    series_path = figure_dir / "figure04_rescaled_bg_series.npz"
    if not series_path.exists():
        raise FileNotFoundError(f"Background series not found: {series_path}")
    data = {key: np.asarray(value) for key, value in np.load(series_path).items()}
    required = {"time_ms", "exp_rescaled"}
    missing = sorted(req for req in required if req not in data)
    if missing:
        raise KeyError(f"Missing arrays in {series_path}: {missing}")
    return data


def load_weights_metadata(figure_dir: Path) -> Dict[str, object]:
    weights_path = figure_dir / "figure04_rescaled_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights JSON not found: {weights_path}")
    with weights_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)
    return meta


def compute_bin_centres(metadata: List[Dict[str, float]]) -> List[float]:
    if not metadata:
        return []
    base_start = metadata[0]["start_us"]
    centres = []
    for item in metadata:
        start_us = item["start_us"]
        end_us = item["end_us"]
        centre = (((start_us + end_us) * 0.5) - base_start) / 1000.0
        centres.append(float(centre))
    return centres


def align_background_to_gt(
    time_ms: np.ndarray,
    exp_rescaled: np.ndarray,
    gt_dir: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    List[str],
    Tuple[float, float],
    List[Tuple[str, np.ndarray, np.ndarray]],
]:
    gt_curves = load_gt_curves(gt_dir)
    gt_start_nm, gt_end_nm = detect_visible_edges(gt_curves)

    gt_norm_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, wl, val in gt_curves:
        mask = (wl >= gt_start_nm - 50.0) & (wl <= gt_end_nm + 50.0)
        wl_sel = wl[mask]
        val_sel = val[mask]
        smooth = moving_average(val_sel, max(21, len(val_sel) // 300))
        region = detect_active_region(smooth)
        norm_val = normalise_curve(smooth, region)
        gt_norm_curves.append((name, wl_sel.astype(np.float32), norm_val.astype(np.float32)))

    # Background edge-based time anchors (entering/leaving flats)
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    t0 = float(time_ms[region_bg.start_idx])
    t1 = float(time_ms[region_bg.end_idx])
    slope = (gt_end_nm - gt_start_nm) / (t1 - t0)
    intercept = gt_start_nm - slope * t0
    wl_series = slope * time_ms + intercept
    series_norm = normalise_curve(exp_rescaled, region_bg)
    gt_names = [name for name, _, _ in gt_curves]
    return wl_series, series_norm, slope, intercept, gt_names, (gt_start_nm, gt_end_nm), gt_norm_curves


def plot_gt_comparison(
    wl_series: np.ndarray,
    series_norm: np.ndarray,
    gt_curves: List[Tuple[str, np.ndarray, np.ndarray]],
    wavelength_range: Tuple[float, float],
    output_dir: Path,
    save_png: bool,
) -> None:
    if wl_series.size == 0 or not gt_curves:
        return

    x_min, x_max = wavelength_range
    mask_bg = (wl_series >= x_min) & (wl_series <= x_max)
    wl_bg = wl_series[mask_bg]
    bg_norm = series_norm[mask_bg]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(wl_bg, bg_norm, label="Rescaled background", color="#1f77b4", linewidth=1.6)
    for name, wl, val in gt_curves:
        mask_gt = (wl >= x_min) & (wl <= x_max)
        if not np.any(mask_gt):
            continue
        ax.plot(wl[mask_gt], val[mask_gt], label=name, linewidth=1.4)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.set_title("Background vs. spectrometer ground-truth")
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)

    out_path = output_dir / "figure04_rescaled_bg_vs_groundtruth.pdf"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figure_dir = args.figure_dir.resolve()
    if not figure_dir.exists():
        raise FileNotFoundError(figure_dir)

    bg_series = load_background_series(figure_dir)
    time_ms = np.asarray(bg_series["time_ms"], dtype=np.float32)
    exp_rescaled = np.asarray(bg_series["exp_rescaled"], dtype=np.float32)

    wl_series, series_norm, slope, intercept, gt_names, gt_range, gt_norm_curves = align_background_to_gt(
        time_ms,
        exp_rescaled,
        args.gt_dir.resolve(),
    )

    weights_meta = load_weights_metadata(figure_dir)
    bin_centres_ms = compute_bin_centres(weights_meta.get("bin_times_us", []))
    bin_mapping = [
        {
            "index": entry.get("index"),
            "time_center_ms": centre,
            "wavelength_nm": float(slope * centre + intercept),
        }
        for entry, centre in zip(weights_meta.get("bin_times_us", []), bin_centres_ms)
    ]

    plot_gt_comparison(
        wl_series,
        series_norm,
        gt_norm_curves,
        gt_range,
        figure_dir,
        args.save_png,
    )

    output_path = args.output_json
    if output_path is None:
        output_path = figure_dir / "figure04_rescaled_bg_alignment.json"
    output_path = output_path.resolve()
    payload = {
        "figure_dir": str(figure_dir),
        "series_npz": "figure04_rescaled_bg_series.npz",
        "weights_json": "figure04_rescaled_weights.json",
        "groundtruth_directory": str(args.gt_dir.resolve()),
        "groundtruth_files": gt_names,
        "neg_scale": float(weights_meta.get("neg_scale", float("nan"))),
        "rescale_step_us": float(weights_meta.get("rescale_step_us", bg_series.get("step_us", np.nan))),
        "alignment": {
            "time_range_ms": [float(time_ms[0]), float(time_ms[-1])],
            "wavelength_range_nm": [float(gt_range[0]), float(gt_range[1])],
            "slope_nm_per_ms": float(slope),
            "intercept_nm": float(intercept),
        },
        "bin_mapping": bin_mapping,
        "bg_vs_gt_plot": "figure04_rescaled_bg_vs_groundtruth.pdf",
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved alignment JSON to {output_path}")


if __name__ == "__main__":
    main()
