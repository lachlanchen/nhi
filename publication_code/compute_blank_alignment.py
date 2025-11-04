#!/usr/bin/env python3
"""
Compute background alignment for a blank dataset using the first Forward segment.

Steps:
 1) Load Scan_1_Forward_events.npz from the dataset's *_segments folder
 2) Load learned a,b (averages) and compensate timestamps
 3) Use 5 ms bins to accumulate weighted frames (pos=+1, neg=-neg_scale)
 4) Compute mean per frame to form a background curve (saved for reference)
 5) Also compute the 5 ms exponential rescaled background series (exp_rescaled)
 6) Auto-fit neg_scale (pos_scale fixed at 1.0) from the first Forward data
 7) Align background curve to wavelength using Figure 5 code (align_with_groundtruth)
 8) Save a compact JSON with a,b, pos/neg weights, time↔wavelength mapping, and dataset info under outputs_root

Usage example:
  python publication_code/compute_blank_alignment.py \
    --dataset scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184724 \
    --gt-dir groundtruth_spectrum_2835 --sensor-width 1280 --sensor-height 720
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Import helpers from publication_code
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "publication_code"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from figure04_rescaled import (
    load_segment_events,
    find_param_file,
    load_params,
    compute_fast_comp_times,
    auto_scale_neg_weight,
    accumulate_bin,
    subtract_background,
)
from figure04_rescaled_allinone import align_with_groundtruth


@dataclass
class DatasetPaths:
    dataset_dir: Path
    segments_dir: Path
    first_forward_npz: Path
    params_npz: Path


def find_first_forward_and_params(dataset_dir: Path) -> DatasetPaths:
    seg_dirs = sorted(dataset_dir.glob("*_segments"))
    if not seg_dirs:
        raise FileNotFoundError(f"No *_segments directory under: {dataset_dir}")
    seg_dir = seg_dirs[0]
    fwd = seg_dir / "Scan_1_Forward_events.npz"
    if not fwd.exists():
        # fallback: pick the smallest index forward
        cand = sorted(seg_dir.glob("Scan_*_Forward_events.npz"))
        if not cand:
            raise FileNotFoundError(f"No Forward segment found in {seg_dir}")
        fwd = cand[0]
    params = find_param_file(fwd)
    if params is None:
        raise FileNotFoundError(f"No learned params found alongside {fwd}")
    return DatasetPaths(dataset_dir=dataset_dir, segments_dir=seg_dir, first_forward_npz=fwd, params_npz=params)


def build_metadata_bins(tmin: float, tmax: float, step_us: float) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []
    num_bins = int(np.floor(max(0.0, tmax - tmin) / step_us))
    for i in range(num_bins):
        start = tmin + i * step_us
        end = start + step_us
        bins.append({
            "index": int(i),
            "start_us": float(start),
            "end_us": float(min(end, tmax)),
            "duration_ms": float((min(end, tmax) - start) / 1000.0),
        })
    return bins


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute blank background alignment from first Forward segment")
    ap.add_argument("--dataset", type=Path, required=True, help="Path to dataset folder (contains *_segments)")
    ap.add_argument("--gt-dir", type=Path, default=Path("groundtruth_spectrum_2835"))
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--bin-width-us", type=float, default=5000.0, help="Temporal bin width in microseconds (default 5 ms)")
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs_root")
    ap.add_argument("--save-png", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = find_first_forward_and_params(args.dataset.resolve())

    # Load events and params
    x, y, t, p = load_segment_events(ds.first_forward_npz)
    params = load_params(ds.params_npz)

    # Compensate timestamps with average a,b
    t_comp, a_avg, b_avg = compute_fast_comp_times(x, y, t, params)

    # Auto-scale neg weight from Forward data at 5 ms step (series based)
    sensor_area = float(args.sensor_width * args.sensor_height)
    neg_scale, series = auto_scale_neg_weight(
        t_comp.astype(np.float32),
        p.astype(np.float32),
        sensor_area=sensor_area,
        step_us=float(args.bin_width_us),
        pos_scale=float(args.pos_scale),
        return_series=True,  # type: ignore[misc]
    )
    neg_scale = float(neg_scale)

    # Accumulate 5 ms frames and compute per-frame mean (for reference curve)
    tmin = float(np.min(t_comp))
    tmax = float(np.max(t_comp))
    metadata_bins = build_metadata_bins(tmin, tmax, float(args.bin_width_us))
    weights = np.where(p >= 0, float(args.pos_scale), -neg_scale).astype(np.float32)
    frame_means: List[float] = []
    for m in metadata_bins:
        start = float(m["start_us"])
        end = float(m["end_us"])
        mask = (t_comp >= start) & (t_comp < end)
        frame = accumulate_bin(x, y, mask, weights, (args.sensor_height, args.sensor_width))
        frame = subtract_background(frame)
        frame_means.append(float(np.mean(frame)))

    # Align background (exp_rescaled series) to groundtruth using Figure 5 code
    out_dir = (args.output_root / "blank_alignment" / ds.dataset_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prepare metadata for alignment (expects 50 ms historically, but supports any bins)
    alignment = align_with_groundtruth(
        series,
        args.gt_dir.resolve(),
        metadata_bins,
        out_dir,
        neg_scale,
        float(args.bin_width_us),
        args.save_png,
    )

    # Prepare JSON payload
    payload = {
        "dataset": {
            "root": str(ds.dataset_dir),
            "segments_dir": str(ds.segments_dir),
            "first_forward_npz": str(ds.first_forward_npz),
        },
        "params_npz": str(ds.params_npz),
        "a_params": params["a_params"].astype(float).tolist(),
        "b_params": params["b_params"].astype(float).tolist(),
        "a_avg": float(a_avg),
        "b_avg": float(b_avg),
        "pos_scale": float(args.pos_scale),
        "neg_scale": float(neg_scale),
        "bin_width_us": float(args.bin_width_us),
        "frame_means_per_bin": frame_means,
        "series_npz": alignment.get("series_npz", "figure04_rescaled_bg_series.npz"),
        "alignment": alignment.get("alignment", {}),
        "bin_mapping": alignment.get("bin_mapping", []),
        "groundtruth": alignment.get("groundtruth_directory", str(args.gt_dir.resolve())),
        "bg_vs_gt_plot": alignment.get("bg_vs_gt_plot", ""),
    }
    out_json = out_dir / "blank_background_alignment.json"
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print("Saved alignment JSON:", out_json)


if __name__ == "__main__":
    main()
