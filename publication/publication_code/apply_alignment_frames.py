#!/usr/bin/env python3
"""
Apply an alignment config (from alignment_configs/*.json) to a target dataset:

1) Load first Forward segment (Scan_1_Forward_events.npz)
2) Compensate timestamps with a_avg, b_avg from config
3) Accumulate weighted frames with 1 ms (or configurable) bins
   - weights: pos=+pos_scale, neg=-neg_scale from config
4) Map each bin center time (relative to t_min) to wavelength using
   slope_nm_per_ms and intercept_nm in config
5) Find indices covering [380,700] nm and compute cumulative frames up to
   each boundary; save the two frames for inspection
6) Save a mapping JSON describing index↔wavelength and the truncated range

Usage:
  python publication_code/apply_alignment_frames.py \
    --dataset scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638 \
    --config alignment_configs/angle_20_blank_2835_20250925_184724_bin5ms_alignment.json \
    --bin-width-us 1000 --sensor-width 1280 --sensor-height 720 --save-png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Reuse helpers from figure04 tooling
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "publication_code"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from figure04_rescaled import load_segment_events, accumulate_bin, subtract_background


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply alignment config to dataset and export cumulative frames at 380/700 nm")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--bin-width-us", type=float, default=1000.0)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--bg-subtract", action="store_true", help="Subtract per-frame background (mean) before accumulation")
    ap.add_argument("--save-every-ms", type=int, default=10, help="Save every this many milliseconds within the valid wavelength range (based on bin width)")
    ap.add_argument("--truncate-min-nm", type=float, default=380.0)
    ap.add_argument("--truncate-max-nm", type=float, default=700.0)
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs_root")
    ap.add_argument("--output-subdir", type=str, default=None, help="Optional subfolder name under outputs_root (default auto)")
    return ap.parse_args()


def load_alignment(cfg_path: Path) -> Dict[str, object]:
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def find_first_forward(dataset_dir: Path) -> Path:
    segs = sorted(dataset_dir.glob("*_segments"))
    if not segs:
        raise FileNotFoundError(f"No *_segments found under {dataset_dir}")
    seg_dir = segs[0]
    fwd = seg_dir / "Scan_1_Forward_events.npz"
    if not fwd.exists():
        cand = sorted(seg_dir.glob("Scan_*_Forward_events.npz"))
        if not cand:
            raise FileNotFoundError(f"No Forward segment in {seg_dir}")
        fwd = cand[0]
    return fwd


def save_frame(path: Path, frame: np.ndarray, save_png: bool, cmap: str = "coolwarm", vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), frame=frame.astype(np.float32))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    cfg = load_alignment(args.config.resolve())

    a_avg = float(cfg.get("a_avg"))
    b_avg = float(cfg.get("b_avg"))
    pos_scale = float(cfg.get("pos_scale", 1.0))
    neg_scale = float(cfg.get("neg_scale", 1.0))
    slope = float(cfg["alignment"]["slope_nm_per_ms"])  # nm per ms
    intercept = float(cfg["alignment"]["intercept_nm"])  # nm at t=0 ms

    # Load first forward events
    seg_npz = find_first_forward(dataset_dir)
    x, y, t, p = load_segment_events(seg_npz)
    # Compensate time with averages from config
    t_comp = t - (a_avg * x + b_avg * y)

    # Bin times at 1 ms (or requested)
    tmin = float(np.min(t_comp))
    tmax = float(np.max(t_comp))
    bin_us = float(args.bin_width_us)
    n_bins = int(np.floor(max(0.0, tmax - tmin) / bin_us))
    sensor_shape = (args.sensor_height, args.sensor_width)
    weights = np.where(p >= 0, pos_scale, -neg_scale).astype(np.float32)

    frames: List[np.ndarray] = []
    bin_meta: List[Dict[str, float]] = []
    for i in range(n_bins):
        start = tmin + i * bin_us
        end = start + bin_us
        mask = (t_comp >= start) & (t_comp < end)
        frame = accumulate_bin(x, y, mask, weights, sensor_shape)
        if args.bg_subtract:
            frame = subtract_background(frame)
        frames.append(frame)
        center_ms = ((start + end) * 0.5 - tmin) / 1000.0
        bin_meta.append({
            "index": int(i),
            "start_us": float(start),
            "end_us": float(min(end, tmax)),
            "time_center_ms": float(center_ms),
        })

    # Map to wavelength for each bin center
    wl_seq = np.array([slope * m["time_center_ms"] + intercept for m in bin_meta], dtype=np.float32)

    # Determine indices covering [380, 700] nm
    lo_nm = float(args.truncate_min_nm)
    hi_nm = float(args.truncate_max_nm)
    valid = np.where((wl_seq >= lo_nm) & (wl_seq <= hi_nm))[0]
    idx_lo = int(valid[0]) if valid.size > 0 else 0
    idx_hi = int(valid[-1]) if valid.size > 0 else max(0, len(wl_seq) - 1)

    # Cumulative sum across the TRUNCATED range [idx_lo, idx_hi]
    # Use float32 for memory; accumulate only within the selected range
    H, W = sensor_shape
    cum_range = np.zeros((H, W), dtype=np.float32)
    # Also retain the cumulative frame at the first and last indices for convenience
    cum_at_lo = None
    cum_at_hi = None
    for i in range(idx_lo, idx_hi + 1):
        cum_range += frames[i].astype(np.float32)
        if i == idx_lo:
            cum_at_lo = cum_range.copy()
        if i == idx_hi:
            cum_at_hi = cum_range.copy()

    # Output structure
    subdir = args.output_subdir or ("apply_alignment_bgsub" if args.bg_subtract else "apply_alignment")
    out_dir = (args.output_root / subdir / dataset_dir.name).resolve()
    # Parallel grayscale export folder
    out_dir_gray = (args.output_root / (subdir + "_gray") / dataset_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_gray.mkdir(parents=True, exist_ok=True)

    # Save frames (NPZ + PDF/PNG)
    # Save cumulative frames at the lower and upper bounds (first and final in-range cumsum)
    if cum_at_lo is not None:
        stem = f"cumulative_start_{int(lo_nm)}nm_bin{int(bin_us/1000)}ms"
        save_frame(out_dir / stem, cum_at_lo, args.save_png, cmap="coolwarm")
        save_frame(out_dir_gray / stem, cum_at_lo, args.save_png, cmap="gray")
    if cum_at_hi is not None:
        stem = f"cumulative_end_{int(hi_nm)}nm_bin{int(bin_us/1000)}ms"
        save_frame(out_dir / stem, cum_at_hi, args.save_png, cmap="coolwarm")
        save_frame(out_dir_gray / stem, cum_at_hi, args.save_png, cmap="gray")

    # Save selected per-bin frames every N within [idx_lo, idx_hi]
    per_bin_dir = out_dir / "per_bin_frames"
    per_bin_dir.mkdir(parents=True, exist_ok=True)
    per_bin_dir_gray = out_dir_gray / "per_bin_frames"
    per_bin_dir_gray.mkdir(parents=True, exist_ok=True)
    # Translate every N milliseconds to bin steps based on bin width
    step = max(1, int(round(float(args.save_every_ms) / (bin_us / 1000.0))))
    saved = []
    # Save every-N cumulative frame (not raw bins) within the truncated range
    cum_tmp = np.zeros((H, W), dtype=np.float32)
    for i in range(idx_lo, idx_hi + 1):
        cum_tmp += frames[i].astype(np.float32)
        if (i - idx_lo) % step == 0 or i == idx_hi:
            wl = float(wl_seq[i])
            stem_name = f"cumsum_{i:04d}_{wl:.1f}nm_bin{int(bin_us/1000)}ms"
            save_frame(per_bin_dir / stem_name, cum_tmp, args.save_png, cmap="coolwarm")
            save_frame(per_bin_dir_gray / stem_name, cum_tmp, args.save_png, cmap="gray")
            saved.append({"index": i, "wavelength_nm": wl, "mode": "cumsum", "folders": [str(per_bin_dir), str(per_bin_dir_gray)]})

    # Save mapping JSON
    mapping = {
        "config": str(args.config.resolve()),
        "dataset": str(dataset_dir),
        "segment": str(seg_npz),
        "bin_width_us": bin_us,
        "slope_nm_per_ms": slope,
        "intercept_nm": intercept,
        "indices": {
            "idx_380": idx_lo,
            "idx_700": idx_hi
        },
        "cumsum": {
            "start_index": idx_lo,
            "end_index": idx_hi,
            "num_bins": int(idx_hi - idx_lo + 1)
        },
        "truncate_nm": [lo_nm, hi_nm],
        "wavelength_nm": [float(v) for v in wl_seq.tolist()],
        "bins": bin_meta,
        "saved_every_n": int(step),
        "saved_frames": saved,
        "notes": "Frames are accumulated with weights (pos=+pos_scale, neg=-neg_scale) and compensated by a_avg,b_avg from config."
    }
    with (out_dir / "index_wavelength_mapping.json").open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp, indent=2)

    print("Output directory:", out_dir)
    print("Grayscale directory:", out_dir_gray)
    print("Idx for 380nm:", idx_lo, "Idx for 700nm:", idx_hi)


if __name__ == "__main__":
    main()
