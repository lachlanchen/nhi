#!/usr/bin/env python3
"""
Compare reconstruction/noise metrics when fusing scan segments.

Scenarios evaluated on a given dataset directory (with segmented NPZ files):
  1) Single segment: first Forward only
  2) Forward + Backward pair: Forward_1 + Backward_2 (time-reversed, polarity-flipped)
  3) Three cycles (F+B): {F1,B2,F3,B4,F5,B6}

Rules:
  - Use the learned parameters (a,b) from the first Forward segment only
    and apply them to all segments (including reversed Backward segments).
  - For backward segments, reverse time (t' = t_min + t_max - t) and invert
    polarity (p' = -p). If polarities are 0/1, map with p' = 1 - p.
  - Build compensated frames with 50 ms bins (configurable). For fusion,
    average per-bin frames across segments (common number of bins), then
    compute noise metrics on the fused frames.

Metrics (per-bin then averaged):
  - std_spatial: standard deviation of the frame after mean subtraction
  - mad_spatial: median absolute deviation after mean subtraction

Outputs:
  - A timestamped folder under publication_code/figures/ with:
      * summary JSON of metrics for each scenario
      * CSV with per-bin metrics
      * optional montage PNG of the first ~20 bins (compensated frames)

Example:
  python publication_code/compare_segment_fusion.py \
    --dataset scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638 \
    --bin-width-us 50000 --sensor-width 1280 --sensor-height 720 --save-png
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Reuse helpers from the Figure 4 tooling
# Locate repo root and make helpers importable like other figure scripts
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "publication_code"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from figure04_rescaled import (
    load_segment_events,
    find_param_file,
    load_params,
    auto_scale_neg_weight,
    accumulate_bin,
    subtract_background,
    setup_style,
)


@dataclass
class SegmentInfo:
    path: Path
    index: int
    direction: str  # "Forward" or "Backward"


def list_segments(segments_dir: Path) -> List[SegmentInfo]:
    segs: List[SegmentInfo] = []
    for p in sorted(segments_dir.glob("Scan_*_*(Forward|Backward)_events.npz")):
        name = p.stem
        # Expect names like: Scan_1_Forward_events
        parts = name.split("_")
        try:
            idx = int(parts[1])
            direction = parts[2]
        except Exception:
            continue
        if direction not in ("Forward", "Backward"):
            continue
        segs.append(SegmentInfo(path=p, index=idx, direction=direction))
    # Fallback if glob didn't expand group (for shells): explicit two globs
    if not segs:
        for direction in ("Forward", "Backward"):
            for p in sorted(segments_dir.glob(f"Scan_*_{direction}_events.npz")):
                parts = p.stem.split("_")
                try:
                    idx = int(parts[1])
                except Exception:
                    continue
                segs.append(SegmentInfo(path=p, index=idx, direction=direction))
    segs.sort(key=lambda s: s.index)
    return segs


def find_segments_root(dataset_dir: Path) -> Path:
    # Use the first *_segments directory under dataset_dir
    candidates = sorted(dataset_dir.glob("*_segments"))
    if not candidates:
        raise FileNotFoundError(f"No segments folder under {dataset_dir}")
    return candidates[0]


def ensure_polarity_pm1(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    if p.min() >= 0.0 and p.max() <= 1.0:
        p = (p - 0.5) * 2.0
    return p


def reverse_backward_time_and_polarity(t: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Reverse time within the segment window: t' = t_min + t_max - t
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    t_rev = (tmin + tmax) - t
    # Invert polarity: for -1/1 it's simply -p; for 0/1 mapping is 1-p, but we already map to -1/1 elsewhere
    p = ensure_polarity_pm1(p)
    p_rev = -p
    return t_rev.astype(np.float32), p_rev.astype(np.float32)


def compute_neg_scale_from_forward(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sensor_area: float) -> float:
    # Use a 5 ms step like Fig.4 tools
    _, series = auto_scale_neg_weight(t, p, sensor_area=sensor_area, step_us=5000.0, pos_scale=1.0, return_series=True)  # type: ignore[misc]
    neg_scale = float(series.get("neg_scale_final", np.array([1.0], dtype=np.float32))[0])
    if not np.isfinite(neg_scale) or neg_scale <= 0:
        neg_scale = 1.0
    return neg_scale


def build_frames_for_segment(
    seg: SegmentInfo,
    params: Dict[str, np.ndarray],
    sensor_shape: Tuple[int, int],
    bin_width_us: float,
    neg_scale: float,
) -> Tuple[List[np.ndarray], int, float, float]:
    """Return list of compensated frames for this segment and its common bin count.

    Also returns (num_bins, tmin, tmax) for later reference.
    """
    x, y, t, p = load_segment_events(seg.path)
    p = ensure_polarity_pm1(p)
    if seg.direction.lower() == "backward":
        t, p = reverse_backward_time_and_polarity(t, p)
    # Fast compensation: t' = t - (a_avg x + b_avg y)
    a_avg = float(np.mean(params["a_params"]))
    b_avg = float(np.mean(params["b_params"]))
    t_comp = t - (a_avg * x + b_avg * y)

    tmin = float(np.min(t_comp))
    tmax = float(np.max(t_comp))
    duration = max(0.0, tmax - tmin)
    num_bins = int(np.floor(duration / bin_width_us))
    if num_bins <= 0:
        return [], 0, tmin, tmax

    # Polarity weights for compensated branch
    weights = np.where(p >= 0, 1.0, -neg_scale).astype(np.float32)

    frames: List[np.ndarray] = []
    for idx in range(num_bins):
        start = tmin + idx * bin_width_us
        end = start + bin_width_us
        mask = (t_comp >= start) & (t_comp < end)
        frame = accumulate_bin(x, y, mask, weights, sensor_shape)
        frames.append(subtract_background(frame))
    return frames, num_bins, tmin, tmax


def fuse_frames_across_segments(frames_list: List[List[np.ndarray]], max_bins: int) -> List[np.ndarray]:
    """Average frames per bin index across segments, up to max_bins.

    Assumes each inner list has at least max_bins frames.
    """
    fused: List[np.ndarray] = []
    for b in range(max_bins):
        stack = np.stack([frames[b] for frames in frames_list], axis=0)
        fused.append(np.mean(stack, axis=0))
    return fused


def frame_metrics(frames: Sequence[np.ndarray]) -> Dict[str, float]:
    if not frames:
        return {"avg_std": float("nan"), "avg_mad": float("nan")}
    stds = [float(np.std(f)) for f in frames]
    mads = [float(np.median(np.abs(f - np.median(f)))) for f in frames]
    return {
        "avg_std": float(np.mean(stds)),
        "avg_mad": float(np.mean(mads)),
    }


def montage_20_bins(
    frames: Sequence[np.ndarray],
    out_stem: Path,
    save_png: bool,
    save_pdf: bool,
) -> None:
    setup_style()
    k = min(20, len(frames))
    if k == 0:
        return
    cols = 10
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.0 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < k:
            im = ax.imshow(frames[i], cmap="coolwarm")
            ax.set_title(f"Bin {i}", fontsize=7)
    fig.tight_layout()
    if save_png:
        fig.savefig(str(out_stem) + ".png", dpi=200, bbox_inches="tight")
    if save_pdf:
        fig.savefig(str(out_stem) + ".pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare fusion performance across scan segments")
    ap.add_argument("--dataset", type=Path, required=True, help="Path to dataset folder containing *_segments")
    ap.add_argument("--bin-width-us", type=float, default=50000.0)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    seg_dir = find_segments_root(dataset_dir)
    segments = list_segments(seg_dir)
    if not segments:
        raise FileNotFoundError(f"No segments found in {seg_dir}")

    # Identify first Forward, first F+B, and up to three F+B
    forwards = [s for s in segments if s.direction == "Forward"]
    backwards = [s for s in segments if s.direction == "Backward"]
    if not forwards:
        raise RuntimeError("No Forward segment found")
    first_forward = forwards[0]

    # Derive param file from first forward
    param_file = find_param_file(first_forward.path)
    if param_file is None:
        raise FileNotFoundError(f"No learned params found alongside {first_forward.path}")
    params = load_params(param_file)

    # Compute neg_scale from first forward
    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)
    x_f, y_f, t_f, p_f = load_segment_events(first_forward.path)
    p_f = ensure_polarity_pm1(p_f)
    neg_scale = compute_neg_scale_from_forward(x_f, y_f, t_f, p_f, sensor_area)

    # Scenario A: Forward only (first forward)
    frames_A, nb_A, _, _ = build_frames_for_segment(first_forward, params, sensor_shape, args.bin_width_us, neg_scale)

    # Scenario B: One F+B pair (assume Backward with next index after first forward)
    pair_segments: List[SegmentInfo] = [first_forward]
    # Find the Backward with the same or next index
    cand_back = [s for s in segments if s.direction == "Backward" and s.index in (first_forward.index, first_forward.index + 1)]
    if not cand_back and backwards:
        cand_back = [backwards[0]]
    if cand_back:
        pair_segments.append(sorted(cand_back, key=lambda s: s.index)[0])

    frames_lists_B: List[List[np.ndarray]] = []
    for s in pair_segments:
        frames, nb, _, _ = build_frames_for_segment(s, params, sensor_shape, args.bin_width_us, neg_scale)
        if nb > 0:
            frames_lists_B.append(frames)
    common_bins_B = min((len(frames) for frames in frames_lists_B), default=0)
    frames_B = fuse_frames_across_segments(frames_lists_B, common_bins_B) if common_bins_B > 0 else []

    # Scenario C: Three F+B cycles (up to 6 segments)
    cycles: List[SegmentInfo] = []
    for idx in range(first_forward.index, first_forward.index + 6):
        match = [s for s in segments if s.index == idx]
        if match:
            cycles.append(match[0])
    frames_lists_C: List[List[np.ndarray]] = []
    for s in cycles:
        frames, nb, _, _ = build_frames_for_segment(s, params, sensor_shape, args.bin_width_us, neg_scale)
        if nb > 0:
            frames_lists_C.append(frames)
    common_bins_C = min((len(frames) for frames in frames_lists_C), default=0)
    frames_C = fuse_frames_across_segments(frames_lists_C, common_bins_C) if common_bins_C > 0 else []

    # Metrics
    metrics_A = frame_metrics(frames_A)
    metrics_B = frame_metrics(frames_B)
    metrics_C = frame_metrics(frames_C)

    # Output directory
    out_root = (Path(__file__).resolve().parent / "figures").resolve()
    if args.output_dir is None:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"segment_fusion_compare_{stamp}"
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summaries
    summary = {
        "dataset": str(dataset_dir),
        "bin_width_us": float(args.bin_width_us),
        "sensor_shape": list(sensor_shape),
        "param_file": str(param_file),
        "neg_scale": float(neg_scale),
        "scenarios": {
            "Forward_only": metrics_A,
            "F_plus_B": metrics_B,
            "three_cycles_F_plus_B": metrics_C,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    # Per-bin CSV for each scenario
    def write_bin_csv(name: str, frames: Sequence[np.ndarray]) -> None:
        path = out_dir / f"{name}_per_bin_metrics.csv"
        with path.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["bin_index", "std", "mad"])
            for i, f in enumerate(frames):
                std = float(np.std(f))
                mad = float(np.median(np.abs(f - np.median(f))))
                w.writerow([i, std, mad])

    write_bin_csv("forward_only", frames_A)
    if frames_B:
        write_bin_csv("f_plus_b", frames_B)
    if frames_C:
        write_bin_csv("three_cycles_f_plus_b", frames_C)

    # Optional montages of first ~20 frames
    if args.save_png or args.save_pdf:
        montage_20_bins(frames_A, out_dir / "forward_only_montage", args.save_png, args.save_pdf)
        if frames_B:
            montage_20_bins(frames_B, out_dir / "f_plus_b_montage", args.save_png, args.save_pdf)
        if frames_C:
            montage_20_bins(frames_C, out_dir / "three_cycles_f_plus_b_montage", args.save_png, args.save_pdf)

    print("Comparison complete. Outputs:")
    print(f"  {out_dir}/summary.json")
    print(f"  {out_dir}/*_per_bin_metrics.csv")
    if args.save_png:
        print(f"  {out_dir}/*_montage.png")


if __name__ == "__main__":
    main()
