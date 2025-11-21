#!/usr/bin/env python3
"""Sliding-window spectrum comparison between compensated events and ground truth.

This script mirrors ``compare_reconstruction_to_gt.py`` but replaces the
infinite cumulative trace with a fixed-duration, sliding window accumulation.
The event-camera spectrum is reconstructed by summing polarity-weighted events
inside each window, the window is shifted by a configurable stride, and the
resulting sequence is aligned to both spectrometer captures by matching their
flat plateaus.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import visualize_cumulative_weighted as vcw  # noqa: E402
from groundtruth_spectrum.compare_reconstruction_to_gt import (  # noqa: E402
    detect_active_region,
    load_ground_truth,
    moving_average,
    normalise_curve,
)


def sliding_window_sum(
    times: np.ndarray,
    weights: np.ndarray,
    window_us: float,
    stride_us: float,
    t_min: float,
    t_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-window sums and window start times (both in µs)."""
    if times.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(times, kind="mergesort")
    times_sorted = times[order]
    weights_sorted = weights[order]

    if window_us <= 0 or stride_us <= 0:
        raise ValueError("window_us and stride_us must be positive")
    if t_max <= t_min:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    total_span = t_max - t_min
    if total_span <= window_us:
        starts = np.array([t_min], dtype=np.float64)
    else:
        count = int(math.floor((total_span - window_us) / stride_us)) + 1
        starts = t_min + stride_us * np.arange(count, dtype=np.float64)
        last_start = t_max - window_us
        if starts.size == 0 or starts[-1] < last_start - 1e-6:
            starts = np.append(starts, last_start)

    ends = starts + window_us
    prefix = np.concatenate([[0.0], np.cumsum(weights_sorted, dtype=np.float64)])
    right = np.searchsorted(times_sorted, ends, side="right")
    left = np.searchsorted(times_sorted, starts, side="left")
    sums = prefix[right] - prefix[left]
    return starts, sums


def build_sliding_series(
    segment_npz: Path,
    window_ms: float,
    stride_ms: float,
    pos_scale: float,
    auto_bounds: Tuple[float, float],
    plateau_frac: float,
    sensor_width: int,
    sensor_height: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute sliding-window, polarity-weighted series (exp transformed)."""
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    params_file = vcw.find_param_file_for_segment(str(segment_npz))
    if params_file is None:
        raise FileNotFoundError("No learned parameter NPZ found next to segment")
    params = vcw.load_parameters_from_npz(params_file)
    t_comp, *_ = vcw.compute_fast_compensated_times(x, y, t, params["a_params"], params["b_params"])

    window_us = window_ms * 1000.0
    stride_us = stride_ms * 1000.0
    t_min = float(np.min(t_comp))
    t_max = float(np.max(t_comp))

    sensor_area = float(sensor_width * sensor_height)

    pos_mask = p >= 0
    neg_mask = ~pos_mask

    starts_pos, counts_pos = sliding_window_sum(
        t_comp[pos_mask],
        np.ones(np.count_nonzero(pos_mask), dtype=np.float64),
        window_us,
        stride_us,
        t_min,
        t_max,
    )
    starts_neg, counts_neg = sliding_window_sum(
        t_comp[neg_mask],
        np.ones(np.count_nonzero(neg_mask), dtype=np.float64),
        window_us,
        stride_us,
        t_min,
        t_max,
    )

    if starts_pos.size and starts_neg.size and not np.array_equal(starts_pos, starts_neg):
        raise RuntimeError("Positive/negative windows misaligned; check inputs")

    if starts_pos.size == 0 and starts_neg.size == 0:
        raise RuntimeError("Sliding-window accumulation produced no samples")
    # Ensure both masks produced aligned start times.
    starts = starts_pos if starts_pos.size else starts_neg

    def build_series(neg_scale: float) -> np.ndarray:
        # Align counts arrays if one polarity had zero events in some windows.
        pos_vals = counts_pos if counts_pos.size else np.zeros_like(counts_neg)
        neg_vals = counts_neg if counts_neg.size else np.zeros_like(counts_pos)
        net = pos_scale * pos_vals - neg_scale * neg_vals
        per_pixel = net / sensor_area
        clipped = np.clip(per_pixel, -20.0, 20.0)
        return np.exp(clipped)

    def plateau_diff(neg_scale: float) -> float:
        series = build_series(neg_scale)
        k = len(series)
        if k <= 2:
            return 0.0
        n = max(5, int(plateau_frac * k))
        return float(np.mean(series[-n:]) - np.mean(series[:n]))

    neg_min, neg_max = auto_bounds
    f_min, f_max = plateau_diff(neg_min), plateau_diff(neg_max)
    if f_min * f_max < 0:
        for _ in range(40):
            mid = 0.5 * (neg_min + neg_max)
            f_mid = plateau_diff(mid)
            if abs(f_mid) < 1e-6:
                neg_min = neg_max = mid
                break
            if f_min * f_mid < 0:
                neg_max, f_max = mid, f_mid
            else:
                neg_min, f_min = mid, f_mid
        chosen_neg = 0.5 * (neg_min + neg_max)
    else:
        grid = np.linspace(neg_min, neg_max, 50)
        vals = np.array([abs(plateau_diff(g)) for g in grid])
        chosen_neg = float(grid[int(np.argmin(vals))])

    series = build_series(chosen_neg)
    centers_us = starts + 0.5 * window_us
    rel_time_ms = (centers_us - centers_us[0]) / 1000.0
    return rel_time_ms, series, chosen_neg


def format_suffix(window_ms: float, stride_ms: float) -> str:
    def normalize(val: float) -> str:
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return str(val).replace(".", "p")

    return f"window{normalize(window_ms)}ms_stride{normalize(stride_ms)}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sliding-window reconstruction to spectrometer ground truth")
    parser.add_argument(
        "--segment",
        type=Path,
        default=REPO_ROOT
        / "scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz",
        help="Segmented NPZ to analyse",
    )
    parser.add_argument("--window_ms", type=float, default=100.0, help="Sliding window width in milliseconds")
    parser.add_argument("--stride_ms", type=float, default=2.0, help="Sliding window stride in milliseconds")
    parser.add_argument("--sensor_width", type=int, default=1280, help="Sensor width (pixels)")
    parser.add_argument("--sensor_height", type=int, default=720, help="Sensor height (pixels)")
    parser.add_argument("--pos_scale", type=float, default=1.0, help="Weight for positive polarity events")
    parser.add_argument("--auto_neg_min", type=float, default=0.1, help="Lower bound for auto negative scale search")
    parser.add_argument("--auto_neg_max", type=float, default=3.0, help="Upper bound for auto negative scale search")
    parser.add_argument(
        "--plateau_frac",
        type=float,
        default=0.05,
        help="Fraction of windows used to estimate start/end plateaus",
    )
    args = parser.parse_args()

    window_ms = float(args.window_ms)
    stride_ms = float(args.stride_ms)
    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError("window_ms and stride_ms must be positive")

    time_ms, reconstruction_series, neg_scale = build_sliding_series(
        args.segment,
        window_ms,
        stride_ms,
        args.pos_scale,
        (args.auto_neg_min, args.auto_neg_max),
        args.plateau_frac,
        args.sensor_width,
        args.sensor_height,
    )

    smooth_span = max(21, int(len(reconstruction_series) // 200))
    smooth_series = moving_average(reconstruction_series, smooth_span)
    recon_region = detect_active_region(smooth_series)
    recon_norm = normalise_curve(reconstruction_series, recon_region)

    recon_start_ms = float(time_ms[recon_region.start_idx])
    recon_end_ms = float(time_ms[recon_region.end_idx])

    gt_paths = [
        REPO_ROOT / "groundtruth_spectrum/USB2F042671_16-04-36-993.txt",
        REPO_ROOT / "groundtruth_spectrum/USB2F042671_16-04-56-391.txt",
    ]
    gt_curves = []
    gt_regions = []
    for path in gt_paths:
        wl, val = load_ground_truth(path)
        mask = (wl >= 350.0) & (wl <= 900.0)
        wl = wl[mask]
        val = moving_average(val[mask], max(21, len(val) // 300))
        region = detect_active_region(val)
        gt_curves.append((wl, val))
        gt_regions.append(region)

    gt_start_nm = float(np.mean([wl[region.start_idx] for (wl, _), region in zip(gt_curves, gt_regions)]))
    gt_end_nm = float(np.mean([wl[region.end_idx] for (wl, _), region in zip(gt_curves, gt_regions)]))

    recon_scale = (gt_end_nm - gt_start_nm) / (recon_end_ms - recon_start_ms)
    recon_wavelengths = gt_start_nm + (time_ms - recon_start_ms) * recon_scale

    gt_norm = []
    for (wl, val), region in zip(gt_curves, gt_regions):
        gt_norm.append((wl, normalise_curve(val, region)))

    suffix = format_suffix(window_ms, stride_ms)
    out_dir = REPO_ROOT / "groundtruth_spectrum"
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"reconstruction_vs_groundtruth_{suffix}.png"
    report_path = out_dir / f"reconstruction_time_to_wavelength_{suffix}.txt"

    plt.figure(figsize=(11, 6))
    plt.axvspan(380, 780, color="0.85", alpha=0.5, label="Visible band (380-780 nm)")

    for (wl, curve), path in zip(gt_norm, gt_paths):
        plt.plot(wl, curve, label=f"Ground truth ({path.name})", linewidth=1.1)

    stride = max(1, len(recon_wavelengths) // 4000)
    plt.plot(
        recon_wavelengths[::stride],
        recon_norm[::stride],
        color="crimson",
        linewidth=1.0,
        label="Sliding-window reconstruction",
    )

    plt.axvline(gt_start_nm, color="k", linestyle="--", linewidth=1.0, label="Aligned blue edge")
    plt.axvline(gt_end_nm, color="k", linestyle="-.", linewidth=1.0, label="Aligned red edge")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalised intensity (a.u.)")
    plt.title(
        "Sliding-window spectrum vs. ground truth"
        f" (window={window_ms:.1f} ms, stride={stride_ms:.1f} ms)"
    )
    plt.xlim(
        min(recon_wavelengths.min(), gt_norm[0][0][0]),
        max(recon_wavelengths.max(), gt_norm[0][0][-1]),
    )
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)

    slope = recon_scale
    intercept = gt_start_nm - slope * recon_start_ms
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Sliding-window reconstruction vs. ground-truth alignment\n")
        f.write(f"Neg polarity scale (auto): {neg_scale:.6f}\n")
        f.write(f"Window width (ms): {window_ms:.3f}\n")
        f.write(f"Window stride (ms): {stride_ms:.3f}\n")
        f.write(f"Reconstruction start (ms): {recon_start_ms:.3f}\n")
        f.write(f"Reconstruction end (ms): {recon_end_ms:.3f}\n")
        f.write(f"Ground-truth start (nm): {gt_start_nm:.3f}\n")
        f.write(f"Ground-truth end (nm): {gt_end_nm:.3f}\n")
        f.write("Mapping formula (ms → nm):\n")
        f.write(f"  λ(t_ms) = {slope:.6f} * t_ms + {intercept:.3f}\n")
        f.write("Where t_ms is the compensated window-centre timeline measured from the earliest window.\n")

    print(f"Saved sliding-window overlay → {plot_path}")
    print(f"Saved mapping details → {report_path}")


if __name__ == "__main__":
    main()
