#!/usr/bin/env python3
"""Dynamic sliding-window spectrum comparison.

Unlike ``compare_reconstruction_sliding.py`` (fixed start/stride windows), this
variant maintains a growing window until it spans ``window_ms`` worth of
compensated event time. Once the window exceeds that duration, older events are
discarded so each evaluation reflects the most recent ``window_ms`` of data.
The resulting series is aligned to spectrometer ground truth via plateau
matching, identical to the cumulative workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

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


def dynamic_window_counts(
    times_us: np.ndarray,
    polarities: np.ndarray,
    window_us: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute running counts within the last ``window_us`` for each event.

    Args:
        times_us: event timestamps (µs), arbitrary order.
        polarities: +1 for positive events, -1 for negative events.
        window_us: window width in microseconds.

    Returns:
        event_times_us: sorted timestamps corresponding to each sample.
        pos_counts: number of positive events inside the window per sample.
        neg_counts: number of negative events inside the window per sample.
    """
    if times_us.size == 0:
        raise ValueError("No events supplied for dynamic window computation")

    order = np.argsort(times_us)
    times_sorted = times_us[order].astype(np.float64, copy=False)
    pol_sorted = polarities[order]

    pos_flags = (pol_sorted >= 0).astype(np.float64)
    neg_flags = (pol_sorted < 0).astype(np.float64)

    pos_prefix = np.concatenate([[0.0], np.cumsum(pos_flags)])
    neg_prefix = np.concatenate([[0.0], np.cumsum(neg_flags)])

    window_starts = times_sorted - window_us
    left_idx = np.searchsorted(times_sorted, window_starts, side="left")
    right_idx = np.arange(times_sorted.size) + 1  # prefix index is +1

    pos_counts = pos_prefix[right_idx] - pos_prefix[left_idx]
    neg_counts = neg_prefix[right_idx] - neg_prefix[left_idx]

    return times_sorted, pos_counts, neg_counts


def build_dynamic_series(
    segment_npz: Path,
    window_ms: float,
    pos_scale: float,
    auto_bounds: Tuple[float, float],
    plateau_frac: float,
    sensor_width: int,
    sensor_height: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate the exponential, polarity-weighted series using dynamic windows."""
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    params_file = vcw.find_param_file_for_segment(str(segment_npz))
    if params_file is None:
        raise FileNotFoundError("No learned parameter NPZ found next to segment")
    params = vcw.load_parameters_from_npz(params_file)

    t_comp, *_ = vcw.compute_fast_compensated_times(
        x, y, t, params["a_params"], params["b_params"]
    )

    window_us = window_ms * 1000.0
    times_us, pos_counts, neg_counts = dynamic_window_counts(t_comp, p, window_us)

    sensor_area = float(sensor_width * sensor_height)

    def build_series(neg_scale: float) -> np.ndarray:
        net = pos_scale * pos_counts - neg_scale * neg_counts
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
    time_ms = (times_us - times_us[0]) / 1000.0
    return time_ms, series, chosen_neg


def format_suffix(window_ms: float) -> str:
    val = int(window_ms) if abs(window_ms - round(window_ms)) < 1e-6 else str(window_ms).replace(".", "p")
    return f"dynamic_window{val}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic sliding-window reconstruction vs ground truth")
    parser.add_argument(
        "--segment",
        type=Path,
        default=REPO_ROOT
        / "scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz",
        help="Segmented NPZ to analyse",
    )
    parser.add_argument("--window_ms", type=float, default=100.0, help="Window width (ms)")
    parser.add_argument("--sensor_width", type=int, default=1280, help="Sensor width (pixels)")
    parser.add_argument("--sensor_height", type=int, default=720, help="Sensor height (pixels)")
    parser.add_argument("--pos_scale", type=float, default=1.0, help="Weight for positive polarity events")
    parser.add_argument("--auto_neg_min", type=float, default=0.1, help="Lower bound for auto negative scale search")
    parser.add_argument("--auto_neg_max", type=float, default=3.0, help="Upper bound for auto negative scale search")
    parser.add_argument(
        "--plateau_frac",
        type=float,
        default=0.05,
        help="Fraction of samples used to estimate start/end plateaus",
    )
    args = parser.parse_args()

    if args.window_ms <= 0:
        raise ValueError("window_ms must be positive")

    time_ms, series, neg_scale = build_dynamic_series(
        args.segment,
        args.window_ms,
        args.pos_scale,
        (args.auto_neg_min, args.auto_neg_max),
        args.plateau_frac,
        args.sensor_width,
        args.sensor_height,
    )

    smooth_span = max(21, int(len(series) // 2000) | 1)
    smooth_series = moving_average(series, smooth_span)
    recon_region = detect_active_region(smooth_series)
    recon_norm = normalise_curve(series, recon_region)

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

    gt_norm_curves = []
    for (wl, val), region in zip(gt_curves, gt_regions):
        gt_norm_curves.append((wl, normalise_curve(val, region)))

    suffix = format_suffix(args.window_ms)
    out_dir = REPO_ROOT / "groundtruth_spectrum"
    out_dir.mkdir(exist_ok=True)
    plot_path = out_dir / f"reconstruction_vs_groundtruth_{suffix}.png"
    report_path = out_dir / f"reconstruction_time_to_wavelength_{suffix}.txt"

    plt.figure(figsize=(11, 6))
    plt.axvspan(380, 780, color="0.85", alpha=0.5, label="Visible band (380-780 nm)")

    for (wl, curve), path in zip(gt_norm_curves, gt_paths):
        plt.plot(wl, curve, label=f"Ground truth ({path.name})", linewidth=1.1)

    stride = max(1, len(recon_wavelengths) // 5000)
    plt.plot(
        recon_wavelengths[::stride],
        recon_norm[::stride],
        color="crimson",
        linewidth=1.0,
        label="Dynamic-window reconstruction",
    )

    plt.axvline(gt_start_nm, color="k", linestyle="--", linewidth=1.0, label="Aligned blue edge")
    plt.axvline(gt_end_nm, color="k", linestyle="-.", linewidth=1.0, label="Aligned red edge")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalised intensity (a.u.)")
    plt.title(
        "Dynamic sliding-window spectrum vs. ground truth"
        f" (window={args.window_ms:.1f} ms)"
    )
    plt.xlim(
        min(recon_wavelengths.min(), gt_norm_curves[0][0][0]),
        max(recon_wavelengths.max(), gt_norm_curves[0][0][-1]),
    )
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)

    slope = recon_scale
    intercept = gt_start_nm - slope * recon_start_ms
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Dynamic sliding-window reconstruction vs. ground-truth alignment\n")
        f.write(f"Neg polarity scale (auto): {neg_scale:.6f}\n")
        f.write(f"Window width (ms): {args.window_ms:.3f}\n")
        f.write(f"Reconstruction start (ms): {recon_start_ms:.3f}\n")
        f.write(f"Reconstruction end (ms): {recon_end_ms:.3f}\n")
        f.write(f"Ground-truth start (nm): {gt_start_nm:.3f}\n")
        f.write(f"Ground-truth end (nm): {gt_end_nm:.3f}\n")
        f.write("Mapping formula (ms → nm):\n")
        f.write(f"  λ(t_ms) = {slope:.6f} * t_ms + {intercept:.3f}\n")
        f.write(
            "Where t_ms is the compensated event timeline measured from the first sample.\n"
        )

    print(f"Saved dynamic-window overlay → {plot_path}")
    print(f"Saved mapping details → {report_path}")


if __name__ == "__main__":
    main()
