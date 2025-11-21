#!/usr/bin/env python3
"""Three-panel publication overlay: GT vs cumulative, log, and derivative.

Panels (left → right):

1. Cumulative exp-intensity: exp-cumulative reconstruction vs ground-truth.
2. Log-intensity: log of the normalised curves.
3. Spectral derivative vs events: d log(GT)/dλ vs event rates in 5 ms bins.

The script reuses the same compensated timeline and auto-tuned negative
polarity scale as ``compare_publication_cumulative.py`` so all three panels
share a consistent λ(t) mapping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import visualize_cumulative_weighted as vcw  # noqa: E402
from compare_reconstruction_to_gt import (  # noqa: E402
    detect_active_region,
    load_ground_truth,
    moving_average,
    normalise_curve,
)
from compare_publication_cumulative import (  # noqa: E402
    detect_visible_edges,
    ensure_output_dir,
    publication_style,
)


def build_cumulative_and_bins(
    segment_npz: Path,
    step_ms: float,
    bin_ms: float,
    pos_scale: float,
    auto_bounds: Tuple[float, float],
    plateau_frac: float,
    sensor_width: int,
    sensor_height: int,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return cumulative (time_ms, exp_series, neg_scale) and 5 ms bin series.

    - Uses compensated times for both cumulative and 5 ms bins.
    - Auto-tunes neg_scale on the cumulative exp-series.
    """
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    t_min, t_max = float(np.min(t)), float(np.max(t))

    param_file = vcw.find_param_file_for_segment(str(segment_npz))
    if param_file is None:
        raise FileNotFoundError("No learned parameter NPZ found next to segment")
    params = vcw.load_parameters_from_npz(param_file)
    t_comp, *_ = vcw.compute_fast_compensated_times(x, y, t, params["a_params"], params["b_params"])

    pos_mask = p >= 0
    neg_mask = ~pos_mask
    hw = float(sensor_width * sensor_height)

    # Cumulative series (exp domain)
    step_us_cum = step_ms * 1000.0
    sums_pos_cum, edges_ms_cum = vcw.base_binned_sums_weighted(
        t_comp[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.float32), t_min, t_max, step_us_cum
    )
    sums_neg_cum, _ = vcw.base_binned_sums_weighted(
        t_comp[neg_mask], np.ones(np.count_nonzero(neg_mask), dtype=np.float32), t_min, t_max, step_us_cum
    )

    def build_cum(neg_scale: float) -> np.ndarray:
        step_sums = pos_scale * sums_pos_cum - neg_scale * sums_neg_cum
        return np.cumsum(step_sums) / hw

    def plateau_diff(neg_scale: float) -> float:
        series = np.exp(build_cum(neg_scale))
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

    cum_linear = build_cum(chosen_neg)
    cum_exp = np.exp(cum_linear)
    time_ms_cum = edges_ms_cum - edges_ms_cum[0]

    # 5 ms (or user-specified) event bins using the same compensated times.
    step_us_bin = bin_ms * 1000.0
    sums_pos_bin, edges_ms_bin = vcw.base_binned_sums_weighted(
        t_comp[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.float32), t_min, t_max, step_us_bin
    )
    sums_neg_bin, _ = vcw.base_binned_sums_weighted(
        t_comp[neg_mask], np.ones(np.count_nonzero(neg_mask), dtype=np.float32), t_min, t_max, step_us_bin
    )
    # For the derivative-style panel, use simple signed event counts per bin.
    net_bin = sums_pos_bin - sums_neg_bin

    # Align bin timeline to cumulative zero (edges_ms_cum[0]).
    time_ms_bin = edges_ms_bin - edges_ms_cum[0]

    return time_ms_cum, cum_exp, chosen_neg, time_ms_bin, net_bin


def load_gt_curves_from_paths(files: Sequence[Path]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for txt in files:
        wl, val = load_ground_truth(txt)
        curves.append((txt.stem, wl, val))
    if not curves:
        raise FileNotFoundError("No ground-truth files supplied")
    return curves


def align_reconstruction_to_gt(
    time_ms: np.ndarray,
    series_exp: np.ndarray,
    gt_curves: Sequence[Tuple[str, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Return (wl_recon, recon_norm, slope, intercept, wl_gt, gt_norm) using first GT."""
    gt_start_nm, gt_end_nm = detect_visible_edges(gt_curves)

    # Normalise first GT curve in detected active region.
    name0, wl0_raw, val0_raw = gt_curves[0]
    mask0 = (wl0_raw >= 300.0) & (wl0_raw <= 900.0)
    wl0 = wl0_raw[mask0]
    val0 = val0_raw[mask0]
    smooth0 = moving_average(val0, max(21, len(val0) // 300))
    region0 = detect_active_region(smooth0)
    gt_norm = normalise_curve(smooth0, region0)

    # Align reconstruction to wavelength domain via active-region endpoints.
    smooth_rec = moving_average(series_exp, max(21, int(len(series_exp) // 200) | 1))
    region_rec = detect_active_region(smooth_rec)
    recon_norm = normalise_curve(series_exp, region_rec)
    t0 = float(time_ms[region_rec.start_idx])
    t1 = float(time_ms[region_rec.end_idx])
    if t1 <= t0:
        raise ValueError("Detected non-increasing active region in reconstruction")
    slope = (gt_end_nm - gt_start_nm) / (t1 - t0)
    intercept = gt_start_nm - slope * t0
    wl_recon = slope * time_ms + intercept

    return wl_recon, recon_norm, slope, intercept, wl0, gt_norm


def parse_args() -> argparse.Namespace:
    default_segment = (
        REPO_ROOT
        / "scan_angle_20_lumileds/angle_20_blank_20250922_170433/"
        "angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz"
    )
    parser = argparse.ArgumentParser(description="Three-panel GT vs reconstruction overlay (cumulative/log/derivative)")
    parser.add_argument("--segment", type=Path, default=default_segment, help="Segmented NPZ to analyse")
    parser.add_argument(
        "--gt_files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more spectrometer TXT files (first is used for shape/derivative)",
    )
    parser.add_argument("--step_ms", type=float, default=2.0, help="Cumulative step size in milliseconds")
    parser.add_argument("--bin_ms", type=float, default=5.0, help="Event bin width for derivative panel (ms)")
    parser.add_argument("--sensor_width", type=int, default=1280)
    parser.add_argument("--sensor_height", type=int, default=720)
    parser.add_argument("--pos_scale", type=float, default=1.0)
    parser.add_argument("--auto_neg_min", type=float, default=0.1)
    parser.add_argument("--auto_neg_max", type=float, default=3.0)
    parser.add_argument("--plateau_frac", type=float, default=0.05)
    parser.add_argument("--xlim", type=float, nargs=2, default=(300.0, 900.0))
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "align_bg_vs_gt_code")
    parser.add_argument("--show", action="store_true", help="Display figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publication_style()

    gt_curves = load_gt_curves_from_paths(args.gt_files)

    time_ms_cum, cum_exp, neg_scale, time_ms_bin, net_bin = build_cumulative_and_bins(
        args.segment,
        step_ms=args.step_ms,
        bin_ms=args.bin_ms,
        pos_scale=args.pos_scale,
        auto_bounds=(args.auto_neg_min, args.auto_neg_max),
        plateau_frac=args.plateau_frac,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
    )

    wl_recon, recon_norm, slope, intercept, wl_gt, gt_norm = align_reconstruction_to_gt(
        time_ms_cum, cum_exp, gt_curves
    )

    # Map event bins to wavelength using the same λ(t) mapping.
    wl_bins = slope * time_ms_bin + intercept

    # Derivative of log(GT) and normalised event rates (minimal processing).
    eps = 1e-3
    log_gt = np.log(np.clip(gt_norm, eps, None))
    dlog_gt = np.gradient(log_gt, wl_gt)
    if np.max(np.abs(dlog_gt)) > 0:
        dlog_gt_norm = dlog_gt / np.max(np.abs(dlog_gt))
    else:
        dlog_gt_norm = dlog_gt

    # Light smoothing and centering on event bins in wavelength domain.
    if net_bin.size:
        smooth_events = moving_average(net_bin, max(5, len(net_bin) // 200 | 1))
        smooth_events = smooth_events - float(np.mean(smooth_events))
    else:
        smooth_events = net_bin
    if np.max(np.abs(smooth_events)) > 0:
        events_norm = smooth_events / np.max(np.abs(smooth_events))
    else:
        events_norm = smooth_events

    # Prepare figure (three columns, shared x-axis)
    out_dir = ensure_output_dir(args.output_root)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True)
    ax1, ax2, ax3 = axes

    xmin, xmax = args.xlim

    # Panel 1: cumulative exp-intensity
    ax1.axvspan(380, 780, color="0.92", zorder=0)
    # SPD (ground-truth) vs event-based reconstruction
    ax1.plot(wl_gt, gt_norm, color="#1f77b4", label="SPD")
    ax1.plot(wl_recon, recon_norm, color="#2ca02c", label="Events")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normalised intensity")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: log-intensity
    ax2.axvspan(380, 780, color="0.92", zorder=0)
    ax2.plot(wl_gt, log_gt, color="#1f77b4", label="GT")
    ax2.plot(wl_recon, np.log(np.clip(recon_norm, eps, None)), color="#2ca02c", label="Recon")
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("log intensity")
    ax2.grid(alpha=0.3)

    # Panel 3: spectral derivative vs events (5 ms bins)
    ax3.axvspan(380, 780, color="0.92", zorder=0)
    ax3.plot(wl_gt, dlog_gt_norm, color="#1f77b4")
    ax3.plot(wl_bins, events_norm, color="#2ca02c")
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Normalised d log(SPD)/dλ and event rate")
    ax3.grid(alpha=0.3)

    fig.tight_layout()

    seg_name = args.segment.stem.replace("_events", "")
    out_name = f"three_panel_{args.bin_ms:.0f}ms_{seg_name}.png"
    fig.savefig(out_dir / out_name, dpi=300)

    if args.show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    print(f"Saved three-panel overlay to: {out_dir / out_name}")


if __name__ == "__main__":
    main()
