#!/usr/bin/env python3
"""Comprehensive reconstruction vs. ground-truth comparison for LED 2835 runs.

This script reproduces the compensated cumulative spectrum (auto-scaled,
exponential form) and two sliding-window variants, then aligns each series to
spectrometer captures by matching their flat plateaus. The output plot overlays
all reconstructions against both ground-truth files, highlights the inferred
visible band, and writes alignment formulas to a summary file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use existing tooling for event processing and alignment helpers
import visualize_cumulative_weighted as vcw  # noqa: E402
from groundtruth_spectrum.compare_reconstruction_to_gt import (  # noqa: E402
    detect_active_region,
    load_ground_truth,
    moving_average,
    normalise_curve,
)
from groundtruth_spectrum.compare_reconstruction_sliding import (  # noqa: E402
    build_sliding_series as build_fixed_window_series,
)
from groundtruth_spectrum.compare_reconstruction_sliding_dynamic import (  # noqa: E402
    build_dynamic_series,
)
from groundtruth_spectrum.compare_light_grating import (  # noqa: E402
    build_product as build_grating_product,
    load_grating_efficiencies,
    load_light_spd,
)


@dataclass
class SeriesResult:
    label: str
    time_ms: np.ndarray
    series: np.ndarray
    neg_scale: float
    window_ms: float | None = None
    stride_ms: float | None = None


@dataclass
class Mapping:
    name: str
    neg_scale: float
    window_ms: float | None
    stride_ms: float | None
    recon_start_ms: float
    recon_end_ms: float
    gt_start_nm: float
    gt_end_nm: float
    slope: float
    intercept: float


GROOVE_DEFAULT = 600
LIGHT_SPD_FILE = REPO_ROOT / 'light_spd.csv'
GRATING_FILE = REPO_ROOT / 'diffraction_grating.csv'

LIGHT_VISIBLE_MIN = 380.0
LIGHT_VISIBLE_MAX = 780.0


def build_cumulative_series(
    segment_npz: Path,
    step_ms: float,
    pos_scale: float,
    auto_bounds: Tuple[float, float],
    plateau_frac: float,
    sensor_width: int,
    sensor_height: int,
) -> SeriesResult:
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    t_min, t_max = float(np.min(t)), float(np.max(t))
    step_us = step_ms * 1000.0
    pos_mask = p >= 0
    neg_mask = ~pos_mask

    sums_pos, edges_ms = vcw.base_binned_sums_weighted(
        t[pos_mask],
        np.ones(np.count_nonzero(pos_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )
    sums_neg, _ = vcw.base_binned_sums_weighted(
        t[neg_mask],
        np.ones(np.count_nonzero(neg_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )

    def build_cum(neg_scale: float) -> np.ndarray:
        hw = float(sensor_width * sensor_height)
        return np.exp(np.cumsum(pos_scale * sums_pos - neg_scale * sums_neg) / hw)

    def plateau_diff(neg_scale: float) -> float:
        series = build_cum(neg_scale)
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
        neg_scale = 0.5 * (neg_min + neg_max)
    else:
        grid = np.linspace(neg_min, neg_max, 50)
        diffs = np.array([abs(plateau_diff(g)) for g in grid])
        neg_scale = float(grid[int(np.argmin(diffs))])

    series = build_cum(neg_scale)
    rel_time_ms = edges_ms - edges_ms[0]
    return SeriesResult(
        label=f"Cumulative (step={step_ms:.1f}ms)",
        time_ms=rel_time_ms,
        series=series,
        neg_scale=neg_scale,
        window_ms=None,
        stride_ms=step_ms,
    )


def prepare_series(
    label: str,
    time_ms: np.ndarray,
    series: np.ndarray,
    neg_scale: float,
    window_ms: float | None,
    stride_ms: float | None,
) -> SeriesResult:
    return SeriesResult(
        label=label,
        time_ms=time_ms,
        series=series,
        neg_scale=neg_scale,
        window_ms=window_ms,
        stride_ms=stride_ms,
    )


def load_ground_truth_curves(gt_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for txt in sorted(gt_dir.glob("*.txt")):
        wl, val = load_ground_truth(txt)
        curves.append((txt.name, wl, val))
    if not curves:
        raise FileNotFoundError(f"No .txt ground-truth files found in {gt_dir}")
    return curves


def detect_visible_region(curves: Sequence[Tuple[str, np.ndarray, np.ndarray]]) -> Tuple[float, float, List[Tuple[str, float, float]]]:
    starts: List[float] = []
    ends: List[float] = []
    per_curve: List[Tuple[str, float, float]] = []
    for name, wl, val in curves:
        mask = (wl >= 350.0) & (wl <= 900.0)
        wl_sel = wl[mask]
        val_sel = moving_average(val[mask], max(21, len(val) // 300))
        region = detect_active_region(val_sel)
        start_nm = float(wl_sel[region.start_idx])
        end_nm = float(wl_sel[region.end_idx])
        per_curve.append((name, start_nm, end_nm))
        starts.append(start_nm)
        ends.append(end_nm)
    return float(np.mean(starts)), float(np.mean(ends)), per_curve


def align_series(
    result: SeriesResult,
    gt_start_nm: float,
    gt_end_nm: float,
    smoothing_divisor: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray, Mapping]:
    series = result.series
    time_ms = result.time_ms
    window = max(21, int(len(series) / smoothing_divisor) | 1)
    smooth = moving_average(series, window)
    region = detect_active_region(smooth)
    recon_norm = normalise_curve(series, region)

    recon_start_ms = float(time_ms[region.start_idx])
    recon_end_ms = float(time_ms[region.end_idx])
    if recon_end_ms <= recon_start_ms:
        raise ValueError(f"Non-increasing active region for {result.label}")
    slope = (gt_end_nm - gt_start_nm) / (recon_end_ms - recon_start_ms)
    intercept = gt_start_nm - slope * recon_start_ms
    wavelengths = slope * time_ms + intercept

    mapping = Mapping(
        name=result.label,
        neg_scale=result.neg_scale,
        window_ms=result.window_ms,
        stride_ms=result.stride_ms,
        recon_start_ms=recon_start_ms,
        recon_end_ms=recon_end_ms,
        gt_start_nm=gt_start_nm,
        gt_end_nm=gt_end_nm,
        slope=slope,
        intercept=intercept,
    )
    return wavelengths, recon_norm, mapping


def ensure_output_dir(root: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"analysis_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    default_segment = (
        REPO_ROOT
        / "scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184724/"
        "angle_20_blank_2835_event_20250925_184724_segments/Scan_1_Forward_events.npz"
    )
    parser = argparse.ArgumentParser(description="Compare multiple reconstruction series with ground truth")
    parser.add_argument("--segment", type=Path, default=default_segment, help="Segmented NPZ to analyse")
    parser.add_argument("--gt_dir", type=Path, default=REPO_ROOT / "groundtruth_spectrum_2835", help="Directory of ground-truth .txt files")
    parser.add_argument("--step_ms", type=float, default=2.0, help="Cumulative step size in milliseconds")
    parser.add_argument("--fixed_windows", type=float, nargs="*", default=[50.0, 100.0], help="Fixed sliding window widths (ms)")
    parser.add_argument("--dynamic_windows", type=float, nargs="*", default=[100.0, 500.0], help="Dynamic window widths (ms)")
    parser.add_argument("--stride_ms", type=float, default=2.0, help="Stride for fixed windows (ms)")
    parser.add_argument("--sensor_width", type=int, default=1280)
    parser.add_argument("--sensor_height", type=int, default=720)
    parser.add_argument("--pos_scale", type=float, default=1.0)
    parser.add_argument("--auto_neg_min", type=float, default=0.1)
    parser.add_argument("--auto_neg_max", type=float, default=3.0)
    parser.add_argument("--plateau_frac", type=float, default=0.05)
    parser.add_argument("--grating_spd", action="store_true", help="Overlay light SPD × grating efficiency (600 grooves/mm) on each subplot")
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "groundtruth_spectrum_2835")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    curves = load_ground_truth_curves(args.gt_dir)
    gt_start_nm, gt_end_nm, per_curve = detect_visible_region(curves)

    series_results: List[SeriesResult] = []

    cumulative = build_cumulative_series(
        args.segment,
        step_ms=args.step_ms,
        pos_scale=args.pos_scale,
        auto_bounds=(args.auto_neg_min, args.auto_neg_max),
        plateau_frac=args.plateau_frac,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
    )
    series_results.append(cumulative)

    for window_ms in args.fixed_windows:
        time_ms, series, neg_scale = build_fixed_window_series(
            args.segment,
            window_ms,
            args.stride_ms,
            args.pos_scale,
            (args.auto_neg_min, args.auto_neg_max),
            args.plateau_frac,
            args.sensor_width,
            args.sensor_height,
        )
        label = f"Fixed {window_ms:.0f}ms (stride {args.stride_ms:.1f}ms)"
        series_results.append(
            prepare_series(label, time_ms, series, neg_scale, window_ms=window_ms, stride_ms=args.stride_ms)
        )

    for window_ms in args.dynamic_windows:
        time_ms, series, neg_scale = build_dynamic_series(
            args.segment,
            window_ms,
            args.pos_scale,
            (args.auto_neg_min, args.auto_neg_max),
            args.plateau_frac,
            args.sensor_width,
            args.sensor_height,
        )
        label = f"Dynamic {window_ms:.0f}ms"
        series_results.append(prepare_series(label, time_ms, series, neg_scale, window_ms=window_ms, stride_ms=None))


    grating_curve = None
    if args.grating_spd:
        light_wl, light_intensity = load_light_spd(LIGHT_SPD_FILE)
        grating_curves = load_grating_efficiencies(GRATING_FILE)
        if GROOVE_DEFAULT not in grating_curves:
            raise ValueError(f'Groove density {GROOVE_DEFAULT} missing in grating file')
        g_wl, g_eff = grating_curves[GROOVE_DEFAULT]
        wl_common, light_interp, grating_interp, product = build_grating_product(light_wl, light_intensity, g_wl, g_eff)
        grating_curve = (wl_common, product)
    output_dir = ensure_output_dir(args.output_root)
    plot_path = output_dir / "reconstruction_vs_groundtruth.png"
    summary_path = output_dir / "alignment_summary.txt"

    gt_plots: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, wl, val in curves:
        mask = (wl >= 300.0) & (wl <= 900.0)
        wl_sel = wl[mask]
        val_sel = val[mask]
        smooth = moving_average(val_sel, max(21, len(val) // 300))
        region = detect_active_region(smooth)
        norm_val = normalise_curve(smooth, region)
        gt_plots.append((name, wl_sel, norm_val))

    mappings: List[Mapping] = []
    num_results = len(series_results)
    cols = 2 if num_results > 1 else 1
    rows = int(np.ceil(num_results / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, result in enumerate(series_results):
        ax = axes_arr[idx]
        ax.axvspan(LIGHT_VISIBLE_MIN, LIGHT_VISIBLE_MAX, color="0.9", alpha=0.5)
        for name, wl_sel, norm_val in gt_plots:
            ax.plot(wl_sel, norm_val, linewidth=1.0, label=f"GT {name}")

        if grating_curve is not None:
            g_wl, g_product = grating_curve
            ax.plot(g_wl, g_product / np.max(g_product), color='green', linestyle='--', linewidth=1.0, label='Light×Grating SPD')

        wl, norm_series, mapping = align_series(result, gt_start_nm, gt_end_nm)
        ax.plot(wl, norm_series, linewidth=1.1, label=result.label)
        ax.axvline(gt_start_nm, color="k", linestyle="--", linewidth=0.9)
        ax.axvline(gt_end_nm, color="k", linestyle="-.", linewidth=0.9)
        ax.set_xlim(300.0, 900.0)
        ax.set_ylim(-0.1, 1.2)
        ax.set_title(result.label)
        ax.grid(alpha=0.3)
        if idx % cols == 0:
            ax.set_ylabel("Normalised intensity (a.u.)")
        if idx // cols == rows - 1:
            ax.set_xlabel("Wavelength (nm)")
        ax.legend(fontsize=7, loc="upper right")
        mappings.append(mapping)

    for ax in axes_arr[num_results:]:
        ax.axis('off')

    fig.suptitle("Reconstruction vs. spectrometer (auto-aligned)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path, dpi=200)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Reconstruction vs. ground-truth alignment summary\n")
        f.write(f"Segment: {args.segment}\n")
        f.write(f"Ground-truth directory: {args.gt_dir}\n")
        f.write(f"Detected blue edge: {gt_start_nm:.3f} nm\n")
        f.write(f"Detected red edge: {gt_end_nm:.3f} nm\n")
        f.write("\nPer-curve mappings (λ = slope * t_ms + intercept):\n")
        for mapping in mappings:
            f.write(f"- {mapping.name}:\n")
            f.write(f"    Neg scale: {mapping.neg_scale:.6f}\n")
            if mapping.window_ms is not None:
                f.write(f"    Window: {mapping.window_ms:.3f} ms\n")
            if mapping.stride_ms is not None:
                f.write(f"    Stride: {mapping.stride_ms:.3f} ms\n")
            f.write(f"    Recon start/end (ms): {mapping.recon_start_ms:.3f} / {mapping.recon_end_ms:.3f}\n")
            f.write(f"    Mapping: λ(t_ms) = {mapping.slope:.6f} * t_ms + {mapping.intercept:.3f}\n")
            f.write("\n")
        f.write("Per-ground-truth detected edges:\n")
        for name, start_nm, end_nm in per_curve:
            f.write(f"  {name}: {start_nm:.3f} nm → {end_nm:.3f} nm\n")

    print(f"Saved overlay plot → {plot_path}")
    print(f"Saved alignment summary → {summary_path}")

    if args.show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Unable to display figure: {exc}")


if __name__ == "__main__":
    main()
