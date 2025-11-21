#!/usr/bin/env python3
"""Compare compensated event-camera spectrum against ground-truth spectrometer curves.

Steps:
1. Reproduce the compensated, auto-scaled exponential cumulative spectrum used by
   `visualize_cumulative_weighted.py` (step size 10 µs, pos_scale 1.0, auto neg scale).
2. Detect the active (non-flat) region on both the reconstruction and the ground-truth
   curves by measuring departures from their start/end plateaus.
3. Derive a linear wavelength<->time map by aligning the detected start/end markers.
4. Plot both ground-truth curves and the mapped reconstruction in the wavelength domain,
   highlighting the inferred visible region.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

import visualize_cumulative_weighted as vcw

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@dataclass
class ActiveRegion:
    start_idx: int
    end_idx: int
    start_value: float
    end_value: float
    baseline_start: float
    baseline_end: float


def load_ground_truth(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse .txt export from OceanView spectrometer (tab-delimited).

    Returns:
        wavelengths_nm, irradiance counts
    """
    data: list[Tuple[float, float]] = []
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if ">>>>>Begin Spectral Data<<<<<" in line:
                break
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\t, ]+", line)
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0])
                val = float(parts[1])
            except ValueError:
                continue
            data.append((wl, val))
    arr = np.asarray(data, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_active_region(curve: np.ndarray, frac: float = 0.05, threshold_ratio: float = 0.05) -> ActiveRegion:
    """Find indices where the curve leaves/returns to its plateaus.

    Args:
        curve: 1-D signal already smoothed.
        frac: fraction of samples at each end considered "plateau".
        threshold_ratio: fraction of dynamic range used as detection threshold.
    """
    n = len(curve)
    plateau_n = max(10, int(frac * n))
    start_plateau = float(np.mean(curve[:plateau_n]))
    end_plateau = float(np.mean(curve[-plateau_n:]))
    dynamic = float(np.max(curve) - min(start_plateau, end_plateau))
    threshold = start_plateau + threshold_ratio * dynamic
    rev_threshold = end_plateau + threshold_ratio * dynamic

    start_idx = next((i for i, v in enumerate(curve) if v >= threshold), 0)
    end_idx = n - 1 - next((i for i, v in enumerate(curve[::-1]) if v >= rev_threshold), 0)

    return ActiveRegion(
        start_idx=start_idx,
        end_idx=end_idx,
        start_value=float(curve[start_idx]),
        end_value=float(curve[end_idx]),
        baseline_start=start_plateau,
        baseline_end=end_plateau,
    )


def normalise_curve(curve: np.ndarray, region: ActiveRegion) -> np.ndarray:
    baseline = region.baseline_start
    shifted = curve - baseline
    peak_slice = shifted[region.start_idx : region.end_idx + 1]
    peak = np.max(peak_slice)
    if peak <= 0:
        peak = np.max(np.abs(shifted)) or 1.0
    normalised = shifted / peak
    return normalised


# ---------------------------------------------------------------------------
# Reconstruction processing
# ---------------------------------------------------------------------------

def build_compensated_series(
    segment_npz: Path,
    step_us: float = 10.0,
    pos_scale: float = 1.0,
    auto_scale_bounds: Tuple[float, float] = (0.1, 3.0),
    plateau_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Recreate the compensated exponential cumulative signal."""
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    t_min, t_max = float(np.min(t)), float(np.max(t))
    hw = float(1280 * 720)

    pos_mask = p >= 0
    neg_mask = ~pos_mask

    sums_pos_orig, edges_ms = vcw.base_binned_sums_weighted(
        t[pos_mask],
        np.ones(np.count_nonzero(pos_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )
    sums_neg_orig, _ = vcw.base_binned_sums_weighted(
        t[neg_mask],
        np.ones(np.count_nonzero(neg_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )

    def build_cum(s_pos: np.ndarray, s_neg: np.ndarray, neg_scale: float) -> np.ndarray:
        return np.cumsum(pos_scale * s_pos - neg_scale * s_neg) / hw

    def plateau_diff(neg_scale: float) -> float:
        series = np.exp(build_cum(sums_pos_orig, sums_neg_orig, neg_scale))
        k = len(series)
        if k <= 2:
            return 0.0
        n = max(5, int(plateau_frac * k))
        return float(np.mean(series[-n:]) - np.mean(series[:n]))

    # Neg polarity scale via bisection/grid search (mirrors CLI tool).
    low, high = auto_scale_bounds
    f_low, f_high = plateau_diff(low), plateau_diff(high)
    if f_low * f_high < 0:
        for _ in range(40):
            mid = 0.5 * (low + high)
            f_mid = plateau_diff(mid)
            if abs(f_mid) < 1e-6:
                low = high = mid
                break
            if f_low * f_mid < 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid
        chosen_neg = 0.5 * (low + high)
    else:
        grid = np.linspace(low, high, 50)
        vals = np.array([abs(plateau_diff(g)) for g in grid])
        chosen_neg = float(grid[int(np.argmin(vals))])

    params_file = vcw.find_param_file_for_segment(str(segment_npz))
    if params_file is None:
        raise FileNotFoundError("No learned parameter NPZ found next to segment")
    params = vcw.load_parameters_from_npz(params_file)
    t_comp, *_ = vcw.compute_fast_compensated_times(x, y, t, params["a_params"], params["b_params"])

    sums_pos_comp, _ = vcw.base_binned_sums_weighted(
        t_comp[pos_mask],
        np.ones(np.count_nonzero(pos_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )
    sums_neg_comp, _ = vcw.base_binned_sums_weighted(
        t_comp[neg_mask],
        np.ones(np.count_nonzero(neg_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )

    comp_series = np.exp(build_cum(sums_pos_comp, sums_neg_comp, chosen_neg))
    rel_time_ms = edges_ms - edges_ms[0]
    return rel_time_ms, comp_series, chosen_neg


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parent.parent
    segment = repo_root / "scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz"

    time_ms, reconstruction, neg_scale = build_compensated_series(segment)

    # Smooth reconstruction for robust edge detection (retain full-resolution for plotting).
    smooth_recon = moving_average(reconstruction, max(51, len(reconstruction) // 2000))
    recon_region = detect_active_region(smooth_recon)
    recon_norm = normalise_curve(reconstruction, recon_region)

    recon_start_ms = float(time_ms[recon_region.start_idx])
    recon_end_ms = float(time_ms[recon_region.end_idx])

    gt_paths = [
        repo_root / "groundtruth_spectrum/USB2F042671_16-04-36-993.txt",
        repo_root / "groundtruth_spectrum/USB2F042671_16-04-56-391.txt",
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

    # Convert reconstruction timeline to wavelength domain.
    recon_wavelengths = gt_start_nm + (time_ms - recon_start_ms) * recon_scale

    gt_norm_curves = []
    for (wl, val), region in zip(gt_curves, gt_regions):
        gt_norm_curves.append((wl, normalise_curve(val, region)))

    # Plot full range but highlight 380-780 nm visible band.
    out_dir = repo_root / "groundtruth_spectrum"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "reconstruction_vs_groundtruth.png"

    plt.figure(figsize=(11, 6))
    plt.axvspan(380, 780, color="0.85", alpha=0.5, label="Visible band (380-780 nm)")

    for (wl, curve), path in zip(gt_norm_curves, gt_paths):
        plt.plot(wl, curve, label=f"Ground truth ({path.name})", linewidth=1.2)

    # Downsample reconstruction for plotting clarity without changing data range.
    stride = max(1, len(recon_wavelengths) // 5000)
    plt.plot(
        recon_wavelengths[::stride],
        recon_norm[::stride],
        color="crimson",
        linewidth=1.0,
        label="Compensated reconstruction",
    )

    plt.axvline(gt_start_nm, color="k", linestyle="--", linewidth=1.0, label="Aligned blue edge")
    plt.axvline(gt_end_nm, color="k", linestyle="-.", linewidth=1.0, label="Aligned red edge")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalised intensity (a.u.)")
    plt.title("Auto-aligned spectrum: event reconstruction vs. spectrometer")
    plt.xlim(min(recon_wavelengths.min(), gt_norm_curves[0][0][0]), max(recon_wavelengths.max(), gt_norm_curves[0][0][-1]))
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)

    mapping_report = out_dir / "reconstruction_time_to_wavelength.txt"
    with mapping_report.open("w", encoding="utf-8") as f:
        f.write("Compensated reconstruction vs. ground-truth alignment\n")
        f.write(f"Neg polarity scale (auto): {neg_scale:.6f}\n")
        f.write(f"Reconstruction start (ms): {recon_start_ms:.3f}\n")
        f.write(f"Reconstruction end (ms): {recon_end_ms:.3f}\n")
        f.write(f"Ground-truth start (nm): {gt_start_nm:.3f}\n")
        f.write(f"Ground-truth end (nm): {gt_end_nm:.3f}\n")
        slope = recon_scale
        intercept = gt_start_nm - slope * recon_start_ms
        f.write("Mapping formula (ms → nm):\n")
        f.write(f"  λ(t_ms) = {slope:.6f} * t_ms + {intercept:.3f}\n")
        f.write("Where t_ms is compensated time measured relative to the earliest bin.\n")

    print(f"Saved overlay plot → {out_path}")
    print(f"Saved mapping details → {mapping_report}")


if __name__ == "__main__":
    main()
