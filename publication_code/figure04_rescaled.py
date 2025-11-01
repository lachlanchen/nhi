#!/usr/bin/env python3
"""
Figure 4 (rescaled): Exponential, polarity-weighted accumulations with background subtraction.

For each 50 ms bin (default) the script renders a two-column comparison:
    - Left: raw event counts (simply the number of events in the bin).
    - Right: FAST-compensated timestamps (if learned params exist) with
      polarity weighting, linear accumulation, optional spatio-temporal
      smoothing, and per-frame mean subtraction (background removal).

The polarity weighting still mirrors `visualize_cumulative_weighted.py
--exp --auto_scale`, but it is applied only to the compensated branch.

Outputs default to timestamped folders to avoid overwriting previous runs,
e.g. `figures/figure04_rescaled_[smooth_]YYYYMMDD_HHMMSS/figure04_rescaled_bin_XX.*`.

Usage example:
    python publication_code/figure04_rescaled.py \
        --segment scan_angle_20_led_2835b/.../Scan_1_Forward_events.npz \
        --save-png
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def load_segment_events(segment_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(segment_path)
    x = data["x"].astype(np.int16)
    y = data["y"].astype(np.int16)
    t = data["t"].astype(np.float32)
    p = data["p"].astype(np.float32)
    if p.min() >= 0.0 and p.max() <= 1.0:
        p = (p - 0.5) * 2.0
    return x, y, t, p


def find_param_file(segment_path: Path) -> Path | None:
    base = segment_path.stem
    seg_dir = segment_path.parent
    patterns = [
        f"{base}_chunked_processing_learned_params_*.npz",
        f"{base}_learned_params_*.npz",
        f"{base}*learned_params*.npz",
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(str(seg_dir / pat)))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return Path(matches[0])


def load_params(param_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(param_path)
    return {
        "a_params": data["a_params"].astype(np.float32),
        "b_params": data["b_params"].astype(np.float32),
    }


def compute_fast_comp_times(x: np.ndarray, y: np.ndarray, t: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, float]:
    a_avg = float(np.mean(params["a_params"]))
    b_avg = float(np.mean(params["b_params"]))
    t_comp = t - (a_avg * x + b_avg * y)
    return t_comp, a_avg, b_avg


def bin_indices(t: np.ndarray, t_min: float, bin_width_us: float, num_bins: int) -> np.ndarray:
    idx = ((t - t_min) // bin_width_us).astype(np.int64)
    idx = np.clip(idx, 0, num_bins)
    return idx


def auto_scale_neg_weight(
    t: np.ndarray,
    p: np.ndarray,
    sensor_area: float,
    step_us: float,
    pos_scale: float = 1.0,
    neg_scale_init: float = 1.5,
    plateau_frac: float = 0.05,
    auto_min: float = 0.1,
    auto_max: float = 3.0,
) -> float:
    """Reproduce --exp auto-scaling logic from visualize_cumulative_weighted.py."""

    def base_binned_sums(times: np.ndarray, weights: np.ndarray, t_min: float, t_max: float) -> Tuple[np.ndarray, int]:
        K = int((t_max - t_min) // step_us)
        if K <= 0:
            return np.zeros(0, dtype=np.float32), 0
        in_range = (times >= t_min) & (times < t_min + K * step_us)
        if not np.any(in_range):
            return np.zeros(K, dtype=np.float32), K
        idx = ((times[in_range] - t_min) // step_us).astype(np.int64)
        sums = np.bincount(idx, weights=weights[in_range], minlength=K).astype(np.float32)
        return sums, K

    t_min, t_max = float(np.min(t)), float(np.max(t))
    pos_mask = p >= 0
    neg_mask = ~pos_mask
    ones_pos = np.ones(np.count_nonzero(pos_mask), dtype=np.float32)
    ones_neg = np.ones(np.count_nonzero(neg_mask), dtype=np.float32)
    sums_pos, K = base_binned_sums(t[pos_mask], ones_pos, t_min, t_max)
    sums_neg, _ = base_binned_sums(t[neg_mask], ones_neg, t_min, t_max)

    if K <= 0:
        return neg_scale_init

    def build_series(neg_scale: float) -> np.ndarray:
        step_sums = pos_scale * sums_pos - neg_scale * sums_neg
        cm = np.cumsum(step_sums) / sensor_area
        return np.exp(cm)

    def plateau_diff(neg_scale: float) -> float:
        series = build_series(neg_scale)
        n = max(5, int(plateau_frac * len(series)))
        start_mean = float(np.mean(series[:n]))
        end_mean = float(np.mean(series[-n:]))
        return end_mean - start_mean

    a = max(auto_min, 1e-6)
    b = max(auto_max, a + 1e-6)
    fa = plateau_diff(a)
    fb = plateau_diff(b)

    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb < 0:
        for _ in range(40):
            m = 0.5 * (a + b)
            fm = plateau_diff(m)
            if abs(fm) < 1e-6:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    grid = np.linspace(a, b, 50)
    diffs = np.array([abs(plateau_diff(g)) for g in grid])
    return float(grid[int(np.argmin(diffs))])


def accumulate_bin(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    sensor_shape: Tuple[int, int],
) -> np.ndarray:
    frame = np.zeros(sensor_shape, dtype=np.float32)
    if not np.any(mask):
        return frame
    np.add.at(frame, (y[mask], x[mask]), weights[mask])
    return frame


def subtract_background(frame: np.ndarray) -> np.ndarray:
    return frame - float(np.mean(frame))


def smooth_volume_3d(volume: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a simple mean filter over a 3x3x3 neighbourhood with reflect padding."""
    assert kernel_size == 3, "Only kernel size 3 is supported."
    pad = kernel_size // 2
    padded = np.pad(volume, ((pad, pad), (pad, pad), (pad, pad)), mode="reflect")
    smoothed = np.zeros_like(volume, dtype=np.float32)
    for dz in range(kernel_size):
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                smoothed += padded[
                    dz : dz + volume.shape[0],
                    dy : dy + volume.shape[1],
                    dx : dx + volume.shape[2],
                ]
    smoothed /= float(kernel_size ** 3)
    return smoothed


def render_panel(
    original: np.ndarray,
    compensated: np.ndarray,
    raw_vmin: float,
    raw_vmax: float,
    comp_vmin: float,
    comp_vmax: float,
    bin_idx: int,
    bin_width_ms: float,
    colormap: str,
    output_dir: Path,
    save_png: bool,
    diverging: bool,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.2), constrained_layout=True)
    meta = f"Bin {bin_idx} ({bin_width_ms:.0f} ms)"

    comp_abs = max(abs(comp_vmin), abs(comp_vmax))
    if diverging:
        raw_norm = None
        comp_norm = TwoSlopeNorm(vmin=-comp_abs, vcenter=0.0, vmax=comp_abs if comp_abs > 0 else 1.0)
    else:
        raw_norm = None
        comp_norm = None

    im0 = axes[0].imshow(
        original,
        cmap=colormap,
        norm=raw_norm,
        vmin=None if raw_norm else raw_vmin,
        vmax=None if raw_norm else raw_vmax,
        origin="lower",
        interpolation="nearest",
    )
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[0].text(
        0.02,
        0.02,
        meta,
        transform=axes[0].transAxes,
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
    )

    im1 = axes[1].imshow(
        compensated,
        cmap=colormap,
        norm=comp_norm,
        vmin=None if comp_norm else comp_vmin,
        vmax=None if comp_norm else comp_vmax,
        origin="lower",
        interpolation="nearest",
    )
    axes[1].set_title("Compensated")
    axes[1].axis("off")

    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    cbar0.ax.set_ylabel("Raw counts", rotation=90)
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    cbar1.ax.set_ylabel("Compensated (a.u.)", rotation=90)
    fig.text(0.012, 0.97, "(a)", fontweight="bold", ha="left", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"figure04_rescaled_bin_{bin_idx:02d}"
    fig.savefig(f"{stem}.pdf", dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescaled exponential accumulations for Figure 4.")
    parser.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_events.npz file.")
    parser.add_argument("--bin-width-us", type=float, default=50000.0, help="Temporal bin width in microseconds (default 50 ms).")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"), help="Global percentile limits for colour scaling.")
    parser.add_argument("--colormap", default="magma", help="Matplotlib colormap name.")
    parser.add_argument("--save-png", action="store_true", help="Emit PNG alongside PDF.")
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--pos-scale", type=float, default=1.0, help="Positive event weight (default 1.0).")
    parser.add_argument("--neg-scale", type=float, default=1.5, help="Initial negative event weight before auto-scaling.")
    parser.add_argument(
        "--diverging",
        action="store_true",
        help="Use a diverging colormap (pos/neg highlighted) with transparency near zero.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to figures/figure04_rescaled_<timestamp>/",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply a 3x3x3 spatio-temporal mean filter to the compensated accumulation before background subtraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment_path = args.segment.resolve()
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    figures_root = Path(__file__).resolve().parent / "figures"
    if args.output_dir is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "figure04_rescaled"
        if args.smooth:
            base += "_smooth"
        if args.diverging:
            base += "_diverging"
        args.output_dir = figures_root / f"{base}_{suffix}"
    else:
        args.output_dir = args.output_dir.resolve()

    x, y, t, p = load_segment_events(segment_path)
    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)

    neg_scale = auto_scale_neg_weight(
        t,
        p,
        sensor_area=sensor_area,
        step_us=args.bin_width_us,
        pos_scale=args.pos_scale,
        neg_scale_init=args.neg_scale,
    )
    print(f"Rescaled weights: pos_scale={args.pos_scale:.3f}, neg_scale={neg_scale:.3f}")
    comp_weights = np.where(p >= 0, args.pos_scale, -neg_scale).astype(np.float32)
    raw_weights = np.ones_like(p, dtype=np.float32)

    params_file = find_param_file(segment_path)
    t_comp = None
    if params_file is not None:
        params = load_params(params_file)
        t_comp, _, _ = compute_fast_comp_times(x.astype(np.float32), y.astype(np.float32), t, params)
    else:
        print("Warning: Learned parameters not found; compensated column will mirror original.")
        t_comp = t

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / args.bin_width_us))

    originals: List[np.ndarray] = []
    comp_raw_frames: List[np.ndarray] = []

    for idx in range(num_bins):
        start = t_min + idx * args.bin_width_us
        end = start + args.bin_width_us
        mask_orig = (t >= start) & (t < end)
        mask_comp = (t_comp >= start) & (t_comp < end)

        orig_frame = accumulate_bin(x, y, mask_orig, raw_weights, sensor_shape)
        comp_frame = accumulate_bin(x, y, mask_comp, comp_weights, sensor_shape)

        originals.append(orig_frame)
        comp_raw_frames.append(comp_frame.astype(np.float32))

    comp_array = np.stack(comp_raw_frames, axis=0)
    if args.smooth:
        comp_array = smooth_volume_3d(comp_array)

    compensations = [subtract_background(frame) for frame in comp_array]

    out_dir = args.output_dir.resolve()
    for idx, (orig_frame, comp_frame) in enumerate(zip(originals, compensations)):
        raw_vmin = float(orig_frame.min())
        raw_vmax = float(orig_frame.max())
        comp_vmin = float(comp_frame.min())
        comp_vmax = float(comp_frame.max())
        if np.isclose(raw_vmax, raw_vmin):
            raw_vmax = raw_vmin + 1e-3
        if np.isclose(comp_vmax, comp_vmin):
            comp_vmax = comp_vmin + 1e-3

        render_panel(
            orig_frame,
            comp_frame,
            raw_vmin,
            raw_vmax,
            comp_vmin,
            comp_vmax,
            idx,
            args.bin_width_us / 1000.0,
            args.colormap,
            out_dir,
            args.save_png,
            args.diverging,
        )


if __name__ == "__main__":
    main()
