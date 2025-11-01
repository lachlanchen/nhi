#!/usr/bin/env python3
"""
Figure 4 (grid): side-by-side comparison using the rescaled accumulation logic.

This script reuses the Figure 4 rescaling pipeline but renders a compact two-row
grid: the first row shows raw accumulations for bins 3–15 (inclusive), the
second row shows the corresponding compensated frames. All other processing
(polarity weights, optional smoothing, colour theme, JSON metadata) matches the
behaviour of `figure04_rescaled.py`. Colourbars are hidden by default; pass
`--show-colorbar` to enable them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, TwoSlopeNorm

from figure04_rescaled import (
    DEFAULT_SHARED_COLORMAP,
    COMP_LIGHTEN_FRACTION,
    RAW_LIGHTEN_FRACTION,
    RESCALE_FINE_STEP_US,
    accumulate_bin,
    auto_scale_neg_weight,
    compute_fast_comp_times,
    find_param_file,
    load_params,
    load_segment_events,
    prepare_colormap,
    save_rescale_series,
    setup_style,
    smooth_volume_3d,
    subtract_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescaled accumulation grid for Figure 4.")
    parser.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_events.npz file.")
    parser.add_argument("--bin-width-us", type=float, default=50000.0, help="Temporal bin width in microseconds (default 50 ms).")
    parser.add_argument("--sensor-width", type=int, default=1280)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--pos-scale", type=float, default=1.0, help="Positive event weight (default 1.0).")
    parser.add_argument("--neg-scale", type=float, default=1.5, help="Initial negative event weight before auto-scaling.")
    parser.add_argument("--colormap", default=None, help="Shared base colormap name applied to both rows (default coolwarm).")
    parser.add_argument("--raw-colormap", default=None, help="Override colormap for raw (first-row) panels.")
    parser.add_argument("--comp-colormap", default=None, help="Override colormap for compensated (second-row) panels.")
    parser.add_argument("--save-png", action="store_true", help="Emit PNG alongside PDF.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory; defaults to figures/figure04_rescaled_grid_<timestamp>/")
    parser.add_argument("--smooth", action="store_true", help="Apply a 3x3x3 spatio-temporal mean filter before background subtraction.")
    parser.add_argument("--start-bin", type=int, default=3, help="Inclusive start bin index for the grid (default 3).")
    parser.add_argument("--end-bin", type=int, default=15, help="Inclusive end bin index for the grid (default 15).")
    parser.add_argument("--show-colorbar", action="store_true", help="Display colourbars beneath each column.")
    return parser.parse_args()


def compute_bins(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    t_comp: np.ndarray,
    bin_width_us: float,
    raw_weights: np.ndarray,
    comp_weights: np.ndarray,
    sensor_shape: Tuple[int, int],
    smooth: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, float]], np.ndarray]:
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_comp_max = float(np.max(t_comp)) if t_comp.size else t_max
    num_bins = int(np.ceil((t_max - t_min) / bin_width_us))

    originals: List[np.ndarray] = []
    comp_frames: List[np.ndarray] = []
    metadata: List[Dict[str, float]] = []

    for idx in range(num_bins):
        start = t_min + idx * bin_width_us
        end = start + bin_width_us
        mask_orig = (t >= start) & (t < end)
        mask_comp = (t_comp >= start) & (t_comp < end)
        if idx == num_bins - 1:
            mask_orig |= t == t_max
            mask_comp |= t_comp == t_comp_max

        orig = accumulate_bin(x, y, mask_orig, raw_weights, sensor_shape)
        comp = accumulate_bin(x, y, mask_comp, comp_weights, sensor_shape)

        originals.append(orig)
        comp_frames.append(comp.astype(np.float32))
        metadata.append(
            {
                "index": idx,
                "start_us": float(start),
                "end_us": float(min(end, t_max)),
                "duration_ms": float((min(end, t_max) - start) / 1000.0),
            }
        )

    comp_volume = np.stack(comp_frames, axis=0)
    if smooth:
        comp_volume = smooth_volume_3d(comp_volume)

    bg_values = comp_volume.mean(axis=(1, 2))
    compensated = [subtract_background(frame) for frame in comp_volume]
    return originals, compensated, metadata, bg_values


def render_grid(
    originals: List[np.ndarray],
    compensated: List[np.ndarray],
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    raw_cmap: Colormap,
    comp_cmap: Colormap,
    show_colorbar: bool,
    output_dir: Path,
    save_png: bool,
    bin_width_ms: float,
) -> None:
    setup_style()

    selected = [
        (orig, comp, meta)
        for orig, comp, meta in zip(originals, compensated, metadata)
        if start_bin <= meta["index"] <= end_bin
    ]
    if not selected:
        raise ValueError(f"No bins found in the range [{start_bin}, {end_bin}].")

    num_cols = len(selected)
    fig = plt.figure(figsize=(1.2 * num_cols + 0.4, 1.6))
    width_ratios = [0.22] + [1.0] * num_cols
    gs = fig.add_gridspec(2, num_cols + 1, wspace=0.045, hspace=0.015, width_ratios=width_ratios)
    axes = np.empty((2, num_cols), dtype=object)
    for row in range(2):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            "Orig." if row == 0 else "Comp.",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.4, edgecolor="none"),
        )
        for col in range(num_cols):
            axes[row, col] = fig.add_subplot(gs[row, col + 1])
    fig.subplots_adjust(left=0.02, right=0.995, top=0.995, bottom=0.04)

    for col, (orig_frame, comp_frame, meta) in enumerate(selected):
        raw_vmin = min(0.0, float(orig_frame.min()))
        raw_vmax = max(0.0, float(orig_frame.max()))
        comp_vmin = min(float(comp_frame.min()), 0.0)
        comp_vmax = max(float(comp_frame.max()), 0.0)
        if np.isclose(raw_vmax, raw_vmin):
            raw_vmax = raw_vmin + 1e-3
        if np.isclose(comp_vmax, comp_vmin):
            comp_vmax = comp_vmin + 1e-3

        comp_norm: TwoSlopeNorm | None = None
        if comp_vmin < 0.0 and comp_vmax > 0.0:
            comp_abs = max(abs(comp_vmin), abs(comp_vmax))
            if comp_abs <= 0.0:
                comp_abs = 1.0
            comp_norm = TwoSlopeNorm(vmin=-comp_abs, vcenter=0.0, vmax=comp_abs)

        ax_orig = axes[0, col]
        im0 = ax_orig.imshow(
            orig_frame,
            cmap=raw_cmap,
            vmin=raw_vmin,
            vmax=raw_vmax,
            origin="lower",
            interpolation="nearest",
        )
        ax_orig.axis("off")

        ax_comp = axes[1, col]
        im1 = ax_comp.imshow(
            comp_frame,
            cmap=comp_cmap,
            vmin=None if comp_norm else comp_vmin,
            vmax=None if comp_norm else comp_vmax,
            norm=comp_norm,
            origin="lower",
            interpolation="nearest",
        )
        ax_comp.axis("off")

        if show_colorbar:
            cbar_raw = fig.colorbar(im0, ax=ax_orig, shrink=0.85, pad=0.01)
            cbar_raw.ax.set_ylabel("Raw counts", rotation=90)
            cbar_comp = fig.colorbar(im1, ax=ax_comp, shrink=0.85, pad=0.01)
            label = "Compensated Δ (a.u.)" if comp_norm is not None else "Compensated (a.u.)"
            cbar_comp.ax.set_ylabel(label, rotation=90)

    out_stem = output_dir / f"figure04_rescaled_grid_bins_{start_bin:02d}_{end_bin:02d}"
    fig.savefig(f"{out_stem}.pdf", dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem}.pdf")


def plot_background_spectrum(
    rescale_series: Dict[str, np.ndarray],
    bg_values: np.ndarray,
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    output_dir: Path,
    save_png: bool,
) -> None:
    time_ms = rescale_series.get("time_ms")
    if time_ms is None or time_ms.size == 0:
        return

    exp_unscaled = rescale_series.get("exp_unscaled")
    exp_rescaled = rescale_series.get("exp_rescaled")

    indices = [meta["index"] for meta in metadata]
    if not indices:
        return

    base_start = metadata[0]["start_us"]
    centres_ms = [((meta["start_us"] - base_start) / 1000.0) + (meta["duration_ms"] / 2.0) for meta in metadata]
    mask = [start_bin <= idx <= end_bin for idx in indices]
    selected = [meta for meta in metadata if start_bin <= meta["index"] <= end_bin]
    highlight_start = highlight_end = None
    if selected:
        highlight_start = (selected[0]["start_us"] - base_start) / 1000.0
        highlight_end = (selected[-1]["end_us"] - base_start) / 1000.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=False, constrained_layout=True)

    ax0.plot(time_ms, exp_unscaled, color="#7f7f7f", linewidth=1.0, alpha=0.7, label="Original exp")
    ax0.plot(time_ms, exp_rescaled, color="#1f77b4", linewidth=1.4, label="Rescaled exp")
    if highlight_start is not None and highlight_end is not None:
        ax0.axvspan(highlight_start, highlight_end, color="#d62728", alpha=0.08, lw=0)
    ax0.set_ylabel("exp(cumulative)")
    ax0.set_title("Rescaled background spectrum")
    ax0.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(centres_ms, bg_values, color="#1f77b4", linewidth=1.4, marker="o", label="50 ms mean")
    if any(mask):
        highlight_x = [x for x, m in zip(centres_ms, mask) if m]
        highlight_y = [y for y, m in zip(bg_values, mask) if m]
        ax1.scatter(highlight_x, highlight_y, color="#d62728", s=30, zorder=3, label="Selected bins")
    if highlight_start is not None and highlight_end is not None:
        ax1.axvspan(highlight_start, highlight_end, color="#d62728", alpha=0.08, lw=0)
    ax1.set_xlabel("Relative time (ms)")
    ax1.set_ylabel("Background mean (a.u.)")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax1.legend(loc="best", fontsize=8)

    out_path = output_dir / "figure04_rescaled_bg_spectrum.pdf"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    segment_path = args.segment.resolve()
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    figures_root = Path(__file__).resolve().parent / "figures"
    if args.output_dir is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "figure04_rescaled_grid"
        if args.smooth:
            base += "_smooth"
        args.output_dir = figures_root / f"{base}_{suffix}"

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, t, p = load_segment_events(segment_path)
    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)

    neg_scale, rescale_series = auto_scale_neg_weight(
        t,
        p,
        sensor_area=sensor_area,
        step_us=RESCALE_FINE_STEP_US,
        pos_scale=args.pos_scale,
        neg_scale_init=args.neg_scale,
        return_series=True,
    )
    print(f"Rescaled weights: pos_scale={args.pos_scale:.3f}, neg_scale={neg_scale:.3f}")

    comp_weights = np.where(p >= 0, args.pos_scale, -neg_scale).astype(np.float32)
    raw_weights = np.ones_like(p, dtype=np.float32)
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_base = args.raw_colormap or shared_base
    comp_base = args.comp_colormap or shared_base
    raw_cmap = prepare_colormap(raw_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(comp_base, "center", COMP_LIGHTEN_FRACTION)

    params_file = find_param_file(segment_path)
    if params_file is not None:
        params = load_params(params_file)
        t_comp, _, _ = compute_fast_comp_times(x.astype(np.float32), y.astype(np.float32), t, params)
    else:
        print("Warning: Learned parameters not found; compensated row will mirror original.")
        t_comp = t

    originals, compensated, metadata, bg_values = compute_bins(
        x,
        y,
        t,
        p,
        t_comp,
        args.bin_width_us,
        raw_weights,
        comp_weights,
        sensor_shape,
        args.smooth,
    )

    render_grid(
        originals,
        compensated,
        metadata,
        args.start_bin,
        args.end_bin,
        raw_cmap,
        comp_cmap,
        args.show_colorbar,
        out_dir,
        args.save_png,
        args.bin_width_us / 1000.0,
    )

    save_rescale_series(rescale_series, out_dir)

    plot_background_spectrum(
        rescale_series,
        bg_values,
        metadata,
        args.start_bin,
        args.end_bin,
        out_dir,
        args.save_png,
    )

    weights_path = out_dir / "figure04_rescaled_weights.json"
    payload = {
        "segment": str(segment_path),
        "pos_scale": args.pos_scale,
        "neg_scale": neg_scale,
        "bin_width_us": args.bin_width_us,
        "bin_width_ms": args.bin_width_us / 1000.0,
        "start_bin": args.start_bin,
        "end_bin": args.end_bin,
        "smooth": bool(args.smooth),
        "bin_times_us": metadata,
        "background_means": bg_values.tolist(),
        "rescale_step_us": RESCALE_FINE_STEP_US,
        "bg_series_npz": "figure04_rescaled_bg_series.npz",
    }
    with weights_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    main()
