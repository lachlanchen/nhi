#!/usr/bin/env python3
"""
Figure 4 one-shot pipeline:

1) Learn polarity weights on a fine 5 ms cumulative background series
2) Align rescaled background to spectrometer GT and save time↔wavelength map
3) Accumulate 50 ms frames (raw + weighted compensated), optional smoothing, bg subtraction
4) Render 2×N grid with optional wavelength labels under the compensated row

Outputs are written to a timestamped folder under publication_code/figures/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure04_rescaled import (
    DEFAULT_SHARED_COLORMAP,
    RAW_LIGHTEN_FRACTION,
    COMP_LIGHTEN_FRACTION,
    RESCALE_FINE_STEP_US,
    load_segment_events,
    find_param_file,
    load_params,
    compute_fast_comp_times,
    auto_scale_neg_weight,
    prepare_colormap,
    accumulate_bin,
    smooth_volume_3d,
    subtract_background,
    setup_style,
)
from groundtruth_spectrum_2835.compare_publication_cumulative import (
    detect_visible_edges,
    load_gt_curves,
)
from groundtruth_spectrum.compare_reconstruction_to_gt import (
    moving_average,
    normalise_curve,
    detect_active_region,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Figure 4 all-in-one pipeline")
    ap.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_events.npz")
    ap.add_argument("--gt-dir", type=Path, default=Path("groundtruth_spectrum_2835"))
    ap.add_argument("--bin-width-us", type=float, default=50000.0)
    ap.add_argument("--fine-step-us", type=float, default=RESCALE_FINE_STEP_US)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--neg-scale", type=float, default=1.5)
    ap.add_argument("--start-bin", type=int, default=3)
    ap.add_argument("--end-bin", type=int, default=15)
    ap.add_argument("--colormap", default=None)
    ap.add_argument("--raw-colormap", default=None)
    ap.add_argument("--comp-colormap", default=None)
    # Default to 3x3x3 smoothing enabled; user can turn off with --no-smooth if needed
    ap.add_argument("--smooth", action="store_true", default=True)
    ap.add_argument("--show-wavelength", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def compute_series_and_weights(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    a_params: np.ndarray | None,
    b_params: np.ndarray | None,
    sensor_area: float,
    pos_scale: float,
    neg_init: float,
    step_us: float,
) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
    if a_params is not None and b_params is not None:
        a_avg = float(np.mean(a_params))
        b_avg = float(np.mean(b_params))
        t_comp = t - (a_avg * x + b_avg * y)
    else:
        t_comp = t
    neg_scale, series = auto_scale_neg_weight(
        t_comp,
        p,
        sensor_area=sensor_area,
        step_us=step_us,
        pos_scale=pos_scale,
        neg_scale_init=neg_init,
        return_series=True,
    )
    weights = np.where(p >= 0, pos_scale, -neg_scale).astype(np.float32)
    return float(neg_scale), series, weights


def save_series(series: Dict[str, np.ndarray], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "figure04_rescaled_bg_series.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in series.items()})
    return path


def align_with_groundtruth(
    series: Dict[str, np.ndarray],
    gt_dir: Path,
    metadata_bins: List[Dict[str, float]],
    out_dir: Path,
    neg_scale: float,
    step_us: float,
    save_png: bool,
) -> Dict[str, object]:
    time_ms = np.asarray(series["time_ms"], dtype=np.float32)
    exp_rescaled = np.asarray(series["exp_rescaled"], dtype=np.float32)
    gt_curves = load_gt_curves(gt_dir)
    gt_start_nm, gt_end_nm = detect_visible_edges(gt_curves)

    # Edge-of-active-region anchors (entering/leaving flats) for background (time) and GT (wavelength)
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    # Background edges in time: first rise from flat and return to flat
    t0 = float(time_ms[region_bg.start_idx])
    t1 = float(time_ms[region_bg.end_idx])

    # GT edges in wavelength (average across curves)
    wl0_list: list[float] = []
    wl1_list: list[float] = []
    x_min, x_max = float(gt_start_nm), float(gt_end_nm)
    for _, wl, val in gt_curves:
        mask = (wl >= x_min) & (wl <= x_max)
        wl_sel = wl[mask]
        val_sel = val[mask]
        if wl_sel.size < 10:
            continue
        smooth_gt = moving_average(val_sel, max(21, len(val_sel) // 300))
        region_gt = detect_active_region(smooth_gt)
        # Use the edges of the active region on GT
        wl0_list.append(float(wl_sel[region_gt.start_idx]))
        wl1_list.append(float(wl_sel[region_gt.end_idx]))
    if wl0_list and wl1_list:
        wl0 = float(np.mean(wl0_list))
        wl1 = float(np.mean(wl1_list))
    else:
        wl0, wl1 = float(gt_start_nm), float(gt_end_nm)

    # Linear map from time→wavelength using plateau centroids
    if t1 == t0:
        slope, intercept = (wl1 - wl0) / 1.0, wl0 - ((wl1 - wl0) / 1.0) * t0
    else:
        slope = (wl1 - wl0) / (t1 - t0)
        intercept = wl0 - slope * t0
    wl_series = slope * time_ms + intercept
    series_norm = normalise_curve(exp_rescaled, region_bg)

    # bin mapping based on 50 ms centers
    base_start = metadata_bins[0]["start_us"] if metadata_bins else 0.0
    mapping = []
    for item in metadata_bins:
        centre_ms = (((item["start_us"] + item["end_us"]) * 0.5) - base_start) / 1000.0
        mapping.append(
            {
                "index": int(item["index"]),
                "time_center_ms": float(centre_ms),
                "wavelength_nm": float(slope * centre_ms + intercept),
            }
        )

    # Plot background vs GT in wavelength domain (all normalised for fair comparison)
    # Determine overall GT wavelength span for plotting
    wl_min_all = min(float(np.min(wl)) for _, wl, _ in gt_curves)
    wl_max_all = max(float(np.max(wl)) for _, wl, _ in gt_curves)
    mask_bg = (wl_series >= wl_min_all) & (wl_series <= wl_max_all)
    wl_bg = wl_series[mask_bg]
    bg_norm = series_norm[mask_bg]

    # Build mean GT raw curve on a common grid for amplitude matching in plot
    wl_grid_plot = np.linspace(wl_min_all, wl_max_all, 2000, dtype=np.float32)
    gt_grid_stack: list[np.ndarray] = []
    for _, wl, val in gt_curves:
        gt_grid_stack.append(np.interp(wl_grid_plot, wl, val).astype(np.float32))
    gt_mean_raw = np.mean(gt_grid_stack, axis=0)

    # Interpolate BG shape onto the same grid and compute affine scale (a,c) that best fits GT mean
    bg_grid = np.interp(wl_grid_plot, wl_bg, bg_norm) if wl_bg.size else np.zeros_like(wl_grid_plot)
    A = np.stack([bg_grid, np.ones_like(bg_grid)], axis=1)
    theta, *_ = np.linalg.lstsq(A, gt_mean_raw, rcond=None)
    a_scale, c_offset = float(theta[0]), float(theta[1])
    bg_scaled_grid = a_scale * bg_grid + c_offset

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(wl_grid_plot, bg_scaled_grid, label="Rescaled background (scaled)", color="#1f77b4", linewidth=2.0)
    gt_palette = ["#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, wl, val) in enumerate(gt_curves):
        ax.plot(wl, val, label=name, linewidth=2.2, color=gt_palette[i % len(gt_palette)])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.set_title("Background vs. ground-truth")
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    out_plot = out_dir / "figure04_rescaled_bg_vs_groundtruth.pdf"
    fig.savefig(out_plot, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_plot.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "series_npz": "figure04_rescaled_bg_series.npz",
        "groundtruth_directory": str(gt_dir),
        "groundtruth_files": [name for name, _, _ in gt_curves],
        "neg_scale": float(neg_scale),
        "rescale_step_us": float(step_us),
        "alignment": {
            "time_range_ms": [float(time_ms[0]), float(time_ms[-1])],
            "wavelength_range_nm": [x_min, x_max],
            "slope_nm_per_ms": float(slope),
            "intercept_nm": float(intercept),
        },
        "bin_mapping": mapping,
        "bg_vs_gt_plot": out_plot.name,
    }
    with (out_dir / "figure04_rescaled_bg_alignment.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return payload


def plot_three_panel_normalised(
    series: Dict[str, np.ndarray],
    gt_curves: List[Tuple[str, np.ndarray, np.ndarray]],
    wl_series: np.ndarray,
    out_dir: Path,
    save_png: bool,
) -> None:
    """Draw 3 subplots:
    (1) Normalised GT (0..1) vs wavelength (each GT separately)
    (2) Normalised BG (0..1) vs time (5 ms series)
    (3) Aligned overlay: BG (0..1) mapped to wavelength vs normalised GT.
    """
    time_ms = series["time_ms"].astype(np.float32)
    exp_rescaled = series["exp_rescaled"].astype(np.float32)

    # Normalise BG by plateau baseline and peak
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    bg_norm = np.clip(normalise_curve(exp_rescaled, region_bg), 0.0, None)

    # Normalise each GT on its own
    gt_norm_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    wl_min = np.inf
    wl_max = -np.inf
    for name, wl, val in gt_curves:
        wl_min = min(wl_min, float(np.min(wl)))
        wl_max = max(wl_max, float(np.max(wl)))
        smooth = moving_average(val, max(21, len(val) // 300))
        region_gt = detect_active_region(smooth)
        val_norm = np.clip(normalise_curve(smooth, region_gt), 0.0, None)
        gt_norm_list.append((name, wl.astype(np.float32), val_norm.astype(np.float32)))

    # Panel 3 prep: BG mapped to wavelength; clip within GT span
    mask = (wl_series >= wl_min) & (wl_series <= wl_max)
    wl_bg = wl_series[mask].astype(np.float32)
    bg_norm_wl = bg_norm[mask]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), constrained_layout=True)

    # (1) GT normalised
    ax0 = axes[0]
    for name, wl, valn in gt_norm_list:
        ax0.plot(wl, valn, linewidth=1.6, label=name)
    ax0.set_title("GT (normalised)")
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Norm. intensity")
    ax0.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax0.legend(loc="best", fontsize=8)

    # (2) BG normalised (vs time)
    ax1 = axes[1]
    ax1.plot(time_ms, bg_norm, color="#1f77b4", linewidth=1.6)
    ax1.set_title("BG (normalised, 5 ms)")
    ax1.set_xlabel("Relative time (ms)")
    ax1.set_ylabel("Norm. intensity")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    # (3) Aligned overlay: BG mapped to wavelength vs GT normalised
    ax2 = axes[2]
    ax2.plot(wl_bg, bg_norm_wl, color="#1f77b4", linewidth=1.8, label="BG mapped")
    for name, wl, valn in gt_norm_list:
        ax2.plot(wl, valn, linewidth=1.4, label=name)
    ax2.set_title("Aligned (normalised)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Norm. intensity")
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax2.legend(loc="best", fontsize=8)

    out_path = out_dir / "figure04_rescaled_bg_gt_threepanel.pdf"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_background_spectrum(
    series: Dict[str, np.ndarray],
    bg_values: np.ndarray,
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    output_dir: Path,
    save_png: bool,
) -> None:
    time_ms = series.get("time_ms")
    if time_ms is None or np.size(time_ms) == 0:
        return
    exp_unscaled = series.get("exp_unscaled")
    exp_rescaled = series.get("exp_rescaled")
    indices = [m["index"] for m in metadata]
    if not indices:
        return
    base_start = metadata[0]["start_us"]
    centres_ms = [((m["start_us"] - base_start) / 1000.0) + (m["duration_ms"] / 2.0) for m in metadata]
    mask = [start_bin <= idx <= end_bin for idx in indices]
    selected = [m for m in metadata if start_bin <= m["index"] <= end_bin]
    hl_start = hl_end = None
    if selected:
        hl_start = (selected[0]["start_us"] - base_start) / 1000.0
        hl_end = (selected[-1]["end_us"] - base_start) / 1000.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.6, 3.6), constrained_layout=True)
    ax0.plot(time_ms, exp_unscaled, color="#7f7f7f", linewidth=1.0, alpha=0.7, label="Original exp")
    ax0.plot(time_ms, exp_rescaled, color="#1f77b4", linewidth=1.6, label="Rescaled exp")
    if hl_start is not None and hl_end is not None:
        ax0.axvspan(hl_start, hl_end, color="#d62728", alpha=0.08, lw=0)
    ax0.set_ylabel("exp(cumulative)")
    ax0.set_title("Rescaled background spectrum")
    ax0.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax0.legend(loc="best", fontsize=8)

    # Use 5 ms series points for the lower panel as well
    ax1.plot(time_ms, np.clip(exp_rescaled, 0, None), color="#1f77b4", linewidth=1.2, label="5 ms series")
    if hl_start is not None and hl_end is not None:
        ax1.axvspan(hl_start, hl_end, color="#d62728", alpha=0.08, lw=0)
    ax1.set_xlabel("Relative time (ms)")
    ax1.set_ylabel("Background (norm.)")
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax1.legend(loc="best", fontsize=8)

    out_path = output_dir / "figure04_rescaled_bg_spectrum.pdf"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_grid(
    originals: List[np.ndarray],
    compensated: List[np.ndarray],
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    raw_cmap: Colormap,
    comp_cmap: Colormap,
    wavelength_lookup: Dict[int, float] | None,
    show_colorbar: bool,
    output_dir: Path,
    save_png: bool,
    title_suffix: str = "",
) -> None:
    setup_style()

    selected = [
        (orig, comp, meta)
        for orig, comp, meta in zip(originals, compensated, metadata)
        if start_bin <= meta["index"] <= end_bin
    ]
    num_cols = len(selected)
    fig = plt.figure(figsize=(1.2 * num_cols + 0.4, 1.8))
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
    fig.subplots_adjust(left=0.02, right=0.995, top=0.995, bottom=0.14)

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
        im0 = ax_orig.imshow(orig_frame, cmap=raw_cmap, vmin=raw_vmin, vmax=raw_vmax, origin="lower")
        ax_orig.axis("off")
        ax_comp = axes[1, col]
        im1 = ax_comp.imshow(
            comp_frame,
            cmap=comp_cmap,
            vmin=None if comp_norm else comp_vmin,
            vmax=None if comp_norm else comp_vmax,
            norm=comp_norm,
            origin="lower",
        )
        ax_comp.axis("off")
        if wavelength_lookup is not None and meta["index"] in wavelength_lookup:
            ax_comp.text(
                0.5,
                -0.12,
                f"{wavelength_lookup[meta['index']]:.0f} nm",
                transform=ax_comp.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45, edgecolor="none"),
            )

        if show_colorbar:
            fig.colorbar(im0, ax=ax_orig, shrink=0.85, pad=0.01)
            fig.colorbar(im1, ax=ax_comp, shrink=0.85, pad=0.01)

    stem = f"figure04_rescaled_grid_bins_{start_bin:02d}_{end_bin:02d}{title_suffix}"
    out_stem = output_dir / stem
    fig.savefig(f"{out_stem}.pdf", dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    segment_path = args.segment.resolve()
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    figures_root = Path(__file__).resolve().parent / "figures"
    # Always use a timestamped folder to avoid overwrites
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (figures_root / f"figure04_allinone_{suffix}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events
    x, y, t, p = load_segment_events(segment_path)
    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)

    # Compensation params
    params_file = find_param_file(segment_path)
    a_params = b_params = None
    if params_file is not None:
        params = load_params(params_file)
        a_params = params["a_params"]
        b_params = params["b_params"]

    # 1) Fine 5 ms cumulative series + neg_scale
    neg_scale, series, comp_weights = compute_series_and_weights(
        x.astype(np.float32),
        y.astype(np.float32),
        t.astype(np.float32),
        p.astype(np.float32),
        a_params,
        b_params,
        sensor_area,
        args.pos_scale,
        args.neg_scale,
        args.fine_step_us,
    )
    save_series(series, out_dir)

    # Prepare colormaps
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", COMP_LIGHTEN_FRACTION)

    # Build 50 ms frames
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / args.bin_width_us))
    originals: List[np.ndarray] = []
    comp_raw_frames: List[np.ndarray] = []
    metadata_bins: List[Dict[str, float]] = []

    if a_params is not None and b_params is not None:
        a_avg = float(np.mean(a_params))
        b_avg = float(np.mean(b_params))
        t_comp_for_bins = t - (a_avg * x + b_avg * y)
    else:
        t_comp_for_bins = t

    for idx in range(num_bins):
        start = t_min + idx * args.bin_width_us
        end = start + args.bin_width_us
        mask_orig = (t >= start) & (t < end)
        mask_comp = (t_comp_for_bins >= start) & (t_comp_for_bins < end)
        originals.append(accumulate_bin(x, y, mask_orig, np.ones_like(p, dtype=np.float32), sensor_shape))
        comp_raw_frames.append(accumulate_bin(x, y, mask_comp, comp_weights, sensor_shape).astype(np.float32))
        end_clamped = float(min(end, t_max))
        metadata_bins.append(
            {
                "index": int(idx),
                "start_us": float(start),
                "end_us": end_clamped,
                "duration_ms": float((end_clamped - start) / 1000.0),
            }
        )

    comp_array = np.stack(comp_raw_frames, axis=0)
    if args.smooth:
        comp_array = smooth_volume_3d(comp_array)
    bg_values = comp_array.mean(axis=(1, 2))
    compensated = [subtract_background(f) for f in comp_array]

    # 2) Save time-domain background spectrum plot (matching prior outputs)
    plot_background_spectrum(series, bg_values, metadata_bins, args.start_bin, args.end_bin, out_dir, args.save_png)

    # 3) Align with GT and save mapping
    alignment = align_with_groundtruth(series, args.gt_dir.resolve(), metadata_bins, out_dir, neg_scale, args.fine_step_us, args.save_png)
    wavelength_lookup = {int(item["index"]): float(item["wavelength_nm"]) for item in alignment["bin_mapping"]}

    # 4) Render grid with wavelength labels
    render_grid(
        originals,
        compensated,
        metadata_bins,
        args.start_bin,
        args.end_bin,
        raw_cmap,
        comp_cmap,
        wavelength_lookup if args.show_wavelength else None,
        False,
        out_dir,
        args.save_png,
    )

    # 5) Three-panel normalised overview
    gt_curves = load_gt_curves(args.gt_dir.resolve())
    wl_series = (alignment["alignment"]["slope_nm_per_ms"] * series["time_ms"] + alignment["alignment"]["intercept_nm"]).astype(np.float32)
    plot_three_panel_normalised(series, gt_curves, wl_series, out_dir, args.save_png)

    # Persist weights metadata
    weights_json = out_dir / "figure04_rescaled_weights.json"
    with weights_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "segment": str(segment_path),
                "pos_scale": args.pos_scale,
                "neg_scale": neg_scale,
                "bin_width_us": args.bin_width_us,
                "bin_width_ms": args.bin_width_us / 1000.0,
                "num_bins": num_bins,
                "smooth": bool(args.smooth),
                "bin_times_us": metadata_bins,
                "background_means": bg_values.tolist(),
                "rescale_step_us": args.fine_step_us,
                "bg_series_npz": "figure04_rescaled_bg_series.npz",
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()
