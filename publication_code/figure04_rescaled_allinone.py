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
    align_series_to_wavelength,
    detect_visible_edges,
    load_gt_curves,
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
    ap.add_argument("--smooth", action="store_true")
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
    wl_series, series_norm, slope, intercept = align_series_to_wavelength(
        time_ms, exp_rescaled, gt_start_nm, gt_end_nm
    )

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

    # Plot background vs GT in wavelength domain
    x_min, x_max = float(gt_start_nm), float(gt_end_nm)
    mask_bg = (wl_series >= x_min) & (wl_series <= x_max)
    wl_bg = wl_series[mask_bg]
    bg_norm = series_norm[mask_bg]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(wl_bg, bg_norm, label="Rescaled background", color="#1f77b4", linewidth=1.6)
    for name, wl, val in gt_curves:
        sel = (wl >= x_min) & (wl <= x_max)
        ax.plot(wl[sel], val[sel], label=name, linewidth=1.4)
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
    if args.output_dir is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = figures_root / f"figure04_allinone_{suffix}"
    out_dir = args.output_dir.resolve()
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

    # 2) Align with GT and save mapping
    alignment = align_with_groundtruth(series, args.gt_dir.resolve(), metadata_bins, out_dir, neg_scale, args.fine_step_us, args.save_png)
    wavelength_lookup = {int(item["index"]): float(item["wavelength_nm"]) for item in alignment["bin_mapping"]}

    # 3) Render grid with wavelength labels
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
