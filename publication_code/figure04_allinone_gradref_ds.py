#!/usr/bin/env python3
"""
Figure 4 (GradRef, downsampled):

Render a 4-row grid (Orig., Comp., Diff., Ref.) plus a wavelength bar, but
downsample the displayed time bins to keep only columns 1,4,7,10,13 within
the chosen [start_bin, end_bin] range (i.e., a stride of 3 over the selected
bins). Between the kept columns, insert narrow spacer columns showing "…" and
shrink the wavelength bar accordingly so it has one stripe per kept column.

This script is a derivative of figure04_allinone_gradref.py, focused on
publication layout with fewer columns.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
from scripts.hs_to_rgb import load_cie_cmf, xyz_to_srgb  # color utilities
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
    ap = argparse.ArgumentParser(description="Figure 4 GradRef (downsampled) pipeline")
    ap.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_events.npz")
    ap.add_argument("--gt-dir", type=Path, default=Path("groundtruth_spectrum_2835"))
    ap.add_argument("--diff-frames-dir", type=Path, required=True, help="Folder of Diff images (e.g., gradient_20nm *_XXXtoYYYnm.png)")
    ap.add_argument("--ref-frames-dir", type=Path, required=True, help="Folder of Ref images (e.g., ROI matched *_XXXnm.png)")
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
    ap.add_argument("--smooth", action="store_true", default=True)
    ap.add_argument("--show-wavelength", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--bar-height-ratio", type=float, default=0.08)
    ap.add_argument("--bar-px", type=int, default=6)
    # Downsampling controls; default pattern keeps 1,4,7,10,13 (stride 3)
    ap.add_argument("--downsample-rate", type=int, default=3, help="Keep every Nth column from the start of the selected range")
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


def load_cie(repo_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wl, xbar, ybar, zbar = load_cie_cmf(repo_root)
    return wl.astype(np.float32), xbar.astype(np.float32), ybar.astype(np.float32), zbar.astype(np.float32)


def nm_to_rgb_color(nm: float, repo_root: Path) -> np.ndarray:
    wl, xbar, ybar, zbar = load_cie(repo_root)
    x = float(np.interp(nm, wl, xbar))
    y = float(np.interp(nm, wl, ybar))
    z = float(np.interp(nm, wl, zbar))
    XYZ = np.array([x, y, z], dtype=np.float32)
    scale = 1.0 / XYZ[1] if XYZ[1] > 0 else 1.0
    rgb = xyz_to_srgb(XYZ[None, :] * scale)[0]
    return rgb


def _parse_start_nm(name: str) -> float | None:
    m = re.match(r".*_(\d+(?:\.\d+)?)nm\.(?:png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = re.match(r".*_(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)nm\.(?:png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = re.match(r".*_(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)\.(?:png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def list_gt_frames_with_nm(frames_dir: Path) -> List[Tuple[float, Path]]:
    items: List[Tuple[float, Path]] = []
    for p in sorted(frames_dir.glob("*.png")):
        nm = _parse_start_nm(p.name)
        if nm is None:
            continue
        items.append((float(nm), p))
    return items


def select_gt_frames_for_bins(
    frames_dir: Path,
    wavelength_lookup: Dict[int, float],
    metadata: Sequence[Dict[str, float]],
) -> List[Tuple[int, float, Path]]:
    items = list_gt_frames_with_nm(frames_dir)
    if not items:
        return []
    wl_vals = np.array([nm for nm, _ in items], dtype=np.float32)
    result: List[Tuple[int, float, Path]] = []
    for m in metadata:
        idx = int(m["index"])
        nm = float(wavelength_lookup.get(idx, np.nan))
        if np.isnan(nm):
            continue
        j = int(np.argmin(np.abs(wl_vals - nm)))
        sel_nm, sel_path = items[j]
        result.append((idx, float(sel_nm), sel_path))
    return result


def nearest_wavelength_lookup(
    continuous_lookup: Dict[int, float],
    gt_curves: List[Tuple[str, np.ndarray, np.ndarray]],
) -> Dict[int, float]:
    if not gt_curves:
        return dict(continuous_lookup)
    _, wl_gt, _ = gt_curves[0]
    wl_gt = wl_gt.astype(np.float32)
    out: Dict[int, float] = {}
    for idx, nm in continuous_lookup.items():
        j = int(np.argmin(np.abs(wl_gt - float(nm))))
        out[int(idx)] = float(wl_gt[j])
    return out


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

    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    t0 = float(time_ms[region_bg.start_idx])
    t1 = float(time_ms[region_bg.end_idx])

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
        wl0_list.append(float(wl_sel[region_gt.start_idx]))
        wl1_list.append(float(wl_sel[region_gt.end_idx]))
    if wl0_list and wl1_list:
        wl0 = float(np.mean(wl0_list))
        wl1 = float(np.mean(wl1_list))
    else:
        wl0, wl1 = float(gt_start_nm), float(gt_end_nm)

    slope = (wl1 - wl0) / (t1 - t0) if t1 != t0 else (wl1 - wl0)
    intercept = wl0 - slope * t0
    wl_series = slope * time_ms + intercept
    series_norm = normalise_curve(exp_rescaled, region_bg)

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

    # Persist alignment summary and BG vs SPD overlay plot
    wl_min_all = min(float(np.min(wl)) for _, wl, _ in gt_curves)
    wl_max_all = max(float(np.max(wl)) for _, wl, _ in gt_curves)
    mask_bg = (wl_series >= wl_min_all) & (wl_series <= wl_max_all)
    wl_bg = wl_series[mask_bg]
    bg_norm = series_norm[mask_bg]

    wl_grid_plot = np.linspace(wl_min_all, wl_max_all, 2000, dtype=np.float32)
    gt_grid_stack: list[np.ndarray] = []
    for _, wl, val in gt_curves:
        gt_grid_stack.append(np.interp(wl_grid_plot, wl, val).astype(np.float32))
    gt_mean_raw = np.mean(gt_grid_stack, axis=0)
    bg_grid = np.interp(wl_grid_plot, wl_bg, bg_norm) if wl_bg.size else np.zeros_like(wl_grid_plot)
    A = np.stack([bg_grid, np.ones_like(bg_grid)], axis=1)
    theta, *_ = np.linalg.lstsq(A, gt_mean_raw, rcond=None)
    a_scale, c_offset = float(theta[0]), float(theta[1])
    bg_scaled_grid = a_scale * bg_grid + c_offset

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    BG_COLOR = "#1f77b4"; SPD_COLOR = "#ff7f0e"
    ax.plot(wl_grid_plot, bg_scaled_grid, label="Rescaled background (scaled)", color=BG_COLOR, linewidth=2.0)
    if gt_curves:
        _, wl_single, val_single = gt_curves[0]
        ax.plot(wl_single, val_single, label="Light SPD", linewidth=2.2, color=SPD_COLOR)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Normalised intensity (a.u.)")
    ax.set_title("Background vs. Light SPD")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
    ax.set_xlim(x_min, x_max); ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    out_plot = out_dir / "figure04_rescaled_bg_vs_groundtruth.pdf"
    fig.savefig(out_plot, dpi=400, bbox_inches="tight")
    ax.legend(loc="best", fontsize=8)
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
    base_start = metadata[0]["start_us"]
    selected = [m for m in metadata if start_bin <= m["index"] <= end_bin]
    hl_start = (selected[0]["start_us"] - base_start) / 1000.0 if selected else None
    hl_end = (selected[-1]["end_us"] - base_start) / 1000.0 if selected else None

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.6, 3.6), constrained_layout=True)
    ax0.plot(time_ms, exp_unscaled, color="#7f7f7f", linewidth=1.0, alpha=0.7, label="Original exp")
    ax0.plot(time_ms, exp_rescaled, color="#1f77b4", linewidth=1.6, label="Rescaled exp")
    if hl_start is not None and hl_end is not None:
        ax0.axvspan(hl_start, hl_end, color="#d62728", alpha=0.08, lw=0)
    ax0.set_ylabel("exp(cumulative)")
    ax0.set_title("Rescaled background spectrum")
    ax0.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax0.legend(loc="best", fontsize=8)

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


def render_grid_downsampled(
    originals: List[np.ndarray],
    compensated: List[np.ndarray],
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    raw_cmap: Colormap,
    comp_cmap: Colormap,
    wavelength_lookup: Dict[int, float] | None,
    output_dir: Path,
    save_png: bool,
    add_gt_row: bool,
    gt_wavelength_lookup: Dict[int, float] | None,
    ref_frame_paths: List[Path] | None,
    diff_frame_paths: List[Path] | None,
    bar_height_ratio: float,
    bar_px: int,
    downsample_rate: int,
) -> None:
    setup_style()

    # Full selection across [start_bin, end_bin]
    full = [
        (orig, comp, meta)
        for orig, comp, meta in zip(originals, compensated, metadata)
        if start_bin <= meta["index"] <= end_bin
    ]
    if not full:
        return
    # Keep positions 0,3,6, ... (1,4,7, ... in 1-based)
    keep_positions = list(range(0, len(full), max(1, int(downsample_rate))))
    # Ensure last column included (if not already)
    if keep_positions[-1] != len(full) - 1:
        keep_positions.append(len(full) - 1)
    kept = [full[i] for i in keep_positions]

    # For external images, downsample in the same pattern
    ref_paths_sel = [ref_frame_paths[i] for i in keep_positions] if ref_frame_paths else []
    diff_paths_sel = [diff_frame_paths[i] for i in keep_positions] if diff_frame_paths else []

    n_img_cols = len(kept)
    n_ellipsis = n_img_cols - 1
    grid_cols = n_img_cols + n_ellipsis  # interleave '…' spacers

    # Determine rows used: 3 content rows + optional Ref row + bar row
    has_gt_images = add_gt_row and bool(ref_paths_sel)
    n_rows = 3 + (1 if has_gt_images else 0) + 1  # include bar row
    height_ratios: List[float] = [1.0, 1.0, 1.0]
    if has_gt_images:
        height_ratios.append(1.0)
    height_ratios.append(max(0.02, float(bar_height_ratio)))

    # Build width ratios: label col + interleaved image/ellipsis columns
    width_ratios: List[float] = [0.22]
    for i in range(grid_cols):
        width_ratios.append(1.0 if (i % 2 == 0) else 0.12)

    fig_height = 3.2 if n_rows == 4 else 3.6
    fig = plt.figure(figsize=(1.2 * n_img_cols + 0.4 + 0.3 * n_ellipsis, fig_height))
    gs = fig.add_gridspec(
        n_rows,
        grid_cols + 1,
        wspace=0.045,
        hspace=0.06,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )

    # Allocate axes for image columns
    axes = np.empty((n_rows, n_img_cols), dtype=object)
    for row in range(n_rows):
        # Row label axis
        label_ax = fig.add_subplot(gs[row, 0]); label_ax.axis("off")
        label_text = (
            "Orig." if row == 0 else ("Comp." if row == 1 else ("Diff." if row == 2 else ("Ref." if (has_gt_images and row == 3) else "")))
        )
        label_ax.text(0.5, 0.5, label_text, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
        # Populate content axes on even grid columns (1,3,5,...) skipping ellipsis columns
        for i in range(n_img_cols):
            grid_col = 1 + 2 * i
            axes[row, i] = fig.add_subplot(gs[row, grid_col])
    fig.subplots_adjust(left=0.02, right=0.995, top=0.995, bottom=0.16)

    # Populate image data
    for i, (orig_frame, comp_frame, meta) in enumerate(kept):
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
            comp_abs = max(abs(comp_vmin), abs(comp_vmax)) or 1.0
            comp_norm = TwoSlopeNorm(vmin=-comp_abs, vcenter=0.0, vmax=comp_abs)

        ax0 = axes[0, i]
        ax0.imshow(orig_frame, cmap=raw_cmap, vmin=raw_vmin, vmax=raw_vmax, origin="lower"); ax0.axis("off")
        ax1 = axes[1, i]
        ax1.imshow(comp_frame, cmap=comp_cmap, vmin=None if comp_norm else comp_vmin, vmax=None if comp_norm else comp_vmax, norm=comp_norm, origin="lower"); ax1.axis("off")
        if wavelength_lookup is not None and meta["index"] in wavelength_lookup:
            ax1.text(0.5, -0.12, f"{wavelength_lookup[meta['index']]:.0f} nm", transform=ax1.transAxes, ha="center", va="top", fontsize=8)

        # Diff row: external images if provided
        ax2 = axes[2, i]
        if diff_paths_sel and i < len(diff_paths_sel) and diff_paths_sel[i] and diff_paths_sel[i].exists():
            import matplotlib.image as mpimg
            img_diff = mpimg.imread(str(diff_paths_sel[i]))
            ax2.imshow(img_diff, origin="lower", cmap="gray")
        else:
            diff_frame = (comp_frame.astype(np.float32) - orig_frame.astype(np.float32))
            ax2.imshow(diff_frame, cmap=comp_cmap, origin="lower")
        ax2.axis("off")

        # Optional Ref row
        if has_gt_images:
            ax3 = axes[3, i]
            if ref_paths_sel and i < len(ref_paths_sel):
                import matplotlib.image as mpimg
                img_ref = mpimg.imread(str(ref_paths_sel[i]))
                ax3.imshow(img_ref, origin="lower", cmap="gray")
            else:
                ax3.imshow(np.zeros_like(comp_frame), cmap="gray", origin="lower")
            ax3.axis("off")

    # Draw ellipsis columns (use the bar row; if no bar row, use Comp. row)
    bar_row_idx = 3 if not has_gt_images else 4
    for k in range(n_ellipsis):
        col = 2 + 2 * k  # between two image cols
        ax_dot = fig.add_subplot(gs[bar_row_idx, col])
        ax_dot.axis("off")
        ax_dot.text(0.5, 0.5, "…", ha="center", va="center", fontsize=14, color="black")

    # Build a wavelength bar with one stripe per kept column (selected wavelengths)
    bar_colors: List[np.ndarray] = []
    for _, _, meta in kept:
        nm_val = None
        if gt_wavelength_lookup and meta["index"] in gt_wavelength_lookup:
            nm_val = float(gt_wavelength_lookup[meta["index"]])
        elif wavelength_lookup and meta["index"] in wavelength_lookup:
            nm_val = float(wavelength_lookup[meta["index"]])
        if nm_val is None:
            rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            rgb = nm_to_rgb_color(nm_val, REPO_ROOT)
        bar_colors.append(np.tile(rgb[None, None, :], (int(bar_px), 1, 1)))  # (H,1,3)
    if bar_colors:
        bar_img = np.concatenate(bar_colors, axis=1)
        ax_bar = fig.add_subplot(gs[bar_row_idx, 1:])
        ax_bar.imshow(bar_img, origin="lower", aspect="auto"); ax_bar.axis("off")

    stem = f"figure04_rescaled_grid_bins_{start_bin:02d}_{end_bin:02d}_ds{downsample_rate}"
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
    out_dir = (figures_root / f"figure04_allinone_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events and params
    x, y, t, p = load_segment_events(segment_path)
    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)
    params_file = find_param_file(segment_path)
    a_params = b_params = None
    if params_file is not None:
        params = load_params(params_file)
        a_params = params["a_params"]; b_params = params["b_params"]

    # Background series + neg weight
    neg_scale, series, comp_weights = compute_series_and_weights(
        x.astype(np.float32), y.astype(np.float32), t.astype(np.float32), p.astype(np.float32),
        a_params, b_params, sensor_area, args.pos_scale, args.neg_scale, args.fine_step_us,
    )
    save_series(series, out_dir)

    # Colormaps
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", COMP_LIGHTEN_FRACTION)

    # Build per-bin frames
    t_min = float(np.min(t)); t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / args.bin_width_us))
    originals: List[np.ndarray] = []
    comp_raw_frames: List[np.ndarray] = []
    metadata_bins: List[Dict[str, float]] = []
    if a_params is not None and b_params is not None:
        a_avg = float(np.mean(a_params)); b_avg = float(np.mean(b_params))
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
        metadata_bins.append({"index": int(idx), "start_us": float(start), "end_us": end_clamped, "duration_ms": float((end_clamped - start) / 1000.0)})

    comp_array = np.stack(comp_raw_frames, axis=0)
    if args.smooth:
        comp_array = smooth_volume_3d(comp_array)
    compensated = [subtract_background(f) for f in comp_array]

    # Background spectrum plot
    plot_background_spectrum(series, comp_array.mean(axis=(1, 2)), metadata_bins, args.start_bin, args.end_bin, out_dir, args.save_png)

    # Align with GT and produce wavelength mapping
    alignment = align_with_groundtruth(series, args.gt_dir.resolve(), metadata_bins, out_dir, neg_scale, args.fine_step_us, args.save_png)
    wavelength_lookup = {int(item["index"]): float(item["wavelength_nm"]) for item in alignment["bin_mapping"]}
    gt_curves = load_gt_curves(args.gt_dir.resolve())

    # Select Diff/Ref frames for full range (we'll downsample later to match kept columns)
    displayed_meta = [m for m in metadata_bins if args.start_bin <= m["index"] <= args.end_bin]
    diff_selections = select_gt_frames_for_bins(args.diff_frames_dir.resolve(), wavelength_lookup, displayed_meta)
    ref_selections = select_gt_frames_for_bins(args.ref_frames_dir.resolve(), wavelength_lookup, displayed_meta)
    diff_frame_paths: List[Path] = []
    ref_frame_paths: List[Path] = []
    diff_sel_dir = out_dir / "diff_selected_frames"; diff_sel_dir.mkdir(parents=True, exist_ok=True)
    ref_sel_dir = out_dir / "gt_selected_frames"; ref_sel_dir.mkdir(parents=True, exist_ok=True)
    ref_sel_dir2 = out_dir / "ref_selected_frames"; ref_sel_dir2.mkdir(parents=True, exist_ok=True)
    for idx, nm_sel, pth in diff_selections:
        dest = diff_sel_dir / f"bin_{idx:02d}_{nm_sel:.0f}nm{pth.suffix}"
        try:
            shutil.copy2(pth, dest); diff_frame_paths.append(dest)
        except Exception:
            diff_frame_paths.append(pth)
    for idx, nm_sel, pth in ref_selections:
        dest = ref_sel_dir / f"bin_{idx:02d}_{nm_sel:.0f}nm{pth.suffix}"
        try:
            shutil.copy2(pth, dest); shutil.copy2(pth, ref_sel_dir2 / dest.name); ref_frame_paths.append(dest)
        except Exception:
            ref_frame_paths.append(pth)

    # Render downsampled grid with interleaved ellipses and adjusted bar
    render_grid_downsampled(
        originals,
        compensated,
        metadata_bins,
        args.start_bin,
        args.end_bin,
        raw_cmap,
        comp_cmap,
        wavelength_lookup if args.show_wavelength else None,
        out_dir,
        args.save_png,
        add_gt_row=True,
        gt_wavelength_lookup=nearest_wavelength_lookup(wavelength_lookup, gt_curves),
        ref_frame_paths=ref_frame_paths,
        diff_frame_paths=diff_frame_paths,
        bar_height_ratio=float(args.bar_height_ratio),
        bar_px=int(args.bar_px),
        downsample_rate=int(args.downsample_rate),
    )

    # Persist weights metadata (same as base script)
    weights_json = out_dir / "figure04_rescaled_weights.json"
    with weights_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "segment": str(segment_path),
                "pos_scale": args.pos_scale,
                "neg_scale": neg_scale,
                "bin_width_us": args.bin_width_us,
                "bin_width_ms": args.bin_width_us / 1000.0,
                "smooth": bool(args.smooth),
                "rescale_step_us": args.fine_step_us,
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()

