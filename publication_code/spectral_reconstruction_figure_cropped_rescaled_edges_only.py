#!/usr/bin/env python3
"""
Spectral reconstruction (cropped, GradRef) with edges-only alignment.

This combines the publication 4-row spectral grid with the stricter
edges-only time→wavelength alignment (rising first-crossing and falling
last-crossing quantiles), so the mapped background matches start/end flats.

Outputs match the usual spectral reconstruction figure plus the full
"all-in-one" background artifacts (alignment JSON, series, BG vs GT
overlays, three-panel normalised, grid bins, weights JSON).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Colormap, TwoSlopeNorm, Normalize

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
    load_gt_curves,
)
from groundtruth_spectrum.compare_reconstruction_to_gt import (
    moving_average,
    normalise_curve,
    detect_active_region,
)
from publication_code.figure04_rescaled_edges_only import align_edges_only


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Spectral reconstruction (cropped) with edges-only alignment")
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
    # Layout gaps
    ap.add_argument("--col-gap", type=float, default=0.045, help="Column gap (matplotlib wspace)")
    ap.add_argument("--row-gap", type=float, default=None, help="Row gap (matplotlib hspace); default = 0.35 * col-gap")
    ap.add_argument("--column-step", type=int, default=1, help="Keep every Nth column after selection (e.g., 2 keeps indices 1,3,5,...)")
    # Column selection by wavelength (optional). If provided, overrides start/end/downsample.
    ap.add_argument("--wl-min", type=float, default=None, help="Minimum wavelength (nm) for column selection")
    ap.add_argument("--wl-max", type=float, default=None, help="Maximum wavelength (nm) for column selection")
    ap.add_argument("--wl-step", type=float, default=None, help="Step in nm for selecting columns (e.g., 20 for 400,420,...) ")
    ap.add_argument("--wl-list", type=str, default=None, help="Comma-separated list of wavelengths (nm) to select as columns, e.g. '400,450,500' ")
    # Bottom spectrum bar extent override
    ap.add_argument("--bar-wl-min", type=float, default=None, help="Force spectrum bar to start at this nm (default: min selected nm)")
    ap.add_argument("--bar-wl-max", type=float, default=None, help="Force spectrum bar to end at this nm (default: max selected nm)")
    # Image aspect controls. Use 'equal' for sensor rows to preserve pixel aspect;
    # 'auto' can be used on external rows to pack vertically.
    ap.add_argument("--image-aspect12", type=str, choices=["equal", "auto"], default="equal",
                    help="Aspect for rows 1–2 (Original/Comp.)")
    ap.add_argument("--image-aspect34", type=str, choices=["equal", "auto"], default="equal",
                    help="Aspect for rows 3–4 (Gradient/Reference)")
    # Row 1–2 colorbars at right
    ap.add_argument("--no-row12-colorbars", action="store_true", help="Disable colorbars for rows 1–2 at the right")
    ap.add_argument("--cbar-ratio", type=float, default=0.10, help="Width ratio for the colorbar column (relative to an image column)")
    # Row 3–4 colorbar at right
    ap.add_argument("--row34-colorbar", action="store_true", help="Add a colorbar for rows 3–4 (external images)")
    ap.add_argument("--row34-cmap", type=str, default="gray", help="Colormap name for the rows 3–4 colorbar (intensity)")
    ap.add_argument("--single-colorbar", action="store_true", help="Use one tall colorbar spanning rows 1–4; overrides individual row colorbars")
    # Shared sensor (rows 1–2) colorbar and mapping
    ap.add_argument("--row12-shared-cbar", action="store_true", help="Use a single shared colorbar for rows 1–2; applies comp colormap/norm to both")
    # Unified scaling for rows 1–2
    ap.add_argument("--unified-row12-scales", action="store_true", help="Use a single global scale per row across all selected columns for rows 1–2")
    ap.add_argument("--raw-global-vmin", type=float, default=None, help="Override global vmin for row 1 (Original)")
    ap.add_argument("--raw-global-vmax", type=float, default=None, help="Override global vmax for row 1 (Original)")
    ap.add_argument("--comp-global-abs", type=float, default=None, help="Override symmetric abs value for TwoSlopeNorm on row 2 (Comp.)")
    ap.add_argument("--downsample-rate", type=int, default=3, help="Keep every Nth column from the selected range (1,4,7,...) ")
    ap.add_argument("--crop-json", type=Path, default=None, help="JSON with bbox {x0,y0,x1,y1} for cropping Orig./Comp. rows (sensor coords)")
    ap.add_argument("--external-crop-json", type=Path, default=None, help="JSON with bbox {x0,y0,x1,y1} for cropping Diff/Ref rows (PNG coords)")
    ap.add_argument("--flip-row12", action="store_true", help="Flip rows 1 and 2 vertically before cropping")
    ap.add_argument("--flip-row34", action="store_true", help="Flip rows 3 and 4 vertically before cropping")
    ap.add_argument("--figure-name", type=str, default=None, help="Base filename for rendered grid (e.g., spectral_reconstruction_scan_rotated_cropped_400_700)")
    ap.add_argument("--edge-quantile", type=float, default=0.05, help="Quantile for edge detection (default 0.05)")
    return ap.parse_args()


def _load_crop_box(path: Path | None, preferred_key: str | None = None) -> Tuple[int, int, int, int] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    cand = None
    keys = []
    if preferred_key:
        keys.append(preferred_key)
    keys += ["bbox", "ref_crop", "template_crop", "bbox_xyxy"]
    for key in keys:
        if key in payload:
            cand = payload[key]
            break
    if isinstance(cand, dict):
        x0, y0, x1, y1 = cand["x0"], cand["y0"], cand["x1"], cand["y1"]
    elif isinstance(cand, (list, tuple)) and len(cand) == 4:
        x0, y0, x1, y1 = cand
    else:
        return None
    return int(y0), int(y1), int(x0), int(x1)


def nm_to_rgb_color(nm: float, repo_root: Path) -> np.ndarray:
    wl, xbar, ybar, zbar = load_cie_cmf(repo_root)
    wl = wl.astype(np.float32); xbar = xbar.astype(np.float32); ybar = ybar.astype(np.float32); zbar = zbar.astype(np.float32)
    x = float(np.interp(nm, wl, xbar)); y = float(np.interp(nm, wl, ybar)); z = float(np.interp(nm, wl, zbar))
    XYZ = np.array([x, y, z], dtype=np.float32); scale = 1.0 / XYZ[1] if XYZ[1] > 0 else 1.0
    rgb = xyz_to_srgb(XYZ[None, :] * scale)[0]
    return rgb


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
        a_avg = float(np.mean(a_params)); b_avg = float(np.mean(b_params))
        t_comp = t - (a_avg * x + b_avg * y)
    else:
        t_comp = t
    neg_scale, series = auto_scale_neg_weight(
        t_comp.astype(np.float32), p.astype(np.float32), sensor_area=sensor_area, step_us=step_us,
        pos_scale=pos_scale, neg_scale_init=neg_init, return_series=True,
    )
    weights = np.where(p >= 0, pos_scale, -neg_scale).astype(np.float32)
    return float(neg_scale), series, weights


def save_series(series: Dict[str, np.ndarray], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "figure04_rescaled_bg_series.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in series.items()})
    return path


def _downsample_indices(start_bin: int, end_bin: int, rate: int) -> List[int]:
    kept: List[int] = []
    idx = start_bin
    while idx <= end_bin:
        kept.append(idx); idx += rate
    return kept


def render_spectral_grid(
    originals: List[np.ndarray],
    compensated: List[np.ndarray],
    diff_paths: List[Path],
    ref_paths: List[Path],
    metadata: List[Dict[str, float]],
    start_bin: int,
    end_bin: int,
    downsample_rate: int,
    raw_cmap: Colormap,
    comp_cmap: Colormap,
    output_dir: Path,
    figure_name: str,
    save_png: bool,
    bar_px: int,
    flip_row12: bool = False,
    flip_row34: bool = False,
    ext_crop: Tuple[int, int, int, int] | None = None,
    sens_crop: Tuple[int, int, int, int] | None = None,
    wavelength_lookup: Dict[int, float] | None = None,
    col_gap: float = 0.045,
    row_gap: float | None = None,
    override_columns: Optional[List[Dict[str, float]]] = None,
    bar_wl_min: Optional[float] = None,
    bar_wl_max: Optional[float] = None,
    image_aspect12: str = "equal",
    image_aspect34: str = "equal",
    column_step: int = 1,
    unify_row12_scales: bool = False,
    raw_global_vmin: float | None = None,
    raw_global_vmax: float | None = None,
    comp_global_abs: float | None = None,
    add_row12_colorbars: bool = True,
    cbar_ratio: float = 0.15,
    add_row34_colorbar: bool = False,
    row34_cmap_name: str = "gray",
    single_colorbar: bool = False,
    row12_shared_cbar: bool = False,
) -> None:
    setup_style()
    if override_columns is not None and len(override_columns) > 0:
        columns = override_columns
    else:
        selected = [m for m in metadata if start_bin <= m["index"] <= end_bin]
        kept_indices = set(_downsample_indices(start_bin, end_bin, downsample_rate))
        columns = [m for m in selected if m["index"] in kept_indices]
    if column_step > 1 and len(columns) > 0:
        columns = columns[::column_step]
    num_cols = len(columns)
    # Build rows
    fig = plt.figure(figsize=(1.2 * num_cols + 0.6, 5.2))
    gap = float(col_gap)
    rgap = float(row_gap) if (row_gap is not None) else max(0.01, 0.35 * gap)
    # Reserve a single narrow column for all colorbars on the far right
    width_ratios = [0.22] + [1] * num_cols
    col_bar = None
    if add_row12_colorbars or add_row34_colorbar or single_colorbar:
        col_bar = len(width_ratios)
        width_ratios.append(cbar_ratio)
    total_cols = len(width_ratios)
    gs = fig.add_gridspec(
        5,
        total_cols,
        wspace=gap,
        hspace=rgap,
        width_ratios=width_ratios,
        height_ratios=[1.0, 1.0, 1.0, 1.0, max(0.02, float(bar_px) / 80.0)],
    )
    # Reinforce spacing in case backend ignores GridSpec hspace/wspace
    try:
        gs.update(hspace=rgap, wspace=gap)
        fig.subplots_adjust(hspace=rgap, wspace=gap)
    except Exception:
        pass
    # Track per-row image axes for precise colorbar alignment later
    row_axes = [[], [], [], []]
    label_axes = []
    def label_column(r: int, text: str) -> None:
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off"); ax.text(0.5, 0.5, text, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
        label_axes.append((r, ax))
    label_column(0, "Events"); label_column(1, "Comp."); label_column(2, "Diff."); label_column(3, "Frames")

    # Precompute a global median-abs scale for Diff (use NPZ if present, else PNG).
    diff_abs_values: List[float] = []
    def _load_diff_scalar(path: Path) -> np.ndarray | None:
        if path is None or (not path.exists()):
            return None
        arr = None
        if path.suffix.lower() == ".npz":
            try:
                arr = np.load(path)["frame"]
            except Exception:
                arr = None
        if arr is None:
            try:
                arr = plt.imread(path)
            except Exception:
                arr = None
        if arr is None:
            return None
        if ext_crop is not None and arr.ndim >= 2:
            y0, y1, x0, x1 = ext_crop
            arr = arr[y0:y1, x0:x1]
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = (0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2]).astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr
    for p in diff_paths:
        arr = _load_diff_scalar(p)
        if arr is None:
            continue
        # Collect 95th absolute values for robust scale
        vals = np.abs(arr).ravel()
        if vals.size > 0:
            diff_abs_values.append(float(np.percentile(vals, 95)))
    diff_global_den = None
    if diff_abs_values:
        diff_global_den = float(np.median(np.array(diff_abs_values, dtype=np.float32)))
        if diff_global_den <= 1e-9:
            diff_global_den = 1.0

    # Precompute unified scales for rows 1–2 if requested
    raw_vmin = raw_global_vmin
    raw_vmax = raw_global_vmax
    raw_abs_q95 = None
    comp_norm = None
    shared_norm = None
    comp_abs_q95 = None
    selected_indices = [int(m["index"]) for m in columns]
    if unify_row12_scales and selected_indices:
        # Build cropped frames for scale calculation
        def crop_sensor(frame: np.ndarray) -> np.ndarray:
            if sens_crop is None:
                return frame
            y0, y1, x0, x1 = sens_crop
            return frame[y0:y1, x0:x1]
        # Collect median abs for comp row
        comp_abs_list: List[float] = []
        # Row 1 global min/max and 95th |value|
        if raw_vmin is None or raw_vmax is None or raw_abs_q95 is None:
            rmins = []
            rmaxs = []
            rqs = []
            for idx in selected_indices:
                if not (0 <= idx < len(originals)):
                    continue
                fr = originals[idx]
                if flip_row12:
                    fr = np.flipud(fr)
                fr = crop_sensor(fr)
                rmins.append(float(np.nanmin(fr)))
                rmaxs.append(float(np.nanmax(fr)))
                rqs.append(float(np.percentile(np.abs(fr), 95)))
            if rmins and rmaxs:
                rvmin = min(rmins); rvmax = max(rmaxs)
                if np.isclose(rvmin, rvmax):
                    rvmax = rvmin + 1e-3
                raw_vmin = rvmin if raw_vmin is None else raw_vmin
                raw_vmax = rvmax if raw_vmax is None else raw_vmax
            if rqs:
                raw_abs_q95 = float(np.median(np.array(rqs, dtype=np.float32)))
                if raw_abs_q95 <= 1e-9:
                    raw_abs_q95 = None
        # Row 2 symmetric TwoSlopeNorm around 0 using global 95th abs
        if comp_abs_list or comp_abs_q95 is None:
            comp_abs_list = comp_abs_list or []
            if not comp_abs_list:
                for idx in selected_indices:
                    if not (0 <= idx < len(compensated)):
                        continue
                    fc = compensated[idx]
                    if flip_row12:
                        fc = np.flipud(fc)
                    fc = crop_sensor(fc)
                    comp_abs_list.append(float(np.percentile(np.abs(fc), 95)))
            if comp_abs_list:
                comp_abs_q95 = float(np.median(np.array(comp_abs_list, dtype=np.float32)))
                if comp_abs_q95 <= 1e-9:
                    comp_abs_q95 = None
    # Shared norm for the single colorbar path (all rows share the same TwoSlopeNorm)
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
    shared_norm = _TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    comp_norm = shared_norm
    # Draw row 1/2
    for row, frames, cmap in [(0, originals, raw_cmap), (1, compensated, comp_cmap)]:
        for ci, meta in enumerate(columns, start=0):
            idx = meta["index"]
            ax = fig.add_subplot(gs[row, ci + 1])
            frame = frames[idx] if idx < len(frames) else np.zeros_like(frames[0])
            if flip_row12:
                frame = np.flipud(frame)
            if sens_crop is not None:
                y0, y1, x0, x1 = sens_crop
                frame = frame[y0:y1, x0:x1]
            if row == 0 and raw_abs_q95 is not None and raw_abs_q95 > 0:
                frame = frame / raw_abs_q95
            if row == 1 and comp_abs_q95 is not None and comp_abs_q95 > 0:
                frame = frame / comp_abs_q95
            if single_colorbar or row12_shared_cbar:
                ax.imshow(frame, cmap=comp_cmap, origin="lower", aspect=image_aspect12, norm=shared_norm)
            else:
                if row == 0:
                    vmin_use = raw_vmin if raw_vmin is not None else -1.0
                    vmax_use = raw_vmax if raw_vmax is not None else 1.0
                    ax.imshow(frame, cmap=cmap, origin="lower", aspect=image_aspect12, vmin=vmin_use, vmax=vmax_use)
                else:
                    ax.imshow(frame, cmap=cmap, origin="lower", aspect=image_aspect12, norm=shared_norm)
            ax.axis("off")
            # record axis for later colorbar alignment
            row_axes[row].append(ax)
    # Draw row 3/4 from file paths (already cropped/rotated externally)
    for row, paths in [(2, diff_paths), (3, ref_paths)]:
        for ci, meta in enumerate(columns, start=0):
            idx = meta["index"]; ax = fig.add_subplot(gs[row, ci + 1])
            img = None
            if 0 <= idx < len(paths) and paths[idx] and paths[idx].exists():
                path = paths[idx]
                if path.suffix.lower() == ".npz":
                    try:
                        img = np.load(path)["frame"]
                    except Exception:
                        img = None
                if img is None and path.suffix.lower() != ".npz":
                    try:
                        img = plt.imread(path)
                    except Exception:
                        img = None
                if img is not None:
                    if ext_crop is not None and img.ndim >= 2:
                        y0, y1, x0, x1 = ext_crop
                        img = img[y0:y1, x0:x1]
                    if flip_row34:
                        img = np.flipud(img)
            if img is None:
                img = np.zeros((10, 10)) if row == 2 else np.zeros((10, 10, 3))
            if row == 2:
                # Diff row: always map to a scalar before applying colormap
                if img.ndim == 3 and img.shape[2] >= 3:
                    # Convert to luminance so cmap applies consistently
                    img_scalar = (0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]).astype(np.float32)
                else:
                    img_scalar = img.astype(np.float32)
                if diff_global_den is not None:
                    img_scalar = img_scalar / diff_global_den
                ax.imshow(img_scalar, origin="lower", aspect=image_aspect34, cmap=comp_cmap if single_colorbar else row34_cmap_name or "gray", norm=shared_norm)
            else:
                if img.ndim == 3 and img.shape[2] >= 3:
                    lum = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]
                else:
                    lum = img.astype(np.float32)
                if single_colorbar:
                    ax.imshow(lum, origin="lower", aspect=image_aspect34, cmap=comp_cmap, norm=shared_norm)
                else:
                    ax.imshow(lum, origin="lower", aspect=image_aspect34, cmap=row34_cmap_name or "gray")
            ax.axis("off")
            row_axes[row].append(ax)

    # Gradient bar with wavelength ticks under the bar
    # Optional dashed box around rows 2 and 3 (Comp. and Diff.)
    for boxed_rows in ([1, 2],):
        axes_list = []
        for r in boxed_rows:
            axes_list.extend(row_axes[r])
        if not axes_list:
            continue
        from matplotlib.transforms import Bbox
        bbox = Bbox.union([ax.get_position() for ax in axes_list])
        x0, y0, x1, y1 = bbox.x0, bbox.y0 + 0.09, bbox.x1, bbox.y1 + 0.045  # adjust box position/height
        pad = 0.002
        rect = mpatches.Rectangle(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            transform=fig.transFigure,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            edgecolor="black",
        )
        fig.add_artist(rect)

    ax_bar_ref = None
    # Determine bar wavelength range: prefer override, else from selected columns' nm, else guess from ref paths
    wl_min = bar_wl_min
    wl_max = bar_wl_max
    if wl_min is None or wl_max is None:
        sel_nms: List[float] = []
        for meta in columns:
            nm_v = meta.get("nm") if isinstance(meta, dict) else None
            if nm_v is not None:
                sel_nms.append(float(nm_v))
        if sel_nms:
            wl_min = min(sel_nms) if wl_min is None else wl_min
            wl_max = max(sel_nms) if wl_max is None else wl_max
        else:
            ref_nms: List[float] = []
            for p in ref_paths:
                if p is None:
                    continue
                m = re.search(r"_(\d+(?:\.\d+)?)nm", p.name)
                if m:
                    ref_nms.append(float(m.group(1)))
            if ref_nms:
                wl_min = min(ref_nms) if wl_min is None else wl_min
                wl_max = max(ref_nms) if wl_max is None else wl_max
    if wl_min is not None and wl_max is not None and wl_max > wl_min:
        samples = np.linspace(wl_min, wl_max, 600)
        wl, xbar, ybar, zbar = load_cie_cmf(REPO_ROOT)
        x = np.interp(samples, wl, xbar); y = np.interp(samples, wl, ybar); z = np.interp(samples, wl, zbar)
        XYZ = np.stack([x, y, z], axis=1).astype(np.float32)
        scale = 1.0 / np.maximum(XYZ[:, 1:2], 1e-6)
        rgb = xyz_to_srgb(XYZ * scale)
        bar_img = np.tile(rgb[None, :, :], (int(bar_px), 1, 1))
        # Spectrum bar spans only the data columns (exclude label + colorbar columns)
        ax_bar = fig.add_subplot(gs[4, 1:1 + num_cols])
        ax_bar_ref = ax_bar
        ax_bar.imshow(bar_img, origin="lower", aspect="auto")
        # Configure ticks: one tick per kept column, centered under each column
        total_w = bar_img.shape[1]
        n_cols = max(1, len(columns))
        xticks = [((i + 0.5) / n_cols) * (total_w - 1) for i in range(n_cols)]
        labels = []
        for meta in columns:
            nm_val = None
            if isinstance(meta, dict) and ("nm" in meta):
                nm_val = float(meta["nm"])  # preferred if provided
            else:
                idx = int(meta["index"])  # fallback lookup
                if wavelength_lookup and idx in wavelength_lookup:
                    nm_val = float(wavelength_lookup[idx])
                elif 0 <= idx < len(ref_paths) and ref_paths[idx] is not None:
                    m = re.search(r"_(\d+(?:\.\d+)?)nm", ref_paths[idx].name)
                    if m:
                        nm_val = float(m.group(1))
            labels.append(f"{nm_val:.0f}" if nm_val is not None else "")
        ax_bar.set_xlim(0, total_w)
        ax_bar.set_xticks(xticks)
        ax_bar.set_xticklabels(labels, fontsize=9)
        ax_bar.set_xlabel("Wavelength (nm)", fontsize=10)
        ax_bar.set_yticks([])
        # Remove spines for a clean bar with labels below
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

    # Defer saving until after colorbars are added

    # Enforce requested row gap by shifting rows (after layout)
    try:
        fig.canvas.draw()
        target_gap = max(0.0, rgap)
        for row_idx in range(1, 4):
            prev_axes = row_axes[row_idx - 1]
            cur_axes = row_axes[row_idx]
            if not prev_axes or not cur_axes:
                continue
            prev_bottom = min(ax.get_position().y0 for ax in prev_axes)
            cur_top = max(ax.get_position().y1 for ax in cur_axes)
            desired_top = prev_bottom - target_gap
            delta = desired_top - cur_top
            if abs(delta) > 1e-6:
                for ax in cur_axes:
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 + delta, pos.width, pos.height])
            # move left label with its row
            for idx, lbl_ax in label_axes:
                if idx == row_idx:
                    pos = lbl_ax.get_position()
                    lbl_ax.set_position([pos.x0, pos.y0 + delta, pos.width, pos.height])
        # Align spectrum bar below row 4 with desired gap
        if ax_bar_ref is not None and row_axes[3]:
            row4_bottom = min(ax.get_position().y0 for ax in row_axes[3])
            pos = ax_bar_ref.get_position()
            new_top = row4_bottom - target_gap
            new_height = pos.height
            new_y0 = max(0.0, new_top - new_height)
            ax_bar_ref.set_position([pos.x0, new_y0, pos.width, new_height])
    except Exception:
        pass

    # If requested, add a single tall colorbar spanning rows 1–4
    if single_colorbar and col_bar is not None:
        # Single bar spanning rows 1–4
        cax_all = fig.add_subplot(gs[0:4, col_bar])
        # Prefer shared_norm if available; fall back to comp_norm or raw range
        if shared_norm is not None:
            sm_all = plt.cm.ScalarMappable(norm=shared_norm, cmap=comp_cmap)
        elif comp_norm is not None:
            sm_all = plt.cm.ScalarMappable(norm=comp_norm, cmap=comp_cmap)
        else:
            rvmin = raw_vmin if raw_vmin is not None else -1.0
            rvmax = raw_vmax if raw_vmax is not None else 1.0
            sm_all = plt.cm.ScalarMappable(norm=Normalize(vmin=rvmin, vmax=rvmax), cmap=comp_cmap)
        sm_all.set_array([])
        cb_all = fig.colorbar(sm_all, cax=cax_all, ticks=[-1, -0.5, 0, 0.5, 1])
        cb_all.ax.set_ylabel("", rotation=90)
        cb_all.outline.set_visible(True); cb_all.outline.set_linewidth(0.8)
        cb_all.ax.tick_params(labelsize=8, width=0.6, length=3)
        # Align bar to rows 0–3 bounds
        try:
            all_axes = row_axes[0] + row_axes[1] + row_axes[2] + row_axes[3]
            top = max(ax.get_position().y1 for ax in all_axes) if all_axes else cax_all.get_position().y1
            bot = min(ax.get_position().y0 for ax in all_axes) if all_axes else cax_all.get_position().y0
            pos = cax_all.get_position()
            # Nudge the bar slightly to the right (10% of gap) for a subtle separation
            offset = 0.1 * gap
            cax_all.set_position([pos.x0 + offset, bot, pos.width, max(0.0, top - bot)])
        except Exception:
            pass
    # Else, if requested, add colorbars for rows 1–2 at the far right
    elif add_row12_colorbars and col_bar is not None:
        if row12_shared_cbar:
            # One tall shared bar spanning rows 0–1 using comp_norm
            cax12 = fig.add_subplot(gs[0:2, col_bar])
            if comp_norm is None:
                # Fallback: compute symmetric
                cmins = []
                cmaxs = []
                for idx in [int(m["index"]) for m in columns]:
                    if 0 <= idx < len(compensated):
                        fr = compensated[idx]
                        if flip_row12:
                            fr = np.flipud(fr)
                        if sens_crop is not None:
                            y0, y1, x0, x1 = sens_crop; fr = fr[y0:y1, x0:x1]
                        cmins.append(float(np.nanmin(fr)))
                        cmaxs.append(float(np.nanmax(fr)))
                comp_abs_fb = max(abs(min(cmins) if cmins else -1.0), abs(max(cmaxs) if cmaxs else 1.0))
                from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
                comp_norm_final = _TwoSlopeNorm(vmin=-comp_abs_fb, vcenter=0.0, vmax=comp_abs_fb)
            else:
                comp_norm_final = comp_norm
            sm12 = plt.cm.ScalarMappable(norm=comp_norm_final, cmap=comp_cmap)
            sm12.set_array([])
            cb12 = fig.colorbar(sm12, cax=cax12)
            cb12.ax.set_ylabel("", rotation=90)
            cb12.outline.set_visible(True); cb12.outline.set_linewidth(0.8)
            cb12.ax.tick_params(labelsize=8, width=0.6, length=3)
            # Align colorbar to rows 0–1 image bounds
            try:
                top01 = max(ax.get_position().y1 for ax in row_axes[0]) if row_axes[0] else cax12.get_position().y1
                bot01 = min(ax.get_position().y0 for ax in row_axes[1]) if row_axes[1] else cax12.get_position().y0
                pos = cax12.get_position()
                cax12.set_position([pos.x0, bot01, pos.width, max(0.0, top01 - bot01)])
            except Exception:
                pass
        else:
            # Legacy: two stacked bars (Raw and Comp.)
            cax1 = fig.add_subplot(gs[0, col_bar])
            rvmin = raw_vmin if raw_vmin is not None else 0.0
            rvmax = raw_vmax if raw_vmax is not None else 1.0
            sm1 = plt.cm.ScalarMappable(norm=Normalize(vmin=rvmin, vmax=rvmax), cmap=raw_cmap)
            sm1.set_array([])
            cb1 = fig.colorbar(sm1, cax=cax1)
            cb1.ax.set_ylabel("Raw counts", rotation=90)
            cb1.outline.set_visible(True); cb1.outline.set_linewidth(0.8)
            cb1.ax.tick_params(labelsize=8, width=0.6, length=3)
            cax2 = fig.add_subplot(gs[1, col_bar])
            sm2 = plt.cm.ScalarMappable(norm=(comp_norm if comp_norm is not None else Normalize(vmin=-1.0, vmax=1.0)), cmap=comp_cmap)
            sm2.set_array([])
            cb2 = fig.colorbar(sm2, cax=cax2)
            cb2.ax.set_ylabel("", rotation=90)
            cb2.outline.set_visible(True); cb2.outline.set_linewidth(0.8)
            cb2.ax.tick_params(labelsize=8, width=0.6, length=3)

    # Optional colorbar for rows 3–4 (external images): use intensity scale
    if (not single_colorbar) and add_row34_colorbar and col_bar is not None:
        # Compute global intensity min/max across selected external images
        emin, emax = None, None
        for idx in [int(m["index"]) for m in columns]:
            for paths in (diff_paths, ref_paths):
                if 0 <= idx < len(paths) and paths[idx] and paths[idx].exists():
                    img = plt.imread(paths[idx])
                    if ext_crop is not None and img.ndim >= 2:
                        y0, y1, x0, x1 = ext_crop; img = img[y0:y1, x0:x1]
                    # Convert to intensity
                    if img.ndim == 3 and img.shape[2] >= 3:
                        # luminance approximation
                        lum = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]
                    else:
                        lum = img.astype(np.float32)
                    vmin = float(np.nanmin(lum)); vmax = float(np.nanmax(lum))
                    emin = vmin if (emin is None or vmin < emin) else emin
                    emax = vmax if (emax is None or vmax > emax) else emax
        if emin is None or emax is None or np.isclose(emin, emax):
            emin, emax = 0.0, 1.0
        ext_cmap = plt.get_cmap(row34_cmap_name)
        # Single tall external bar spanning rows 2–3
        cax34 = fig.add_subplot(gs[2:4, col_bar])
        sm34 = plt.cm.ScalarMappable(norm=Normalize(vmin=emin, vmax=emax), cmap=ext_cmap)
        sm34.set_array([])
        cb34 = fig.colorbar(sm34, cax=cax34)
        cb34.ax.set_ylabel("", rotation=90)
        cb34.outline.set_visible(True); cb34.outline.set_linewidth(0.8)
        cb34.ax.tick_params(labelsize=8, width=0.6, length=3)
        # Align colorbar to rows 2–3 image bounds
        try:
            top23 = max(ax.get_position().y1 for ax in row_axes[2]) if row_axes[2] else cax34.get_position().y1
            bot23 = min(ax.get_position().y0 for ax in row_axes[3]) if row_axes[3] else cax34.get_position().y0
            pos = cax34.get_position()
            cax34.set_position([pos.x0, bot23, pos.width, max(0.0, top23 - bot23)])
        except Exception:
            pass

    # Save used/selected frames similar to cropped pipeline
    def save_frame_png(path: Path, data: np.ndarray, cmap: Colormap, vmin: float | None = None, vmax: float | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = data
        if vmin is None: vmin = float(arr.min())
        if vmax is None: vmax = float(arr.max())
        if np.isclose(vmin, vmax): vmax = vmin + 1e-3
        normed = (arr - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)
        rgba = cmap(normed)
        plt.imsave(path, rgba, origin="lower")

    if override_columns is not None and len(override_columns) > 0:
        kept_idx = [int(m["index"]) for m in columns]
        kept_nm  = [float(m.get("nm", np.nan)) for m in columns]
    else:
        kept_idx = [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin]
        kept_idx = kept_idx[:: max(1, int(downsample_rate))] or kept_idx
        kept_nm = [float(wavelength_lookup.get(idx)) if wavelength_lookup is not None and idx in wavelength_lookup else np.nan for idx in kept_idx]
        if kept_idx and kept_idx[-1] != ( [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin ][-1] ):
            kept_idx.append( [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin ][-1] )
            kept_nm.append(float(wavelength_lookup.get(kept_idx[-1])) if wavelength_lookup is not None and kept_idx[-1] in wavelength_lookup else np.nan)

    orig_used_dir = output_dir / "orig_used_frames"; orig_used_dir.mkdir(parents=True, exist_ok=True)
    comp_used_dir = output_dir / "comp_used_frames"; comp_used_dir.mkdir(parents=True, exist_ok=True)
    diff_used_dir = output_dir / "diff_used_frames"; diff_used_dir.mkdir(parents=True, exist_ok=True)
    ref_used_dir  = output_dir / "ref_used_frames";  ref_used_dir.mkdir(parents=True, exist_ok=True)
    diff_sel_dir  = output_dir / "diff_selected_frames"; diff_sel_dir.mkdir(parents=True, exist_ok=True)
    gt_sel_dir    = output_dir / "gt_selected_frames";   gt_sel_dir.mkdir(parents=True, exist_ok=True)
    ref_sel_dir   = output_dir / "ref_selected_frames";  ref_sel_dir.mkdir(parents=True, exist_ok=True)

    # Save downsampled frames with names bin_XX_YYYnm.png
    for i, idx in enumerate(kept_idx):
        nm_label = None
        if override_columns is not None and i < len(kept_nm) and not np.isnan(kept_nm[i]):
            nm_label = float(kept_nm[i])
        elif wavelength_lookup and idx in wavelength_lookup:
            nm_label = float(wavelength_lookup[idx])
        fname = f"bin_{idx:02d}" + (f"_{nm_label:.0f}nm" if nm_label is not None else "") + ".png"
        # Sensor rows
        f_raw = originals[idx] if idx < len(originals) else np.zeros_like(originals[0])
        f_comp = compensated[idx] if idx < len(compensated) else np.zeros_like(compensated[0])
        if flip_row12:
            f_raw = np.flipud(f_raw); f_comp = np.flipud(f_comp)
        if sens_crop is not None:
            y0, y1, x0, x1 = sens_crop
            f_raw = f_raw[y0:y1, x0:x1]; f_comp = f_comp[y0:y1, x0:x1]
        save_frame_png(orig_used_dir / fname, f_raw, raw_cmap)
        save_frame_png(comp_used_dir / fname, f_comp, comp_cmap)
        # External rows: copy/crop matching images by index if available
        if 0 <= idx < len(diff_paths) and diff_paths[idx] and diff_paths[idx].exists():
            p = diff_paths[idx]
            if p.suffix.lower() == ".npz":
                try:
                    img = np.load(p)["frame"]
                except Exception:
                    img = None
            else:
                img = plt.imread(p)
            if img is not None:
                if ext_crop is not None and img.ndim >= 2:
                    y0, y1, x0, x1 = ext_crop; img = img[y0:y1, x0:x1]
                if flip_row34: img = np.flipud(img)
                plt.imsave(diff_used_dir / fname, img, origin="lower")
                # Also record as selected
                plt.imsave(diff_sel_dir / fname, img, origin="lower")
        if 0 <= idx < len(ref_paths) and ref_paths[idx] and ref_paths[idx].exists():
            img = plt.imread(ref_paths[idx])
            if ext_crop is not None:
                y0, y1, x0, x1 = ext_crop; img = img[y0:y1, x0:x1]
            if flip_row34: img = np.flipud(img)
            plt.imsave(ref_used_dir / fname, img, origin="lower")
            # Selected copies
            plt.imsave(ref_sel_dir / fname, img, origin="lower")
            plt.imsave(gt_sel_dir / fname, img, origin="lower")

    # Now save the main figure with all colorbars included
    stem = Path(figure_name).stem if figure_name else "spectral_reconstruction_scan"
    out_stem = output_dir / stem
    fig.savefig(f"{out_stem}.pdf", dpi=400, bbox_inches="tight", pad_inches=0.0)
    if save_png:
        fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    segment_path = args.segment.resolve()
    figures_root = Path(__file__).resolve().parent / "figures"
    base_name = Path(args.figure_name).stem if args.figure_name else "spectral_reconstruction_scan"
    out_dir = (figures_root / f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
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

    # Build fine 5 ms series and neg scale
    neg_scale, series, comp_weights = compute_series_and_weights(
        x.astype(np.float32), y.astype(np.float32), t.astype(np.float32), p.astype(np.float32),
        a_params, b_params, sensor_area, args.pos_scale, args.neg_scale, args.fine_step_us,
    )
    save_series(series, out_dir)

    # Edges-only alignment and overlays saved into out_dir
    payload = align_edges_only(series, args.gt_dir.resolve(), args.edge_quantile, out_dir, neg_scale, args.fine_step_us, args.save_png)

    # Build 50 ms frames (Orig./Comp.)
    t_min = float(np.min(t)); t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / args.bin_width_us))
    originals: List[np.ndarray] = []
    compensated: List[np.ndarray] = []
    metadata_bins: List[Dict[str, float]] = []
    t_comp = t - ((float(np.mean(a_params)) * x + float(np.mean(b_params)) * y) if (a_params is not None and b_params is not None) else 0.0)
    weights = np.where(p >= 0, args.pos_scale, -neg_scale).astype(np.float32)
    raw_weights = p.astype(np.float32)  # preserve polarity in the raw branch
    for k in range(num_bins):
        s = t_min + k * args.bin_width_us; e = min(t_min + (k + 1) * args.bin_width_us, t_max)
        mask = (t >= s) & (t < e); comp_mask = (t_comp >= s) & (t_comp < e)
        frame_raw = accumulate_bin(x, y, mask, raw_weights, sensor_shape)
        frame_comp = accumulate_bin(x, y, comp_mask, weights, sensor_shape)
        if args.smooth:
            frame_comp = smooth_volume_3d(np.stack([frame_comp], axis=0), 3)[0]
        frame_comp = subtract_background(frame_comp)
        originals.append(frame_raw); compensated.append(frame_comp)
        metadata_bins.append({"index": k, "start_us": s, "end_us": e, "duration_ms": (e - s) / 1000.0})

    # Prepare external rows (Diff/Ref) paths;
    # Build wavelength-indexed selectors for external rows
    def parse_ref_nm(path: Path) -> Optional[float]:
        m = re.search(r"_(\d+(?:\.\d+)?)nm", path.name, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def parse_grad_range(path: Path) -> Optional[Tuple[float, float]]:
        # Patterns like: grad_bin_001_400.0to420.png or *_400to420.png
        m = re.search(r"_(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", path.stem, re.IGNORECASE)
        if not m:
            return None
        a = float(m.group(1)); b = float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return (lo, hi)

    def choose_ref_for_nm(folder: Path, nm: float) -> Optional[Path]:
        best: Tuple[float, Path] | None = None
        for p in folder.glob('*.png'):
            nm_p = parse_ref_nm(p)
            if nm_p is None:
                continue
            d = abs(nm_p - nm)
            if best is None or d < best[0]:
                best = (d, p)
        return best[1] if best else None

    def choose_grad_for_nm(folder: Path, nm: float) -> Optional[Path]:
        best: Tuple[float, Path] | None = None
        for p in folder.glob('*.png'):
            rng = parse_grad_range(p)
            if rng is None:
                continue
            lo, hi = rng
            if lo <= nm <= hi:
                # Exact containment wins
                return p
            centre = 0.5 * (lo + hi)
            d = abs(centre - nm)
            if best is None or d < best[0]:
                best = (d, p)
        return best[1] if best else None

    # Prepare crop boxes and wavelength lookup for labels
    sens_crop = _load_crop_box(args.crop_json, preferred_key="ref_crop")
    ext_crop = _load_crop_box(args.external_crop_json, preferred_key="template_crop")
    # Compute wavelength lookup per bin using edges-only slope/intercept
    slope = float(payload["alignment"]["slope_nm_per_ms"])  # type: ignore
    intercept = float(payload["alignment"]["intercept_nm"])  # type: ignore
    base_start = metadata_bins[0]["start_us"] if metadata_bins else 0.0
    wavelength_lookup: Dict[int, float] = {}
    for item in metadata_bins:
        centre_ms = (((item["start_us"] + item["end_us"]) * 0.5) - base_start) / 1000.0
        wavelength_lookup[int(item["index"])] = float(slope * centre_ms + intercept)

    # Rebuild external path arrays using wavelength lookup
    diff_pngs: List[Optional[Path]] = [None] * num_bins
    ref_pngs: List[Optional[Path]] = [None] * num_bins
    diff_dir = args.diff_frames_dir.resolve(); ref_dir = args.ref_frames_dir.resolve()
    def parse_ref_nm(path: Path) -> Optional[float]:
        m = re.search(r"_(\d+(?:\.\d+)?)nm", path.name, re.IGNORECASE)
        return float(m.group(1)) if m else None
    def parse_grad_range(path: Path) -> Optional[Tuple[float, float]]:
        m = re.search(r"_(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", path.stem, re.IGNORECASE)
        if not m: return None
        a = float(m.group(1)); b = float(m.group(2)); lo, hi = (a,b) if a<=b else (b,a); return (lo,hi)
    def choose_ref_for_nm(nm: float) -> Optional[Path]:
        best: Tuple[float, Path] | None = None
        for p in ref_dir.glob('*.png'):
            nm_p = parse_ref_nm(p)
            if nm_p is None: continue
            d = abs(nm_p - nm)
            if best is None or d < best[0]: best = (d, p)
        return best[1] if best else None
    def choose_grad_for_nm(nm: float) -> Optional[Path]:
        best: Tuple[float, Path] | None = None
        # Prefer NPZ (signed) if available, else fall back to PNG
        for ext in ("*.npz", "*.png"):
            for p in diff_dir.glob(ext):
                rng = parse_grad_range(p)
                if rng is None:
                    continue
                lo, hi = rng
                if lo <= nm <= hi:
                    return p
                centre = 0.5 * (lo + hi)
                d = abs(centre - nm)
                if best is None or d < best[0]:
                    best = (d, p)
        return best[1] if best else None
    for it in metadata_bins:
        idx = int(it['index']); nm = wavelength_lookup.get(idx)
        if nm is None: continue
        ref_pngs[idx] = choose_ref_for_nm(nm)
        diff_pngs[idx] = choose_grad_for_nm(nm)

    # If wavelength-based column selection is requested, build the target columns now
    override_columns: List[Dict[str, float]] = []
    if args.wl_list or (args.wl_min is not None and args.wl_max is not None and args.wl_step is not None):
        # Build a list of target wavelengths
        targets: List[float] = []
        if args.wl_list:
            try:
                targets = [float(s.strip()) for s in str(args.wl_list).split(',') if s.strip()]
            except Exception:
                targets = []
        else:
            # Inclusive range using wl_min..wl_max with the given step
            cur = float(args.wl_min)
            while cur <= float(args.wl_max) + 1e-6:
                targets.append(cur)
                cur += float(args.wl_step)
        # Map every bin to its wavelength
        idx_to_nm = {int(it['index']): float(wavelength_lookup.get(int(it['index']))) for it in metadata_bins if wavelength_lookup.get(int(it['index'])) is not None}
        # For each target, find the nearest bin index (deduplicate)
        used_idx: set[int] = set()
        for nm in targets:
            best_idx = None
            best_d = None
            for idx, nm_val in idx_to_nm.items():
                d = abs(nm_val - nm)
                if (best_d is None) or (d < best_d):
                    best_d = d; best_idx = idx
            if best_idx is not None and best_idx not in used_idx:
                used_idx.add(best_idx)
                override_columns.append({"index": best_idx, "nm": float(idx_to_nm[best_idx])})
        # Sort by wavelength ascending
        override_columns.sort(key=lambda m: m.get("nm", 0.0))

    # Render spectral grid (cropped rows and flips applied here)
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", COMP_LIGHTEN_FRACTION)
    render_spectral_grid(
        originals=originals,
        compensated=compensated,
        diff_paths=diff_pngs,
        ref_paths=ref_pngs,
        metadata=metadata_bins,
        start_bin=args.start_bin,
        end_bin=args.end_bin,
        downsample_rate=args.downsample_rate,
        raw_cmap=raw_cmap,
        comp_cmap=comp_cmap,
        output_dir=out_dir,
        figure_name=base_name,
        save_png=bool(args.save_png),
        bar_px=int(args.bar_px),
        flip_row12=bool(args.flip_row12),
        flip_row34=bool(args.flip_row34),
        ext_crop=ext_crop,
        sens_crop=sens_crop,
        wavelength_lookup=wavelength_lookup,
        col_gap=float(args.col_gap),
        row_gap=(None if args.row_gap is None else float(args.row_gap)),
        override_columns=(override_columns if len(override_columns) > 0 else None),
        bar_wl_min=(float(args.bar_wl_min) if args.bar_wl_min is not None else None),
        bar_wl_max=(float(args.bar_wl_max) if args.bar_wl_max is not None else None),
        image_aspect12=str(args.image_aspect12),
        image_aspect34=str(args.image_aspect34),
        unify_row12_scales=bool(getattr(args, 'unified_row12_scales', False)),
        raw_global_vmin=(float(args.raw_global_vmin) if args.raw_global_vmin is not None else None),
        raw_global_vmax=(float(args.raw_global_vmax) if args.raw_global_vmax is not None else None),
        comp_global_abs=(float(args.comp_global_abs) if args.comp_global_abs is not None else None),
        add_row12_colorbars=(not bool(args.no_row12_colorbars)),
        cbar_ratio=float(args.cbar_ratio),
        add_row34_colorbar=bool(getattr(args, 'row34_colorbar', False)),
        row34_cmap_name=str(getattr(args, 'row34_cmap', 'gray')),
        single_colorbar=bool(getattr(args, 'single_colorbar', False)),
        column_step=max(1, int(getattr(args, 'column_step', 1))),
        row12_shared_cbar=bool(getattr(args, 'row12_shared_cbar', False)),
    )

    # Save weights summary (mirrors all-in-one)
    weights_json = out_dir / "figure04_rescaled_weights.json"
    with weights_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "segment": str(segment_path),
                "pos_scale": args.pos_scale,
                "neg_scale": float(payload.get("neg_scale", neg_scale) if isinstance(payload, dict) else neg_scale),
                "bin_width_us": args.bin_width_us,
                "bin_width_ms": args.bin_width_us / 1000.0,
                "num_bins": num_bins,
                "smooth": bool(args.smooth),
                "bin_times_us": metadata_bins,
                "background_means": [float(np.mean(f)) for f in compensated],
                "rescale_step_us": args.fine_step_us,
                "bg_series_npz": "figure04_rescaled_bg_series.npz",
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()
