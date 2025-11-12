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
) -> None:
    setup_style()
    selected = [m for m in metadata if start_bin <= m["index"] <= end_bin]
    kept_indices = set(_downsample_indices(start_bin, end_bin, downsample_rate))
    columns = [m for m in selected if m["index"] in kept_indices]
    num_cols = len(columns)
    # Build rows
    fig = plt.figure(figsize=(1.2 * num_cols + 0.6, 5.2))
    gap = 0.045  # make row and column gaps identical
    gs = fig.add_gridspec(
        5,
        num_cols + 1,
        wspace=gap,
        hspace=gap,
        width_ratios=[0.22] + [1] * num_cols,
        height_ratios=[1.0, 1.0, 1.0, 1.0, max(0.02, float(bar_px) / 80.0)],
    )
    def label_column(r: int, text: str) -> None:
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off"); ax.text(0.5, 0.5, text, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    label_column(0, "Original"); label_column(1, "Comp."); label_column(2, "Diff."); label_column(3, "Reference")

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
            ax.imshow(frame, cmap=cmap, origin="lower")
            ax.axis("off")
    # Draw row 3/4 from file paths (already cropped/rotated externally)
    for row, paths in [(2, diff_paths), (3, ref_paths)]:
        for ci, meta in enumerate(columns, start=0):
            idx = meta["index"]; ax = fig.add_subplot(gs[row, ci + 1])
            if 0 <= idx < len(paths) and paths[idx] and paths[idx].exists():
                img = plt.imread(paths[idx])
                if ext_crop is not None and img.ndim >= 2:
                    y0, y1, x0, x1 = ext_crop
                    img = img[y0:y1, x0:x1]
                if flip_row34:
                    img = np.flipud(img)
                ax.imshow(img, origin="lower")
            else:
                ax.imshow(np.zeros((10, 10)), origin="lower", cmap="gray")
            ax.axis("off")

    # Gradient bar using GT reference wavelengths from filenames if available
    ref_nms: List[float] = []
    for p in ref_paths:
        if p is None: continue
        m = re.search(r"_(\d+(?:\.\d+)?)nm", p.name)
        if m:
            ref_nms.append(float(m.group(1)))
    if ref_nms:
        wl_min, wl_max = min(ref_nms), max(ref_nms)
        samples = np.linspace(wl_min, wl_max, 600)
        wl, xbar, ybar, zbar = load_cie_cmf(REPO_ROOT)
        x = np.interp(samples, wl, xbar); y = np.interp(samples, wl, ybar); z = np.interp(samples, wl, zbar)
        XYZ = np.stack([x, y, z], axis=1).astype(np.float32)
        scale = 1.0 / np.maximum(XYZ[:, 1:2], 1e-6)
        rgb = xyz_to_srgb(XYZ * scale)
        bar_img = np.tile(rgb[None, :, :], (int(bar_px), 1, 1))
        ax_bar = fig.add_subplot(gs[4, 1:]); ax_bar.imshow(bar_img, origin="lower", aspect="auto"); ax_bar.axis("off")

    stem = Path(figure_name).stem if figure_name else "spectral_reconstruction_scan"
    out_stem = output_dir / stem
    fig.savefig(f"{out_stem}.pdf", dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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

    kept_idx = [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin]
    kept_idx = kept_idx[:: max(1, int(downsample_rate))] or kept_idx
    if kept_idx and kept_idx[-1] != ( [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin ][-1] ):
        kept_idx.append( [m["index"] for m in metadata if start_bin <= m["index"] <= end_bin ][-1] )

    orig_used_dir = output_dir / "orig_used_frames"; orig_used_dir.mkdir(parents=True, exist_ok=True)
    comp_used_dir = output_dir / "comp_used_frames"; comp_used_dir.mkdir(parents=True, exist_ok=True)
    diff_used_dir = output_dir / "diff_used_frames"; diff_used_dir.mkdir(parents=True, exist_ok=True)
    ref_used_dir  = output_dir / "ref_used_frames";  ref_used_dir.mkdir(parents=True, exist_ok=True)
    diff_sel_dir  = output_dir / "diff_selected_frames"; diff_sel_dir.mkdir(parents=True, exist_ok=True)
    gt_sel_dir    = output_dir / "gt_selected_frames";   gt_sel_dir.mkdir(parents=True, exist_ok=True)
    ref_sel_dir   = output_dir / "ref_selected_frames";  ref_sel_dir.mkdir(parents=True, exist_ok=True)

    # Save downsampled frames with names bin_XX_YYYnm.png
    for idx in kept_idx:
        nm_label = None
        if wavelength_lookup and idx in wavelength_lookup:
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
            img = plt.imread(diff_paths[idx])
            if ext_crop is not None:
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
    for k in range(num_bins):
        s = t_min + k * args.bin_width_us; e = min(t_min + (k + 1) * args.bin_width_us, t_max)
        mask = (t >= s) & (t < e); comp_mask = (t_comp >= s) & (t_comp < e)
        frame_raw = accumulate_bin(x, y, mask, np.ones_like(weights), sensor_shape)
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

    # Build per-bin external paths based on wavelength mapping
    diff_pngs: List[Optional[Path]] = [None] * num_bins
    ref_pngs: List[Optional[Path]] = [None] * num_bins
    diff_dir = args.diff_frames_dir.resolve()
    ref_dir = args.ref_frames_dir.resolve()
    for it in metadata_bins:
        idx = int(it['index'])
        nm = wavelength_lookup.get(idx) if 'wavelength_lookup' in locals() else None
        if nm is None:
            continue
        ref_pngs[idx] = choose_ref_for_nm(ref_dir, nm)
        diff_pngs[idx] = choose_grad_for_nm(diff_dir, nm)

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
        for p in diff_dir.glob('*.png'):
            rng = parse_grad_range(p)
            if rng is None: continue
            lo, hi = rng
            if lo <= nm <= hi: return p
            centre = 0.5 * (lo + hi); d = abs(centre - nm)
            if best is None or d < best[0]: best = (d, p)
        return best[1] if best else None
    for it in metadata_bins:
        idx = int(it['index']); nm = wavelength_lookup.get(idx)
        if nm is None: continue
        ref_pngs[idx] = choose_ref_for_nm(nm)
        diff_pngs[idx] = choose_grad_for_nm(nm)

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
