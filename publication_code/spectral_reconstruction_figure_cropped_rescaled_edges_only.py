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


def _load_crop_box(path: Path | None) -> Tuple[int, int, int, int] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    cand = None
    for key in ("bbox", "ref_crop", "template_crop", "bbox_xyxy"):
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
) -> None:
    setup_style()
    selected = [m for m in metadata if start_bin <= m["index"] <= end_bin]
    kept_indices = set(_downsample_indices(start_bin, end_bin, downsample_rate))
    columns = [m for m in selected if m["index"] in kept_indices]
    num_cols = len(columns)
    # Build rows
    fig = plt.figure(figsize=(1.2 * num_cols + 0.6, 5.2))
    gs = fig.add_gridspec(5, num_cols + 1, wspace=0.06, hspace=0.08, width_ratios=[0.22] + [1] * num_cols,
                          height_ratios=[1.0, 1.0, 1.0, 1.0, max(0.02, float(bar_px) / 80.0)])
    def label_column(r: int, text: str) -> None:
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off"); ax.text(0.5, 0.5, text, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    label_column(0, "Original"); label_column(1, "Compensated"); label_column(2, "Diff."); label_column(3, "Reference")

    # Draw row 1/2
    for row, frames, cmap in [(0, originals, raw_cmap), (1, compensated, comp_cmap)]:
        for ci, meta in enumerate(columns, start=0):
            idx = meta["index"]
            ax = fig.add_subplot(gs[row, ci + 1])
            frame = frames[idx] if idx < len(frames) else np.zeros_like(frames[0])
            ax.imshow(frame, cmap=cmap, origin="lower")
            ax.axis("off")
    # Draw row 3/4 from file paths (already cropped/rotated externally)
    for row, paths in [(2, diff_paths), (3, ref_paths)]:
        for ci, meta in enumerate(columns, start=0):
            idx = meta["index"]; ax = fig.add_subplot(gs[row, ci + 1])
            if 0 <= idx < len(paths) and paths[idx] and paths[idx].exists():
                img = plt.imread(paths[idx])
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
    def list_pngs_sorted(folder: Path) -> List[Path]:
        items = sorted(folder.glob('*.png'))
        return items
    diff_pngs = list_pngs_sorted(args.diff_frames_dir.resolve())
    ref_pngs = list_pngs_sorted(args.ref_frames_dir.resolve())

    # Render spectral grid (cropped rows assumed preprocessed by caller)
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", COMP_LIGHTEN_FRACTION)
    render_spectral_grid(
        originals, compensated, diff_pngs, ref_pngs, metadata_bins, args.start_bin, args.end_bin, args.downsample_rate,
        raw_cmap, comp_cmap, out_dir, base_name, bool(args.save_png), int(args.bar_px),
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

