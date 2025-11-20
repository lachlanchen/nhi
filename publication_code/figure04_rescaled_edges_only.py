#!/usr/bin/env python3
"""
Figure 4 (edges-only): third-only overlay focusing on start/end match.

This variant fits a strict linear time→wavelength map using only the
visible-band edges (rising and falling) and ignores the interior shape.
It produces the minimal overlay (BG mapped vs a single GT curve) and
annotates dashed lines at the inferred band edges.

Outputs are written to a timestamped folder under
`publication_code/figures/figure04_edges_only_YYYYMMDD_HHMMSS/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure04_rescaled import (  # type: ignore
    load_segment_events,
    load_params,
    auto_scale_neg_weight,
    setup_style,
    prepare_colormap,
    accumulate_bin,
    subtract_background,
    smooth_volume_3d,
    find_param_file,
    compute_fast_comp_times,
)
from publication_code.figure04_rescaled_allinone import render_grid  # type: ignore
from groundtruth_spectrum.compare_reconstruction_to_gt import (  # type: ignore
    moving_average,
    normalise_curve,
    detect_active_region,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Figure 4 edges-only overlay (third-only plot)")
    ap.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_events.npz")
    ap.add_argument("--gt-dir", type=Path, default=Path("groundtruth_spectrum_2835"))
    ap.add_argument("--bin-width-us", type=float, default=50000.0)
    ap.add_argument("--fine-step-us", type=float, default=5000.0)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--neg-scale", type=float, default=1.5)
    ap.add_argument("--edge-quantile", type=float, default=0.05, help="Quantile for edge detection (e.g., 0.05)")
    ap.add_argument("--start-bin", type=int, default=3)
    ap.add_argument("--end-bin", type=int, default=15)
    ap.add_argument("--smooth", action="store_true", default=True)
    ap.add_argument("--raw-colormap", default=None)
    ap.add_argument("--comp-colormap", default=None)
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
    """Reproduce fine (e.g., 5 ms) exponential series with auto neg scaling.
    Returns (neg_scale_final, series_dict, weights_per_event)."""
    if a_params is not None and b_params is not None:
        a_avg = float(np.mean(a_params))
        b_avg = float(np.mean(b_params))
        t_comp = t - (a_avg * x + b_avg * y)
    else:
        t_comp = t
    neg_scale, series = auto_scale_neg_weight(
        t_comp.astype(np.float32),
        p.astype(np.float32),
        sensor_area=sensor_area,
        step_us=step_us,
        pos_scale=pos_scale,
        neg_scale_init=neg_init,
        return_series=True,
    )
    weights = np.where(p >= 0, pos_scale, -neg_scale).astype(np.float32)
    return float(neg_scale), series, weights


def quantile_edge_first(x: np.ndarray, y: np.ndarray, q: float) -> float:
    y = np.clip(y, 0.0, 1.0)
    idx = int(np.argmax(y >= q))
    if idx == 0:
        return float(x[0])
    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if y1 == y0:
        return float(x1)
    t = (q - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def quantile_edge_last(x: np.ndarray, y: np.ndarray, q: float) -> float:
    y = np.clip(y, 0.0, 1.0)
    rev = y[::-1]
    ridx = int(np.argmax(rev >= q))
    idx = len(y) - 1 - ridx
    if idx <= 0:
        return float(x[-1])
    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if y1 == y0:
        return float(x1)
    t = (q - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def align_edges_only(
    series: Dict[str, np.ndarray],
    gt_dir: Path,
    edge_q: float,
    out_dir: Path,
    neg_scale: float,
    step_us: float,
    save_png: bool,
) -> Dict[str, object]:
    time_ms = np.asarray(series["time_ms"], dtype=np.float32)
    exp_rescaled = np.asarray(series["exp_rescaled"], dtype=np.float32)

    # Load two GT curves
    gt_files = sorted(gt_dir.glob("*.txt"))
    gt_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for txt in gt_files:
        try:
            wl, val = np.loadtxt(txt, skiprows=17, usecols=[0, 1], unpack=True)
        except Exception:
            # Fallback to parser in helpers
            from groundtruth_spectrum.compare_reconstruction_to_gt import load_ground_truth  # type: ignore
            wl, val = load_ground_truth(txt)
        if wl.size >= 50 and val.size == wl.size:
            gt_curves.append((txt.stem, wl.astype(np.float32), val.astype(np.float32)))
    if not gt_curves:
        raise FileNotFoundError(f"No spectrometer TXT files in {gt_dir}")

    # BG edges
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    bg_norm = np.clip(normalise_curve(exp_rescaled, region_bg), 0.0, 1.0)
    t_low = quantile_edge_first(time_ms, bg_norm, edge_q)
    t_high = quantile_edge_last(time_ms, bg_norm, edge_q)

    # GT edges (median across curves after normalisation)
    wl_lows: List[float] = []
    wl_highs: List[float] = []
    wl_span_min = +np.inf
    wl_span_max = -np.inf
    for _, wl, val in gt_curves:
        wl_span_min = min(wl_span_min, float(np.min(wl)))
        wl_span_max = max(wl_span_max, float(np.max(wl)))
        smooth_gt = moving_average(val, max(21, len(val) // 300))
        region_gt = detect_active_region(smooth_gt)
        gt_norm = np.clip(normalise_curve(smooth_gt, region_gt), 0.0, 1.0)
        wl_lows.append(quantile_edge_first(wl, gt_norm, edge_q))
        wl_highs.append(quantile_edge_last(wl, gt_norm, edge_q))
    wl_low = float(np.median(wl_lows))
    wl_high = float(np.median(wl_highs))

    # Two-point fit (edges only)
    if t_high == t_low:
        slope, intercept = (wl_high - wl_low) / 1.0, wl_low - ((wl_high - wl_low) / 1.0) * t_low
    else:
        slope = (wl_high - wl_low) / (t_high - t_low)
        intercept = wl_low - slope * t_low

    wl_series = slope * time_ms + intercept

    # Plot background vs a single GT (normalised) with dashed edge lines
    name0, wl0, val0 = gt_curves[0]
    smooth0 = moving_average(val0, max(21, len(val0) // 300))
    region0 = detect_active_region(smooth0)
    gt0_norm = np.clip(normalise_curve(smooth0, region0), 0.0, 1.0)

    mask = (wl_series >= wl_span_min) & (wl_series <= wl_span_max)
    wl_bg = wl_series[mask]
    bg_norm_wl = bg_norm[mask]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    BG_COLOR = "#1f77b4"; SPD_COLOR = "#ff7f0e"
    ax.plot(wl_bg, bg_norm_wl, color=BG_COLOR, linewidth=1.8, label="Background")
    ax.plot(wl0, gt0_norm, color=SPD_COLOR, linewidth=1.8, label="Light SPD")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Normalised intensity")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.axvline(wl_low, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(wl_high, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    # Save with names matching all-in-one as well as edges-only
    out_plot = out_dir / "figure04_rescaled_bg_gt_third_only.pdf"
    fig.savefig(out_plot, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_plot.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Also save edges-only alias for convenience
    alias = out_dir / "figure04_edges_only_third.pdf"
    try:
        fig2, ax2 = plt.subplots(figsize=(5.6, 3.0))
        ax2.plot(wl_bg, bg_norm_wl, color=BG_COLOR, linewidth=1.8, label="Background")
        ax2.plot(wl0, gt0_norm, color=SPD_COLOR, linewidth=1.8, label="Light SPD")
        ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Normalised intensity")
        ax2.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax2.axvline(wl_low, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.axvline(wl_high, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.9)
        fig2.savefig(alias, dpi=400, bbox_inches="tight")
        if save_png:
            fig2.savefig(alias.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig2)
    except Exception:
        pass

    payload = {
        "series_npz": "figure04_edges_only_series.npz",
        "groundtruth_directory": str(gt_dir),
        "groundtruth_files": [name for name, _, _ in gt_curves],
        "neg_scale": float(neg_scale),
        "rescale_step_us": float(step_us),
        "alignment": {
            "edge_quantile": float(edge_q),
            "slope_nm_per_ms": float(slope),
            "intercept_nm": float(intercept),
            "bg_edges_ms": [float(t_low), float(t_high)],
            "gt_edges_nm": [float(wl_low), float(wl_high)],
        },
        "bg_vs_gt_plot": out_plot.name,
    }
    # Save with all-in-one filename for compatibility
    with (out_dir / "figure04_rescaled_bg_alignment.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return payload


def save_series_npz(series: Dict[str, np.ndarray], out_dir: Path) -> None:
    path = out_dir / "figure04_rescaled_bg_series.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in series.items()})


def plot_three_panel_normalised(series: Dict[str, np.ndarray], gt_curves: List[Tuple[str, np.ndarray, np.ndarray]], wl_series: np.ndarray, out_dir: Path, save_png: bool) -> None:
    time_ms = series["time_ms"].astype(np.float32)
    exp_rescaled = series["exp_rescaled"].astype(np.float32)
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    bg_norm = np.clip(normalise_curve(exp_rescaled, region_bg), 0.0, None)

    gt_norm_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    wl_min = np.inf; wl_max = -np.inf
    for name, wl, val in gt_curves:
        wl_min = min(wl_min, float(np.min(wl)))
        wl_max = max(wl_max, float(np.max(wl)))
        smooth = moving_average(val, max(21, len(val) // 300))
        region_gt = detect_active_region(smooth)
        val_norm = np.clip(normalise_curve(smooth, region_gt), 0.0, None)
        gt_norm_list.append((name, wl.astype(np.float32), val_norm.astype(np.float32)))

    mask = (wl_series >= wl_min) & (wl_series <= wl_max)
    wl_bg = wl_series[mask].astype(np.float32)
    bg_norm_wl = bg_norm[mask]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), constrained_layout=True)
    ax0, ax1, ax2 = axes
    for name, wl, valn in gt_norm_list:
        ax0.plot(wl, valn, linewidth=1.6, label=name)
    ax0.set_title("Ref. (normalised)"); ax0.set_xlabel("Wavelength (nm)"); ax0.set_ylabel("Norm. intensity"); ax0.grid(alpha=0.3, linestyle="--", linewidth=0.6); ax0.legend(loc="best", fontsize=8)
    ax1.plot(time_ms, bg_norm, color="#1f77b4", linewidth=1.6); ax1.set_title("BG (normalised, 5 ms)"); ax1.set_xlabel("Relative time (ms)"); ax1.set_ylabel("Norm. intensity"); ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax2.plot(wl_bg, bg_norm_wl, color="#1f77b4", linewidth=1.8, label="BG mapped")
    for name, wl, valn in gt_norm_list:
        ax2.plot(wl, valn, linewidth=1.4, label=name)
    ax2.set_title("Aligned (normalised)"); ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Norm. intensity"); ax2.grid(alpha=0.3, linestyle="--", linewidth=0.6); ax2.legend(loc="best", fontsize=8)
    out_path = out_dir / "figure04_rescaled_bg_gt_threepanel.pdf"; fig.savefig(out_path, dpi=400, bbox_inches="tight");
    if save_png: fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight");
    plt.close(fig)


def plot_bg_vs_gt(series: Dict[str, np.ndarray], gt_curves: List[Tuple[str, np.ndarray, np.ndarray]], wl_series: np.ndarray, out_dir: Path, save_png: bool) -> None:
    time_ms = np.asarray(series["time_ms"], dtype=np.float32)
    exp_rescaled = np.asarray(series["exp_rescaled"], dtype=np.float32)
    wl_min_all = min(float(np.min(wl)) for _, wl, _ in gt_curves)
    wl_max_all = max(float(np.max(wl)) for _, wl, _ in gt_curves)
    mask_bg = (wl_series >= wl_min_all) & (wl_series <= wl_max_all)
    wl_bg = wl_series[mask_bg]
    smooth_bg = moving_average(exp_rescaled, max(21, int(len(exp_rescaled) // 200) | 1))
    region_bg = detect_active_region(smooth_bg)
    series_norm = normalise_curve(exp_rescaled, region_bg)
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
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Normalised intensity (a.u.)"); ax.set_title("Background vs. Light SPD"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    out_plot = out_dir / "figure04_rescaled_bg_vs_groundtruth.pdf"; fig.savefig(out_plot, dpi=400, bbox_inches="tight");
    if save_png: fig.savefig(out_plot.with_suffix(".png"), dpi=300, bbox_inches="tight");
    plt.close(fig)


def build_bin_frames(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, a_params: np.ndarray | None, b_params: np.ndarray | None, sensor_shape: Tuple[int, int], bin_width_us: float, start_bin: int, end_bin: int, pos_scale: float, neg_scale: float, smooth: bool) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
    h, w = sensor_shape
    t_min = float(np.min(t)); t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / bin_width_us))
    # Compute compensated times if params are available
    if a_params is not None and b_params is not None:
        t_comp = t - (float(np.mean(a_params)) * x + float(np.mean(b_params)) * y)
    else:
        t_comp = t
    weights = np.where(p >= 0, pos_scale, -neg_scale).astype(np.float32)
    originals: List[np.ndarray] = []
    compensated: List[np.ndarray] = []
    meta: List[Dict[str, float]] = []
    for k in range(num_bins):
        s = t_min + k * bin_width_us
        e = min(t_min + (k + 1) * bin_width_us, t_max)
        mask = (t >= s) & (t < e)
        comp_mask = (t_comp >= s) & (t_comp < e)
        frame_raw = accumulate_bin(x, y, mask, np.ones_like(weights), sensor_shape)
        frame_comp = accumulate_bin(x, y, comp_mask, weights, sensor_shape)
        if smooth:
            vol = np.stack([frame_comp], axis=0)
            frame_comp = smooth_volume_3d(vol, 3)[0]
        frame_comp = subtract_background(frame_comp)
        originals.append(frame_raw)
        compensated.append(frame_comp)
        meta.append({"index": k, "start_us": s, "end_us": e, "duration_ms": (e - s) / 1000.0})
    # Keep only requested range when rendering grid; meta still has all entries
    return originals, compensated, meta


def main() -> None:
    args = parse_args()
    setup_style()
    segment_path = args.segment.resolve(); gt_dir = args.gt_dir.resolve()
    out_dir = (args.output_dir or (Path(__file__).resolve().parent / "figures" / f"figure04_edges_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events and (optional) params
    x, y, t, p = load_segment_events(segment_path)
    sensor_area = float(args.sensor_width * args.sensor_height)
    params_file = find_param_file(segment_path)
    a_params = b_params = None
    if params_file.exists():
        params = load_params(params_file)
        a_params = params.get("a_params"); b_params = params.get("b_params")

    # Build fine series
    neg_scale, series, _ = compute_series_and_weights(
        x.astype(np.float32), y.astype(np.float32), t.astype(np.float32), p.astype(np.float32),
        a_params, b_params, sensor_area, args.pos_scale, args.neg_scale, args.fine_step_us,
    )
    save_series_npz(series, out_dir)

    # Strict edges-only alignment and overlay, plus all-in-one style artifacts
    payload = align_edges_only(series, gt_dir, args.edge_quantile, out_dir, neg_scale, args.fine_step_us, args.save_png)

    # Load GT curves again for auxiliary plots
    gt_files = sorted(gt_dir.glob("*.txt"))
    gt_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for txt in gt_files:
        try:
            wl, val = np.loadtxt(txt, skiprows=17, usecols=[0, 1], unpack=True)
        except Exception:
            from groundtruth_spectrum.compare_reconstruction_to_gt import load_ground_truth  # type: ignore
            wl, val = load_ground_truth(txt)
        if wl.size >= 50 and val.size == wl.size:
            gt_curves.append((txt.stem, wl.astype(np.float32), val.astype(np.float32)))
    wl_series = (payload["alignment"]["slope_nm_per_ms"] * series["time_ms"] + payload["alignment"]["intercept_nm"]).astype(np.float32)  # type: ignore
    plot_three_panel_normalised(series, gt_curves, wl_series, out_dir, args.save_png)
    plot_bg_vs_gt(series, gt_curves, wl_series, out_dir, args.save_png)

    # Build bin frames and render grid
    sensor_shape = (args.sensor_height, args.sensor_width)
    originals, compensated, meta = build_bin_frames(
        x.astype(np.int16), y.astype(np.int16), t.astype(np.float32), p.astype(np.float32),
        a_params, b_params, sensor_shape, args.bin_width_us, args.start_bin, args.end_bin,
        args.pos_scale, float(payload["neg_scale"]) if isinstance(payload.get("neg_scale"), float) else neg_scale, bool(args.smooth),
    )
    shared_base = "coolwarm"
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", 0.25)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", 0.30)
    render_grid(
        originals,
        compensated,
        meta,
        args.start_bin,
        args.end_bin,
        raw_cmap,
        comp_cmap,
        None,
        False,
        out_dir,
        args.save_png,
    )

    # Save weights/frames summary mirroring all-in-one names
    # Background means for saved frames
    bg_means = [float(np.mean(f)) for f in compensated]
    weights_json = out_dir / "figure04_rescaled_weights.json"
    with weights_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "segment": str(segment_path),
                "pos_scale": args.pos_scale,
                "neg_scale": float(payload.get("neg_scale", neg_scale)),
                "bin_width_us": args.bin_width_us,
                "bin_width_ms": args.bin_width_us / 1000.0,
                "num_bins": int(np.ceil((float(np.max(t)) - float(np.min(t))) / args.bin_width_us)),
                "smooth": bool(args.smooth),
                "bin_times_us": meta,
                "background_means": bg_means,
                "rescale_step_us": args.fine_step_us,
                "bg_series_npz": "figure04_rescaled_bg_series.npz",
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()
