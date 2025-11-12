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
)
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
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Normalised intensity (a.u.)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.axvline(wl_low, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(wl_high, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    out_plot = out_dir / "figure04_edges_only_third.pdf"
    fig.savefig(out_plot, dpi=400, bbox_inches="tight")
    if save_png:
        fig.savefig(out_plot.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    with (out_dir / "figure04_edges_only_alignment.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return payload


def save_series_npz(series: Dict[str, np.ndarray], out_dir: Path) -> None:
    path = out_dir / "figure04_edges_only_series.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in series.items()})


def main() -> None:
    args = parse_args()
    setup_style()
    segment_path = args.segment.resolve(); gt_dir = args.gt_dir.resolve()
    out_dir = (args.output_dir or (Path(__file__).resolve().parent / "figures" / f"figure04_edges_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events and (optional) params
    x, y, t, p = load_segment_events(segment_path)
    sensor_area = float(args.sensor_width * args.sensor_height)
    params_file = segment_path.with_name(segment_path.stem + "_chunked_processing_learned_params_n13.npz")
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

    # Strict edges-only alignment and overlay
    align_edges_only(series, gt_dir, args.edge_quantile, out_dir, neg_scale, args.fine_step_us, args.save_png)


if __name__ == "__main__":
    main()

