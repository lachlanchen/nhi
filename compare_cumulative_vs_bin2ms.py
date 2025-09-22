#!/usr/bin/env python3
"""
Compare cumulative 2ms-step frame means vs 2ms bin/shift frame means,
for both original and compensated timestamps. No frames are saved; only
per-frame means are computed and plotted.

Usage:
  python compare_cumulative_vs_bin2ms.py <segment_npz>

Optional args:
  --sensor_width  1280
  --sensor_height 720
  --output_dir    <defaults to <segment_dir>/cumulative_vs_bin2ms_TIMESTAMP>
  --sample_label  Custom label for plot title

Notes:
- Compensation uses the FAST linear form with a_avg and b_avg computed
  from the learned parameter NPZ stored next to <segment_npz>.
- The cumulative mean at time T_k uses all events t < T_k (or t'_comp < T_k),
  divided by (H*W). The sliding (2ms bin) mean uses events inside
  [t0 + k*2ms, t0 + (k+1)*2ms) divided by (H*W).
- Because means are per-pixel averages, we never construct frames; we
  only sum polarities and divide by H*W.
"""

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def load_npz_events(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    data = np.load(npz_path)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)
    # Convert polarity to [-1, 1] if in [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2.0
    return x, y, t, p


def find_param_file_for_segment(segment_npz):
    """Return the most recent learned-params NPZ next to the segment file."""
    d = os.path.dirname(segment_npz)
    base = os.path.splitext(os.path.basename(segment_npz))[0]
    patterns = [
        f"{base}_chunked_processing_learned_params_*.npz",
        f"{base}_learned_params_*.npz",
        f"{base}*learned_params*.npz",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(d, pat)))
    if not candidates:
        raise FileNotFoundError(f"No learned parameter NPZ found for {segment_npz}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_parameters_from_npz(param_file):
    data = np.load(param_file)
    return {
        'a_params': data['a_params'].astype(np.float32),
        'b_params': data['b_params'].astype(np.float32),
        'num_params': int(data['num_params']),
        'duration': float(data['duration']),
        'temperature': float(data['temperature']),
    }


def compute_fast_compensated_times(x, y, t, a_params, b_params):
    a_avg = float(np.mean(a_params))
    b_avg = float(np.mean(b_params))
    # t' = t - a*x - b*y
    t_comp = t - (a_avg * x + b_avg * y)
    return t_comp, a_avg, b_avg


def cumulative_means(times, p, t_min, t_max, step_us, hw_size):
    """Compute cumulative per-pixel means at T_k = t_min + k*step.
    Return (T_edges_ms, means)."""
    # Sort by time for prefix sums
    order = np.argsort(times)
    t_sorted = times[order]
    p_sorted = p[order]
    p_cumsum = np.cumsum(p_sorted)

    # Edges (exclude zero): k = 1..K where t_min + k*step <= t_max
    K = int((t_max - t_min) // step_us)
    if K <= 0:
        return np.array([]), np.array([])
    edges = t_min + step_us * (np.arange(1, K + 1, dtype=np.float32))

    # For each edge, find index of first time > edge (np.searchsorted, 'right')
    idxs = np.searchsorted(t_sorted, edges, side='right')
    # Prefix sums up to idxs-1
    sums = np.where(idxs > 0, p_cumsum[idxs - 1], 0.0)
    means = sums / float(hw_size)
    return edges / 1000.0, means  # ms


def sliding_bin_means(times, p, t_min, t_max, step_us, hw_size):
    """Compute sliding-bin (bin=step, shift=step) per-pixel means for k bins fully inside [t_min, t_max).
    Return (bin_centers_ms, means)."""
    # Bin index relative to t_min
    K = int((t_max - t_min) // step_us)
    if K <= 0:
        return np.array([]), np.array([])

    # Ignore events outside [t_min, t_min + K*step)
    in_range = (times >= t_min) & (times < t_min + K * step_us)
    if not np.any(in_range):
        centers = t_min + step_us * (np.arange(K, dtype=np.float32) + 0.5)
        return centers / 1000.0, np.zeros(K, dtype=np.float32)

    idx = ((times[in_range] - t_min) // step_us).astype(np.int64)
    w = p[in_range]
    sums = np.bincount(idx, weights=w, minlength=K).astype(np.float32)
    means = sums / float(hw_size)
    # Bin centers for plotting
    centers = t_min + step_us * (np.arange(K, dtype=np.float32) + 0.5)
    return centers / 1000.0, means  # ms


def finite_difference_per_step(y):
    """Simple forward difference of cumulative means, per 2ms step (no 1/Δ scaling).
    This is directly comparable to the 2ms-bin mean values."""
    if len(y) < 2:
        return y
    dy = np.diff(y)
    return np.concatenate([[dy[0]], dy])


def main():
    ap = argparse.ArgumentParser(description="Compare cumulative 2ms-step means vs 2ms-bin means (orig vs compensated)")
    ap.add_argument("segment_npz", help="Path to a segmented events NPZ (e.g., Scan_1_Forward_events.npz)")
    ap.add_argument("--sensor_width", type=int, default=1280)
    ap.add_argument("--sensor_height", type=int, default=720)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sample_label", default=None)
    args = ap.parse_args()

    x, y, t, p = load_npz_events(args.segment_npz)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    step_us = 2000.0  # 2 ms
    hw_size = float(args.sensor_width * args.sensor_height)

    # Load learned parameters and compute FAST compensated times
    param_file = find_param_file_for_segment(args.segment_npz)
    params = load_parameters_from_npz(param_file)
    t_comp, a_avg, b_avg = compute_fast_compensated_times(x, y, t, params['a_params'], params['b_params'])

    # Means for cumulative (orig vs comp)
    t_edges_ms, cum_mean_orig = cumulative_means(t, p, t_min, t_max, step_us, hw_size)
    _, cum_mean_comp = cumulative_means(t_comp, p, t_min, t_max, step_us, hw_size)

    # Means for sliding bins (orig vs comp)
    t_bins_ms, bin_mean_orig = sliding_bin_means(t, p, t_min, t_max, step_us, hw_size)
    _, bin_mean_comp = sliding_bin_means(t_comp, p, t_min, t_max, step_us, hw_size)

    # Finite-difference of cumulative means (per ms) to compare with bin means
    # Scale to comparable magnitude: bin means already are per-frame means; fd is per ms change of the cumulative mean.
    fd_cum_orig = finite_difference_per_step(cum_mean_orig)
    fd_cum_comp = finite_difference_per_step(cum_mean_comp)

    # Output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.path.dirname(args.segment_npz), f"cumulative_vs_bin2ms_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    title_suffix = args.sample_label if args.sample_label else os.path.basename(args.segment_npz)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) Cumulative means
    ax = axes[0]
    ax.plot(t_edges_ms, cum_mean_orig, 'b-', label='Cumulative mean (orig)')
    ax.plot(t_edges_ms, cum_mean_comp, 'r-', label='Cumulative mean (comp)')
    ax.set_title(f'Cumulative means @2ms steps\n{title_suffix}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Sliding 2ms-bin means
    ax = axes[1]
    ax.plot(t_bins_ms, bin_mean_orig, 'b-', label='2ms bin mean (orig)')
    ax.plot(t_bins_ms, bin_mean_comp, 'r-', label='2ms bin mean (comp)')
    ax.set_title('Sliding means (bin=2ms, shift=2ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Sliding vs finite-diff of cumulative
    ax = axes[2]
    ax.plot(t_bins_ms[:len(fd_cum_orig)], fd_cum_orig[:len(t_bins_ms)], 'c-', label='d/dt cumulative (orig)')
    ax.plot(t_bins_ms, bin_mean_orig, 'b--', alpha=0.7, label='2ms bin mean (orig)')
    ax.plot(t_bins_ms[:len(fd_cum_comp)], fd_cum_comp[:len(t_bins_ms)], 'm-', label='d/dt cumulative (comp)')
    ax.plot(t_bins_ms, bin_mean_comp, 'r--', alpha=0.7, label='2ms bin mean (comp)')
    ax.set_title('Comparison: derivative of cumulative vs 2ms-bin means')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean (per 2ms step)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(args.output_dir, 'cumulative_vs_bin2ms_means.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
