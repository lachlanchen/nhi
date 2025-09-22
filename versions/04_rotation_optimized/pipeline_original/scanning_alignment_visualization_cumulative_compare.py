#!/usr/bin/env python3
"""
Scanning Visualization: Cumulative vs Multi-Bin Means (2ms shift)

Compares continuous cumulative 2ms-step means with sliding bin means at
2ms, 50ms, and 100ms bin widths (all with 2ms shift), for both original
and FAST-compensated timestamps. No frames are saved; only per-frame
means are computed and plotted, similar naming style to other
scanning_alignment_visualization_* scripts.

Usage:
  python scanning_alignment_visualization_cumulative_compare.py <data_file.npz>

Optional args:
  --sensor_width  1280
  --sensor_height 720
  --output_dir    <defaults to <data_dir>/FIXED_visualization_<TS>/cumulative_compare>
  --sample_label  Custom label for plot title

Notes:
- Compensation uses the FAST linear form with a_avg and b_avg computed
  from the learned parameter NPZ stored next to <data_file.npz>.
- Cumulative mean at step k uses all events t < t_min + (k+1)*2ms, divided
  by (H*W). Sliding-bin mean uses events inside [start, start+bin_width)
  divided by (H*W), with start stepping every 2ms.
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


def find_param_file_for_segment(data_npz):
    """Return the most recent learned-params NPZ next to the data file."""
    d = os.path.dirname(data_npz)
    base = os.path.splitext(os.path.basename(data_npz))[0]
    patterns = [
        f"{base}_chunked_processing_learned_params_*.npz",
        f"{base}_learned_params_*.npz",
        f"{base}*learned_params*.npz",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(d, pat)))
    if not candidates:
        raise FileNotFoundError(f"No learned parameter NPZ found for {data_npz}")
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


def base_2ms_binned_sums(times, p, t_min, t_max, step_us):
    """Return per-2ms-bin sums and corresponding time arrays.
    times/p are numpy arrays, step_us=2000.0.
    Output:
      sums: shape [K], sums of p per 2ms bin
      t_centers_ms: centers of each 2ms bin (ms)
      t_edges_ms: right-edge times of cumulative steps (ms)
    """
    K = int((t_max - t_min) // step_us)
    if K <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    in_range = (times >= t_min) & (times < t_min + K * step_us)
    idx = ((times[in_range] - t_min) // step_us).astype(np.int64)
    sums = np.bincount(idx, weights=p[in_range], minlength=K).astype(np.float32)
    t_centers_ms = (t_min + step_us * (np.arange(K, dtype=np.float32) + 0.5)) / 1000.0
    t_edges_ms = (t_min + step_us * (np.arange(1, K + 1, dtype=np.float32))) / 1000.0
    return sums, t_centers_ms, t_edges_ms


def moving_window_means_from_base(base_sums, R, hw):
    """Compute sliding window sums of base_sums with window length R bins.
    Returns per-pixel means and count K-R+1."""
    if len(base_sums) < R:
        return np.zeros(0, dtype=np.float32)
    c = np.cumsum(base_sums, dtype=np.float64)
    # moving sums: s[k] = c[k+R-1] - (c[k-1] if k>0 else 0)
    # vectorized via c padding
    c_pad = np.concatenate([[0.0], c])
    moving = c_pad[R:] - c_pad[:-R]
    means = moving.astype(np.float32) / float(hw)
    return means


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
    """Simple forward difference of cumulative means, per 2ms step (no 1/Δ scaling)."""
    if len(y) < 2:
        return y
    dy = np.diff(y)
    return np.concatenate([[dy[0]], dy])


def main():
    ap = argparse.ArgumentParser(description="Scanning Visualization: Cumulative vs 2/50/100ms bin means (2ms shift)")
    ap.add_argument("data_file", help="Path to NPZ segment file (e.g., Scan_1_Forward_events.npz)")
    ap.add_argument("--sensor_width", type=int, default=1280)
    ap.add_argument("--sensor_height", type=int, default=720)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sample_label", default=None)
    args = ap.parse_args()

    # Load events
    x, y, t, p = load_npz_events(args.data_file)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    step_us = 2000.0  # 2 ms step
    hw_size = float(args.sensor_width * args.sensor_height)

    # Load learned parameters and compute FAST compensated times
    param_file = find_param_file_for_segment(args.data_file)
    params = load_parameters_from_npz(param_file)
    t_comp, a_avg, b_avg = compute_fast_compensated_times(x, y, t, params['a_params'], params['b_params'])

    # Base 2ms bins for orig and comp
    base_sums_orig, centers_ms, edges_ms = base_2ms_binned_sums(t, p, t_min, t_max, step_us)
    base_sums_comp, centers_ms_comp, edges_ms_comp = base_2ms_binned_sums(t_comp, p, t_min, t_max, step_us)

    # Cumulative means (2ms steps)
    cum_means_orig = np.cumsum(base_sums_orig) / hw_size
    cum_means_comp = np.cumsum(base_sums_comp) / hw_size

    # Sliding 2ms means are just base/bin means
    bin2_means_orig = base_sums_orig / hw_size
    bin2_means_comp = base_sums_comp / hw_size

    # Sliding 50ms means (R=25 bins) and 100ms means (R=50 bins)
    R50, R100 = 25, 50
    bin50_means_orig = moving_window_means_from_base(base_sums_orig, R50, hw_size)
    bin50_means_comp = moving_window_means_from_base(base_sums_comp, R50, hw_size)
    bin100_means_orig = moving_window_means_from_base(base_sums_orig, R100, hw_size)
    bin100_means_comp = moving_window_means_from_base(base_sums_comp, R100, hw_size)

    # Time axes for 50ms and 100ms windows (centers)
    centers50_ms = (t_min + step_us * (np.arange(len(bin50_means_orig), dtype=np.float32) + R50 / 2.0)) / 1000.0
    centers100_ms = (t_min + step_us * (np.arange(len(bin100_means_orig), dtype=np.float32) + R100 / 2.0)) / 1000.0

    # Output directory styled like other visualization scripts
    data_dir = os.path.dirname(args.data_file)
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(data_dir, f"FIXED_visualization_{ts}", "cumulative_compare")
    os.makedirs(args.output_dir, exist_ok=True)

    title_suffix = args.sample_label if args.sample_label else os.path.basename(args.data_file)

    print("=" * 80)
    print("FIXED: CUMULATIVE VS MULTI-BIN MEANS (2ms shift)")
    print(f"Input file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sensor: {args.sensor_width}x{args.sensor_height}")
    print(f"FAST compensation a_avg={a_avg:.4f}, b_avg={b_avg:.4f} from {os.path.basename(param_file)}")
    print("=" * 80)

    # Plots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # (0,0) Cumulative means
    ax = axes[0, 0]
    ax.plot(edges_ms, cum_means_orig, 'b-', label='Cumulative (orig)')
    ax.plot(edges_ms, cum_means_comp, 'r-', label='Cumulative (comp)')
    ax.set_title(f'Cumulative means @2ms steps\n{title_suffix}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (0,1) 2ms sliding means
    ax = axes[0, 1]
    ax.plot(centers_ms, bin2_means_orig, 'b-', label='2ms (orig)')
    ax.plot(centers_ms, bin2_means_comp, 'r-', label='2ms (comp)')
    ax.set_title('Sliding means (bin=2ms, shift=2ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,0) 50ms sliding means
    ax = axes[1, 0]
    ax.plot(centers50_ms, bin50_means_orig, 'b-', label='50ms (orig)')
    ax.plot(centers50_ms, bin50_means_comp, 'r-', label='50ms (comp)')
    ax.set_title('Sliding means (bin=50ms, shift=2ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,1) 100ms sliding means
    ax = axes[1, 1]
    ax.plot(centers100_ms, bin100_means_orig, 'b-', label='100ms (orig)')
    ax.plot(centers100_ms, bin100_means_comp, 'r-', label='100ms (comp)')
    ax.set_title('Sliding means (bin=100ms, shift=2ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Per-pixel mean')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(args.output_dir, 'cumulative_vs_multi_bin_means.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
