#!/usr/bin/env python3
"""
Weighted cumulative means visualization (2ms steps) with adjustable
polarity scaling and optional compensation.

Usage:
  python visualize_cumulative_weighted.py <segment_npz>
    --sensor_width 1280 --sensor_height 720 \
    --pos_scale 1.0 --neg_scale 1.5 \
    --ymin 0.1 --ymax 2.0 \
    [--no_comp]

Notes:
- Loads learned parameters (NPZ) next to the segment file to compute
  FAST compensation (t' = t - a_avg x - b_avg y) unless --no_comp.
- Cumulative means are computed at 2ms steps from t_min to t_max and
  divided by (sensor_height * sensor_width) for per-pixel means.
- Polarity weights: positive events contribute +pos_scale, negative
  events contribute -neg_scale (default 1.5 for negatives).
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
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def load_parameters_from_npz(param_file):
    data = np.load(param_file)
    return {
        'a_params': data['a_params'].astype(np.float32),
        'b_params': data['b_params'].astype(np.float32),
    }


def compute_fast_compensated_times(x, y, t, a_params, b_params):
    a_avg = float(np.mean(a_params))
    b_avg = float(np.mean(b_params))
    t_comp = t - (a_avg * x + b_avg * y)
    return t_comp, a_avg, b_avg


def base_binned_sums_weighted(times, weights, t_min, t_max, step_us):
    """Return per-step sums and time edges for cumulative computation."""
    K = int((t_max - t_min) // step_us)
    if K <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    in_range = (times >= t_min) & (times < t_min + K * step_us)
    if not np.any(in_range):
        edges_ms = (t_min + step_us * (np.arange(1, K + 1, dtype=np.float32))) / 1000.0
        return np.zeros(K, dtype=np.float32), edges_ms
    idx = ((times[in_range] - t_min) // step_us).astype(np.int64)
    sums = np.bincount(idx, weights=weights[in_range], minlength=K).astype(np.float32)
    edges_ms = (t_min + step_us * (np.arange(1, K + 1, dtype=np.float32))) / 1000.0
    return sums, edges_ms


def main():
    ap = argparse.ArgumentParser(description="Weighted cumulative means (2ms steps) with polarity scaling and optional compensation")
    ap.add_argument("segment_npz", help="Path to segmented events NPZ (e.g., Scan_1_Forward_events.npz)")
    ap.add_argument("--sensor_width", type=int, default=1280)
    ap.add_argument("--sensor_height", type=int, default=720)
    ap.add_argument("--pos_scale", type=float, default=1.0, help="Weight for positive events (default 1.0)")
    ap.add_argument("--neg_scale", type=float, default=1.5, help="Weight for negative events (default 1.5)")
    ap.add_argument("--step_us", type=float, default=2000.0, help="Step size in microseconds (default 2000)")
    ap.add_argument("--ymin", type=float, default=0.1, help="Y-axis min for plotting (default 0.1)")
    ap.add_argument("--ymax", type=float, default=2.0, help="Y-axis max for plotting (default 2.0)")
    ap.add_argument("--no_comp", action='store_true', help="Do not compute/show compensated series")
    args = ap.parse_args()

    x, y, t, p = load_npz_events(args.segment_npz)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    hw_size = float(args.sensor_width * args.sensor_height)

    # Build weights: +pos_scale for p>0, -neg_scale for p<0
    weights = np.where(p >= 0, args.pos_scale, -args.neg_scale).astype(np.float32)

    # Base per-step weighted sums and cumulative means (original time)
    sums_orig, edges_ms = base_binned_sums_weighted(t, weights, t_min, t_max, args.step_us)
    cum_means_orig = np.cumsum(sums_orig) / hw_size

    # Compensated series
    t_comp = None
    cum_means_comp = None
    a_avg = b_avg = None
    if not args.no_comp:
        param_file = find_param_file_for_segment(args.segment_npz)
        if param_file is not None:
            params = load_parameters_from_npz(param_file)
            t_comp, a_avg, b_avg = compute_fast_compensated_times(x, y, t, params['a_params'], params['b_params'])
            sums_comp, _ = base_binned_sums_weighted(t_comp, weights, t_min, t_max, args.step_us)
            cum_means_comp = np.cumsum(sums_comp) / hw_size
        else:
            print("Warning: no learned params found; skipping compensated series")

    # Output directory
    out_dir = os.path.join(os.path.dirname(args.segment_npz), f"FIXED_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "cumulative_weighted")
    os.makedirs(out_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(edges_ms, cum_means_orig, 'b-', label=f'Orig (pos={args.pos_scale}, neg={args.neg_scale})')
    if cum_means_comp is not None:
        lbl = f'Comp (pos={args.pos_scale}, neg={args.neg_scale})'
        if a_avg is not None and b_avg is not None:
            lbl += f" | a_avg={a_avg:.3f}, b_avg={b_avg:.3f}"
        plt.plot(edges_ms, cum_means_comp, 'r-', label=lbl)
    plt.xlabel('Time (ms)')
    plt.ylabel('Per-pixel mean (weighted)')
    plt.ylim(args.ymin, args.ymax)
    plt.title('Weighted Cumulative Means (2ms steps)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = os.path.join(out_dir, f"weighted_cumulative_pos{args.pos_scale}_neg{args.neg_scale}.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n✓ Weighted cumulative plot saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()

