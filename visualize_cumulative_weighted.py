#!/usr/bin/env python3
"""
Weighted cumulative means visualization (configurable step size) with adjustable
polarity scaling and optional compensation.

Usage:
  python visualize_cumulative_weighted.py <segment_npz>
    --sensor_width 1280 --sensor_height 720 \
    --pos_scale 1.0 --neg_scale 1.5 \
    --step_us 100 \
    --ymin 0.1 --ymax 2.0 \
    [--no_comp] [--exp] [--auto_scale]

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
    ap.add_argument("--step_us", type=float, default=100.0, help="Step size in microseconds (default 100)")
    ap.add_argument("--ymin", type=float, default=0.1, help="Y-axis min for plotting (default 0.1)")
    ap.add_argument("--ymax", type=float, default=2.0, help="Y-axis max for plotting (default 2.0)")
    ap.add_argument("--exp", action='store_true', help="Plot exponential of accumulation (apply exp() to series)")
    ap.add_argument("--no_comp", action='store_true', help="Do not compute/show compensated series")
    ap.add_argument("--auto_scale", action='store_true', help="Auto-tune neg_scale to equalize start/end plateaus")
    ap.add_argument("--plateau_frac", type=float, default=0.05, help="Fraction of bins at each end used for plateau averaging (default 0.05)")
    ap.add_argument("--auto_min", type=float, default=0.1, help="Min neg_scale for auto search (default 0.1)")
    ap.add_argument("--auto_max", type=float, default=3.0, help="Max neg_scale for auto search (default 3.0)")
    args = ap.parse_args()

    x, y, t, p = load_npz_events(args.segment_npz)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    hw_size = float(args.sensor_width * args.sensor_height)

    # Build weights: +pos_scale for p>0, -neg_scale for p<0
    weights = np.where(p >= 0, args.pos_scale, -args.neg_scale).astype(np.float32)

    # Helper: per-polarity binned sums (original)
    pos_mask = p >= 0
    neg_mask = ~pos_mask
    sums_pos_orig, edges_ms = base_binned_sums_weighted(t[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.float32), t_min, t_max, args.step_us)
    sums_neg_orig, _ = base_binned_sums_weighted(t[neg_mask], np.ones(np.count_nonzero(neg_mask), dtype=np.float32), t_min, t_max, args.step_us)

    def build_cum_from_sums(s_pos, s_neg, pos_scale, neg_scale):
        step_sums = pos_scale * s_pos - neg_scale * s_neg
        return np.cumsum(step_sums) / hw_size

    # Auto-scale (neg_scale) to equalize plateaus on the plotted series (linear or exp)
    chosen_neg = args.neg_scale
    if args.auto_scale:
        # Define objective on current domain (after optional exp)
        def plateau_diff(neg_scale):
            cm = build_cum_from_sums(sums_pos_orig, sums_neg_orig, args.pos_scale, neg_scale)
            series = np.exp(cm) if args.exp else cm
            K = len(series)
            if K <= 2:
                return 0.0
            n = max(5, int(args.plateau_frac * K))
            start_mean = float(np.mean(series[:n]))
            end_mean = float(np.mean(series[-n:]))
            return end_mean - start_mean

        # Bisection with fallback grid search
        a, b = max(1e-6, args.auto_min), max(args.auto_min + 1e-6, args.auto_max)
        fa, fb = plateau_diff(a), plateau_diff(b)
        if fa == 0:
            chosen_neg = a
        elif fb == 0:
            chosen_neg = b
        elif fa * fb < 0:
            # Bisection
            for _ in range(40):
                m = 0.5 * (a + b)
                fm = plateau_diff(m)
                if abs(fm) < 1e-6:
                    chosen_neg = m
                    break
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                chosen_neg = 0.5 * (a + b)
        else:
            # Fallback: coarse grid search
            grid = np.linspace(a, b, 50)
            vals = np.array([abs(plateau_diff(g)) for g in grid])
            chosen_neg = float(grid[int(np.argmin(vals))])
        print(f"Auto-scale: pos_scale={args.pos_scale:.3f}, neg_scale={chosen_neg:.3f}")

    # Base per-step weighted sums and cumulative means (original time) with chosen neg scale
    cum_means_orig = build_cum_from_sums(sums_pos_orig, sums_neg_orig, args.pos_scale, chosen_neg)

    # Compensated series
    t_comp = None
    cum_means_comp = None
    a_avg = b_avg = None
    if not args.no_comp:
        param_file = find_param_file_for_segment(args.segment_npz)
        if param_file is not None:
            params = load_parameters_from_npz(param_file)
            t_comp, a_avg, b_avg = compute_fast_compensated_times(x, y, t, params['a_params'], params['b_params'])
            # Per-polarity comp sums
            sums_pos_comp, _ = base_binned_sums_weighted(t_comp[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.float32), t_min, t_max, args.step_us)
            sums_neg_comp, _ = base_binned_sums_weighted(t_comp[neg_mask], np.ones(np.count_nonzero(neg_mask), dtype=np.float32), t_min, t_max, args.step_us)
            cum_means_comp = build_cum_from_sums(sums_pos_comp, sums_neg_comp, args.pos_scale, chosen_neg)
        else:
            print("Warning: no learned params found; skipping compensated series")

    # Optional exponential transform
    if args.exp:
        plot_orig = np.exp(cum_means_orig)
        plot_comp = np.exp(cum_means_comp) if cum_means_comp is not None else None
    else:
        plot_orig = cum_means_orig
        plot_comp = cum_means_comp

    # Output directory
    out_dir = os.path.join(os.path.dirname(args.segment_npz), f"FIXED_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "cumulative_weighted")
    os.makedirs(out_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(edges_ms, plot_orig, 'b-', label=f'Orig (pos={args.pos_scale}, neg={chosen_neg:.3f})')
    if plot_comp is not None:
        lbl = f'Comp (pos={args.pos_scale}, neg={chosen_neg:.3f})'
        if a_avg is not None and b_avg is not None:
            lbl += f" | a_avg={a_avg:.3f}, b_avg={b_avg:.3f}"
        plt.plot(edges_ms, plot_comp, 'r-', label=lbl)
    plt.xlabel('Time (ms)')
    plt.ylabel('Per-pixel mean (weighted)' + (" (exp)" if args.exp else ""))
    plt.ylim(args.ymin, args.ymax)
    # Title reflects chosen step size
    step_label = (f"{args.step_us/1000:.3f} ms" if args.step_us >= 1000.0 else f"{args.step_us:.0f} μs")
    plt.title(f'Weighted Cumulative Means ({step_label} steps)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    suffix = "_exp" if args.exp else ""
    out_png = os.path.join(out_dir, f"weighted_cumulative_pos{args.pos_scale}_neg{chosen_neg:.3f}{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n✓ Weighted cumulative plot saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
