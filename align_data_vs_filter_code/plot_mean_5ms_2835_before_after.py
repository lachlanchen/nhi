#!/usr/bin/env python3
"""
Plot per-bin mean (net event rate) before and after compensation (5 ms bins).

The wavelength axis is obtained from a JSON mapping produced by
`compare_publication_three_panel.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import visualize_cumulative_weighted as vcw  # type: ignore
from compensate_multiwindow_train_saved_params import Compensate  # type: ignore


def bin_polarity_sums(
    times_us: np.ndarray,
    p: np.ndarray,
    bin_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edges_ms, sums_pos, sums_neg) for given timestamps and polarities."""
    t_min, t_max = float(np.min(times_us)), float(np.max(times_us))
    pos_mask = p >= 0
    neg_mask = ~pos_mask
    step_us = bin_ms * 1000.0

    sums_pos, edges_ms = vcw.base_binned_sums_weighted(
        times_us[pos_mask],
        np.ones(np.count_nonzero(pos_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )
    sums_neg, _ = vcw.base_binned_sums_weighted(
        times_us[neg_mask],
        np.ones(np.count_nonzero(neg_mask), dtype=np.float32),
        t_min,
        t_max,
        step_us,
    )
    return edges_ms, sums_pos, sums_neg


def build_net_rate(
    sums_pos: np.ndarray,
    sums_neg: np.ndarray,
    hw: float,
    pos_scale: float,
    neg_scale: float,
) -> np.ndarray:
    """Per-bin net event rate from per-polarity sums."""
    return (pos_scale * sums_pos - neg_scale * sums_neg) / hw


def load_multiwindow_params(param_path: Path) -> dict:
    """Load multi-window parameters from a learned-params NPZ file."""
    data = np.load(param_path)
    return {
        "a_params": data["a_params"].astype(np.float32),
        "b_params": data["b_params"].astype(np.float32),
        "num_params": int(data["num_params"]),
        "temperature": float(data.get("temperature", 5000.0)),
    }


def compensate_times_multiwindow(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    params: dict,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Full multi-window compensation: return t' = t - compensation(x, y, t).

    This mirrors the logic used in publication/publication_code/event_cloud_multiwindow.py.
    """
    # Work in a time-shifted domain starting at 0 μs
    t_min = float(t.min())
    t_shift = t - t_min
    duration = float(t_shift.max())

    comp = Compensate(
        params["a_params"],
        params["b_params"],
        duration,
        num_params=params["num_params"],
        temperature=params["temperature"],
        device="cpu",
        a_fixed=True,
        b_fixed=True,
        boundary_trainable=False,
        debug=False,
    )

    with torch.no_grad():
        compensation = comp.compute_event_compensation(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t_shift, dtype=torch.float32),
            chunk_size=chunk_size,
            debug=False,
        ).cpu().numpy()

    return t_shift - compensation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot 5 ms net event rate vs wavelength before/after compensation."
    )
    ap.add_argument(
        "--segment",
        type=Path,
        required=True,
        help="Segment NPZ file (e.g., Scan_1_Forward_events.npz)",
    )
    ap.add_argument(
        "--mapping-json",
        type=Path,
        required=True,
        help="JSON file with slope/intercept (from *_mapping.json)",
    )
    ap.add_argument("--bin-ms", type=float, default=5.0)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help="Weight for positive events when forming bin means (default 1.0)",
    )
    ap.add_argument(
        "--neg-scale",
        type=float,
        default=1.260171,
        help="Weight for negative events (default 1.260171, as used in Lumileds cumsum plots)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for multi-window compensation (events per batch)",
    )
    ap.add_argument(
        "--auto-neg-min",
        type=float,
        default=0.1,
        help="Minimum neg_scale when auto-aligning cumsum plateaus (default 0.1)",
    )
    ap.add_argument(
        "--auto-neg-max",
        type=float,
        default=3.0,
        help="Maximum neg_scale when auto-aligning cumsum plateaus (default 3.0)",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.05,
        help="Fraction of bins at each end used for plateau averaging (default 0.05)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path (default: alongside segment, suffix _mean_5ms_before_after)",
    )
    args = ap.parse_args()

    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    slope = float(mapping["slope_nm_per_ms"])
    intercept = float(mapping["intercept_nm"])

    x, y, t, p = vcw.load_npz_events(str(args.segment))
    hw = float(args.sensor_width * args.sensor_height)

    # After compensation: use full multi-window Compensate (if params found).
    param_file = vcw.find_param_file_for_segment(str(args.segment))
    t_comp = None
    if param_file is not None:
        params = load_multiwindow_params(Path(param_file))
        t_comp = compensate_times_multiwindow(x, y, t, params, chunk_size=args.chunk_size)

    # Bin per-polarity sums for raw and compensated times
    edges_raw_ms, sums_pos_raw, sums_neg_raw = bin_polarity_sums(t, p, bin_ms=args.bin_ms)
    if t_comp is not None:
        edges_comp_ms, sums_pos_comp, sums_neg_comp = bin_polarity_sums(
            t_comp, p, bin_ms=args.bin_ms
        )
    else:
        edges_comp_ms, sums_pos_comp, sums_neg_comp = edges_raw_ms, sums_pos_raw, sums_neg_raw

    # Auto-scale negative weight so the cumulative (compensated) plateaus align
    neg_scale_used = args.neg_scale
    if sums_pos_comp.size > 0 and sums_neg_comp.size > 0:

        def plateau_diff(neg_scale: float) -> float:
            net_comp_tmp = build_net_rate(
                sums_pos_comp, sums_neg_comp, hw, args.pos_scale, neg_scale
            )
            series = np.cumsum(net_comp_tmp)
            k = len(series)
            if k <= 2:
                return 0.0
            n = max(5, int(args.plateau_frac * k))
            start_mean = float(np.mean(series[:n]))
            end_mean = float(np.mean(series[-n:]))
            return end_mean - start_mean

        a = max(1e-6, args.auto_neg_min)
        b = max(a + 1e-6, args.auto_neg_max)
        fa, fb = plateau_diff(a), plateau_diff(b)
        if fa == 0:
            neg_scale_used = a
        elif fb == 0:
            neg_scale_used = b
        elif fa * fb < 0:
            # Bisection search
            for _ in range(40):
                m = 0.5 * (a + b)
                fm = plateau_diff(m)
                if abs(fm) < 1e-6:
                    neg_scale_used = m
                    break
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                neg_scale_used = 0.5 * (a + b)
        else:
            # Fallback: coarse grid search
            grid = np.linspace(a, b, 50, dtype=np.float32)
            vals = np.array([abs(plateau_diff(g)) for g in grid])
            neg_scale_used = float(grid[int(np.argmin(vals))])

    print(f"Auto-scaled neg_scale: {neg_scale_used:.6f} (pos_scale={args.pos_scale:.3f})")

    # Build net-rate series using the chosen neg_scale
    net_raw = build_net_rate(sums_pos_raw, sums_neg_raw, hw, args.pos_scale, neg_scale_used)
    net_comp = build_net_rate(sums_pos_comp, sums_neg_comp, hw, args.pos_scale, neg_scale_used)

    # Time and wavelength axes
    t_raw_ms = edges_raw_ms - edges_raw_ms[0]
    wl_raw = slope * t_raw_ms + intercept
    t_comp_ms = edges_comp_ms - edges_comp_ms[0]
    wl_comp = slope * t_comp_ms + intercept

    # Plot per-bin means
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_raw, net_raw, color="tab:gray", alpha=0.6, label="Raw time (before comp)")
    ax.plot(wl_comp, net_comp, color="tab:green", label="Compensated time")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Per-pixel net event rate (5 ms bins)")
    ax.set_title(f"Mean event rate vs wavelength\n{args.segment.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    weight_suffix = f"_pos{args.pos_scale:.3f}_neg{neg_scale_used:.3f}"

    # Output paths
    if args.output is None:
        # Default: save into a timestamped analysis folder next to this script.
        out_root = Path(__file__).resolve().parent
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"analysis_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = args.segment.with_suffix("").name
        out_path = out_dir / f"{base_name}_mean_5ms_before_after{weight_suffix}.png"
        cumsum_path = out_dir / f"{base_name}_mean_5ms_cumsum_after_comp{weight_suffix}.png"
    else:
        out_path = args.output
        cumsum_path = args.output.with_name(
            args.output.stem + f"_cumsum{weight_suffix}" + args.output.suffix
        )
        out_dir = out_path.parent

    fig.savefig(out_path, dpi=300)
    print("Saved", out_path)

    # Second plot: cumulative sum of compensated net rate vs wavelength
    if wl_comp.size > 0 and net_comp.size > 0:
        cumsum = np.cumsum(net_comp)
        # Normalise for easier visual comparison
        if np.max(np.abs(cumsum)) > 0:
            cumsum = cumsum / np.max(np.abs(cumsum))

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(wl_comp, cumsum, color="tab:blue", label="Cumulative (compensated)")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Normalised cumulative net rate")
        ax2.set_title(f"Cumulative event rate vs wavelength\n{args.segment.name}")
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(cumsum_path, dpi=300)
        print("Saved", cumsum_path)

        # Save weights used for this run
        weights_txt = out_dir / "weights_used.txt"
        with weights_txt.open("w", encoding="utf-8") as f:
            f.write(f"segment: {args.segment}\n")
            f.write(f"mapping_json: {args.mapping_json}\n")
            f.write(f"bin_ms: {args.bin_ms}\n")
            f.write(f"pos_scale: {args.pos_scale:.6f}\n")
            f.write(f"neg_scale_auto: {neg_scale_used:.6f}\n")
            f.write(f"auto_neg_min: {args.auto_neg_min:.6f}\n")
            f.write(f"auto_neg_max: {args.auto_neg_max:.6f}\n")
            f.write(f"plateau_frac: {args.plateau_frac:.6f}\n")
        print(f"Saved weights to {weights_txt}")


if __name__ == "__main__":
    main()
