#!/usr/bin/env python3
"""
Plot per-bin mean (net event rate) before and after compensation (5 ms bins).

The wavelength axis is obtained from a JSON mapping produced by
`compare_publication_three_panel.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import visualize_cumulative_weighted as vcw  # type: ignore


def bin_net_rate(times_us: np.ndarray, p: np.ndarray, bin_ms: float, hw: float):
    """Return (time_ms, net_rate) for given timestamps and polarities."""
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

    net_rate = (sums_pos - sums_neg) / hw
    time_ms = edges_ms - edges_ms[0]
    return time_ms, net_rate


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

    # Before compensation: use raw timestamps t.
    t_raw_ms, net_raw = bin_net_rate(t, p, bin_ms=args.bin_ms, hw=hw)
    wl_raw = slope * t_raw_ms + intercept

    # After compensation: use fast compensated times (if params found).
    param_file = vcw.find_param_file_for_segment(str(args.segment))
    if param_file is not None:
        params = vcw.load_parameters_from_npz(param_file)
        t_comp, *_ = vcw.compute_fast_compensated_times(
            x, y, t, params["a_params"], params["b_params"]
        )
        t_comp_ms, net_comp = bin_net_rate(t_comp, p, bin_ms=args.bin_ms, hw=hw)
        wl_comp = slope * t_comp_ms + intercept
    else:
        wl_comp = wl_raw
        net_comp = net_raw

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_raw, net_raw, color="tab:gray", alpha=0.6, label="Raw time (before comp)")
    ax.plot(wl_comp, net_comp, color="tab:green", label="Compensated time")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Per-pixel net event rate (5 ms bins)")
    ax.set_title(f"Mean event rate vs wavelength\n{args.segment.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output is None:
        base = args.segment.with_suffix("")
        out_path = base.with_name(base.name + "_mean_5ms_before_after.png")
    else:
        out_path = args.output
    fig.savefig(out_path, dpi=300)
    print("Saved", out_path)


if __name__ == "__main__":
    main()

