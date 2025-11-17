#!/usr/bin/env python3
"""
Render 3D event clouds before and after multi-window compensation.

Inputs:
  - Segment NPZ (Scan_*_events.npz)
  - Learned params NPZ (auto-discovered next to the segment)

Outputs (timestamped folder under publication_code/figures/):
  - event_cloud_before_after.pdf/.png

Notes:
  - Time is stretched and placed on the Z-axis; view is tilted so the cloud runs
    from south-west to north-east.
  - Points are rasterized to keep PDFs lightweight.
"""

from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compensate_multiwindow_train_saved_params import Compensate


def find_param_file(segment_npz: Path) -> Path:
    """Locate the learned params NPZ next to the segment."""
    d = segment_npz.parent
    base = segment_npz.stem
    patterns = [
        f"{base}_chunked_processing_learned_params_*.npz",
        f"{base}_learned_params_*.npz",
        f"{base}*learned_params*.npz",
    ]
    for pat in patterns:
        candidates = sorted(d.glob(pat))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    candidates = sorted(d.glob("*learned_params*.npz"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No learned params NPZ found next to {segment_npz}")


def load_params_npz(param_path: Path) -> dict:
    z = np.load(param_path)
    return {
        "a_params": z["a_params"].astype(np.float32),
        "b_params": z["b_params"].astype(np.float32),
        "num_params": int(z["num_params"]),
        "temperature": float(z.get("temperature", 5000.0)),
    }


def load_events(segment_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(segment_npz)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)
    t = d["t"].astype(np.float32)
    p = d["p"].astype(np.float32)
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2  # map {0,1}->{-1,1}
    return x, y, t, p


def compensate_times(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, params: dict, chunk_size: int = 500000
) -> np.ndarray:
    """Compute compensated timestamps t' = t - compensation."""
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


def sample_events(x, y, t, p, fraction: float, max_points: int = 400_000):
    n = len(x)
    k = min(int(n * fraction), max_points)
    idx = random.sample(range(n), k) if k < n else list(range(n))
    idx = np.array(idx, dtype=np.int64)
    return x[idx], y[idx], t[idx], p[idx]


def plot_cloud(ax, x, y, t_ms, p, title: str, time_scale: float):
    pos = p > 0
    neg = p <= 0
    colors = {"pos": "#ffbb78", "neg": "#aec7e8"}
    ax.scatter(
        x[pos],
        time_scale * t_ms[pos],
        y[pos],
        c=colors["pos"],
        s=0.1,
        alpha=0.6,
        marker=".",
        rasterized=True,
    )
    ax.scatter(
        x[neg],
        time_scale * t_ms[neg],
        y[neg],
        c=colors["neg"],
        s=0.1,
        alpha=0.6,
        marker=".",
        rasterized=True,
    )
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Time (ms)")
    ax.set_zlabel("Y (px)")
    ax.set_title(title)
    # View with time on Y, spatial on X/Z; tilt to see all three axes
    ax.view_init(elev=15, azim=-60)
    ax.set_box_aspect([1, 0.6 * time_scale, 1])
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def main():
    ap = argparse.ArgumentParser(description="3D event clouds before/after compensation")
    ap.add_argument("segment_npz", type=Path, help="Path to Scan_*_events.npz")
    ap.add_argument("--sample", type=float, default=0.02, help="Fraction of events to plot (default: 0.02)")
    ap.add_argument("--time-scale", type=float, default=1.5, help="Stretch factor for time axis (default: 1.5)")
    ap.add_argument("--chunk-size", type=int, default=400000, help="Chunk size for compensation (default: 400000)")
    ap.add_argument("--output-dir", type=Path, default=Path("publication_code/figures"), help="Output directory")
    args = ap.parse_args()

    x, y, t, p = load_events(args.segment_npz)
    param_path = find_param_file(args.segment_npz)
    params = load_params_npz(param_path)

    t_warp = compensate_times(x, y, t, params, chunk_size=args.chunk_size)

    # Sample for plotting
    xs, ys, ts, ps = sample_events(x, y, t - t.min(), p, args.sample)
    xs_w, ys_w, ts_w, ps_w = sample_events(x, y, t_warp, p, args.sample)

    ts_ms = ts / 1000.0
    ts_w_ms = ts_w / 1000.0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"multiwindow_compensation_figure_3d_cloud_with_image_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_cloud(ax1, xs, ys, ts_ms, ps, "Before compensation", args.time_scale)
    plot_cloud(ax2, xs_w, ys_w, ts_w_ms, ps_w, "After compensation", args.time_scale)
    fig.tight_layout()
    fig.savefig(out_dir / "event_cloud_before_after.pdf", dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / "event_cloud_before_after.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved 3D clouds to {out_dir}")


if __name__ == "__main__":
    main()
