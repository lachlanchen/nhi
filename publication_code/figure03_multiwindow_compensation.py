#!/usr/bin/env python3
"""
Generate publication-ready Figure 3 panels for multi-window compensation:

Panels (saved separately):
  - figure03_a_events.pdf      – X–T and Y–T projections with learned boundaries
  - figure03_b_variance.pdf    – Variance per 50 ms bin: original vs compensated
  - figure03_c_bin50ms.pdf     – Single-bin comparison (original vs compensated)

Inputs:
  - Path to a segmented events NPZ (e.g., Scan_1_Forward_events.npz)
  - Auto-discovers the matching learned-params NPZ and time-binned NPZ/CSV

Style:
  - Clean, consistent axes; minimal grid; standard labels
  - Panel markers (a)/(b)/(c) at the figure’s top-left corner
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": True,
        }
    )


def find_latest(path_glob: str) -> str | None:
    files = glob.glob(path_glob)
    return max(files, key=os.path.getmtime) if files else None


def find_param_file(segment_npz: Path) -> Path:
    d = segment_npz.parent
    base = segment_npz.stem
    patterns = [
        f"{base}_chunked_processing_learned_params_*.npz",
        f"{base}_learned_params_*.npz",
        f"{base}*learned_params*.npz",
    ]
    for pat in patterns:
        p = find_latest(str(d / pat))
        if p:
            return Path(p)
    raise FileNotFoundError("Could not locate learned params NPZ next to segment file.")


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
        p = (p - 0.5) * 2
    return x, y, t, p


def compute_boundary_lines(a_params: np.ndarray, b_params: np.ndarray, duration_us: float, sensor_w: int, sensor_h: int,
                           coord_range: np.ndarray, mode: str) -> List[np.ndarray]:
    num_params = len(a_params)
    main_windows = num_params - 3
    main_size = duration_us / max(1, main_windows)
    offsets = np.array([(i - 1) * main_size for i in range(num_params)], dtype=np.float32)
    lines: List[np.ndarray] = []
    if mode == "x":
        y_c = sensor_h / 2.0
        for i in range(num_params):
            lines.append(a_params[i] * coord_range + b_params[i] * y_c + offsets[i])
    else:
        x_c = sensor_w / 2.0
        for i in range(num_params):
            lines.append(a_params[i] * x_c + b_params[i] * coord_range + offsets[i])
    return lines


def render_panel_a(segment_npz: Path, params: dict, sensor_w: int, sensor_h: int, sample: float, out_dir: Path) -> None:
    x, y, t, p = load_events(segment_npz)
    n = len(x)
    k = max(1, int(n * sample))
    idx = np.random.choice(n, k, replace=False)
    x, y, t, p = x[idx], y[idx], t[idx], p[idx]
    t0 = float(t.min())
    t_norm_ms = (t - t0) / 1000.0
    duration_us = float(t.max() - t.min())

    pos = p > 0
    neg = ~pos

    xs = np.linspace(0, sensor_w, 120)
    ys = np.linspace(0, sensor_h, 120)
    x_lines = compute_boundary_lines(params["a_params"], params["b_params"], duration_us, sensor_w, sensor_h, xs, "x")
    y_lines = compute_boundary_lines(params["a_params"], params["b_params"], duration_us, sensor_w, sensor_h, ys, "y")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    # X–T
    if np.any(pos):
        ax1.scatter(x[pos], t_norm_ms[pos], s=0.3, c="#d62728", alpha=0.6, rasterized=True)
    if np.any(neg):
        ax1.scatter(x[neg], t_norm_ms[neg], s=0.3, c="#1f77b4", alpha=0.6, rasterized=True)
    for ln in x_lines:
        ax1.plot(xs, ln / 1000.0, color="#4d4d4d", linewidth=1.0, alpha=0.9)
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_ylim(0.0, duration_us / 1000.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Y–T
    if np.any(pos):
        ax2.scatter(y[pos], t_norm_ms[pos], s=0.3, c="#d62728", alpha=0.6, rasterized=True)
    if np.any(neg):
        ax2.scatter(y[neg], t_norm_ms[neg], s=0.3, c="#1f77b4", alpha=0.6, rasterized=True)
    for ln in y_lines:
        ax2.plot(ys, ln / 1000.0, color="#4d4d4d", linewidth=1.0, alpha=0.9)
    ax2.set_xlabel("Y (pixels)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_ylim(0.0, duration_us / 1000.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(0.012, 0.985, "(a)", fontweight="bold", ha="left", va="top")
    fig.tight_layout()
    out_path = out_dir / "figure03_a_events.pdf"
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_dir / "figure03_a_events.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def find_timebin_csv_and_npz(segments_dir: Path, base: str) -> Tuple[Path, Path]:
    tb_dir = segments_dir / "time_binned_frames"
    csv = find_latest(str(tb_dir / f"{base}_chunked_processing_time_bin_statistics_*.csv"))
    npz = find_latest(str(tb_dir / f"{base}_chunked_processing_all_time_bins_data_*.npz"))
    if not (csv and npz):
        raise FileNotFoundError("Could not find time-binned CSV/NPZ in time_binned_frames.")
    return Path(csv), Path(npz)


def render_panel_b(segments_dir: Path, base: str, out_dir: Path) -> None:
    csv_path, _ = find_timebin_csv_and_npz(segments_dir, base)
    # Read CSV with comments prefixed by '#'
    # Robust read: seek the real header row starting with 'bin_idx,'
    lines = [ln.strip() for ln in open(csv_path, "r").read().splitlines()]
    try:
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("bin_idx,"))
    except StopIteration:
        raise RuntimeError(f"Could not find header row in {csv_path}")
    clean = "\n".join(lines[header_idx:])
    from io import StringIO
    import pandas as pd
    df = pd.read_csv(StringIO(clean))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(df["time_ms"], df["orig_std"], color="#7f7f7f", linewidth=1.4, label="Original")
    ax.plot(df["time_ms"], df["comp_std"], color="#1f77b4", linewidth=1.4, label="Compensated")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Variance (std)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.012, 0.985, "(b)", fontweight="bold", ha="left", va="top")
    fig.tight_layout()
    out_path = out_dir / "figure03_b_variance.pdf"
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_dir / "figure03_b_variance.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def render_panel_c(segments_dir: Path, base: str, out_dir: Path, choose: str = "best") -> None:
    # Use the aggregated 50 ms bin NPZ for clean single-bin images
    _, allbins_path = find_timebin_csv_and_npz(segments_dir, base)
    d = np.load(allbins_path, allow_pickle=False)
    # Heuristic: choose bin with max (orig_std - comp_std) if stats CSV present; otherwise pick middle
    csv_path = find_latest(str(segments_dir / "time_binned_frames" / f"{base}_chunked_processing_time_bin_statistics_*.csv"))
    if csv_path:
        import pandas as pd
        from io import StringIO
        lines = [ln.strip() for ln in open(csv_path, "r").read().splitlines()]
        try:
            header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("bin_idx,"))
        except StopIteration:
            header_idx = None
        if header_idx is not None:
            df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
            idx = int((df["orig_std"] - df["comp_std"]).idxmax())
        else:
            idx = 12
    else:
        idx = 12

    orig = d[f"original_bin_{idx}"]
    comp = d[f"compensated_bin_{idx}"]

    vmin = float(min(orig.min(), comp.min()))
    vmax = float(max(orig.max(), comp.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    im1 = ax1.imshow(orig, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title(f"Original – Bin {idx}")
    ax1.set_xlabel("X (px)")
    ax1.set_ylabel("Y (px)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    im2 = ax2.imshow(comp, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(f"Compensated – Bin {idx}")
    ax2.set_xlabel("X (px)")
    ax2.set_ylabel("Y (px)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Value", rotation=90)

    fig.text(0.012, 0.985, "(c)", fontweight="bold", ha="left", va="top")
    fig.tight_layout()
    out_path = out_dir / "figure03_c_bin50ms.pdf"
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_dir / "figure03_c_bin50ms.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3: multi-window compensation panels")
    parser.add_argument("segment_npz", type=Path, help="Path to Scan_*_events.npz segment file")
    parser.add_argument("--sensor_width", type=int, default=1280)
    parser.add_argument("--sensor_height", type=int, default=720)
    parser.add_argument("--sample", type=float, default=0.10, help="Event sampling fraction for panel (a)")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()

    setup_style()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = args.segment_npz.parent
    base = args.segment_npz.stem
    param_path = find_param_file(args.segment_npz)
    params = load_params_npz(param_path)

    render_panel_a(args.segment_npz, params, args.sensor_width, args.sensor_height, args.sample, out_dir)
    render_panel_b(segments_dir, base, out_dir)
    render_panel_c(segments_dir, base, out_dir)


if __name__ == "__main__":
    main()
