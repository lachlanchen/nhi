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
    # Fallback: any learned params in the directory
    any_params = find_latest(str(d / "*learned_params*.npz"))
    if any_params:
        return Path(any_params)
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
    # Load full events first to compute true time extent, then sample
    x_full, y_full, t_full, p_full = load_events(segment_npz)
    n = len(x_full)
    k = max(1, int(n * sample))
    idx = np.random.choice(n, k, replace=False)
    x, y, t, p = x_full[idx], y_full[idx], t_full[idx], p_full[idx]
    # Use full-range normalization (not sample) to match saved FIXED plot
    t0_full = float(t_full.min())
    t1_full = float(t_full.max())
    duration_us = t1_full - t0_full
    t_norm_ms = (t - t0_full) / 1000.0

    pos = p > 0
    neg = ~pos

    xs = np.linspace(0, sensor_w, 120)
    ys = np.linspace(0, sensor_h, 120)
    x_lines = compute_boundary_lines(params["a_params"], params["b_params"], duration_us, sensor_w, sensor_h, xs, "x")
    y_lines = compute_boundary_lines(params["a_params"], params["b_params"], duration_us, sensor_w, sensor_h, ys, "y")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)

    # X–T
    if np.any(pos):
        ax1.scatter(x[pos], t_norm_ms[pos], s=0.25, c="#d62728", alpha=0.5, rasterized=True)
    if np.any(neg):
        ax1.scatter(x[neg], t_norm_ms[neg], s=0.25, c="#1f77b4", alpha=0.5, rasterized=True)
    for ln in x_lines:
        ax1.plot(xs, ln / 1000.0, color="#4d4d4d", linewidth=1.0, alpha=0.9)
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Time (ms)")
    # Ensure left y ticks/labels explicitly visible
    ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax1.spines["left"].set_visible(True)
    ax1.set_ylim(0.0, duration_us / 1000.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Y–T
    if np.any(pos):
        ax2.scatter(y[pos], t_norm_ms[pos], s=0.25, c="#d62728", alpha=0.5, rasterized=True)
    if np.any(neg):
        ax2.scatter(y[neg], t_norm_ms[neg], s=0.25, c="#1f77b4", alpha=0.5, rasterized=True)
    for ln in y_lines:
        ax2.plot(ys, ln / 1000.0, color="#4d4d4d", linewidth=1.0, alpha=0.9)
    ax2.set_xlabel("Y (pixels)")
    # Share y-axis with left panel; hide label and tick labels on the right
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2.set_ylim(0.0, duration_us / 1000.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.subplots_adjust(wspace=0.06, left=0.08, right=0.99, top=0.92, bottom=0.14)
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


def render_panel_b(segments_dir: Path, base: str, out_dir: Path, *,
                   mode: str = "npz", var_bin_us: int = 5000,
                   segment_npz: Path | None = None,
                   params: dict | None = None,
                   sensor_w: int = 1280, sensor_h: int = 720) -> None:
    """
    Plot per-bin VARIANCE exactly as in the trainer panel using the saved
    time-binned NPZ (original_bin_i / compensated_bin_i). This matches the
    numbers shown in the training results figure.
    """
    if mode == "npz":
        # Use trainer-saved 50 ms (or whatever was saved) frames
        _, allbins_path = find_timebin_csv_and_npz(segments_dir, base)
        d = np.load(allbins_path, allow_pickle=False)
        orig_keys = sorted(k for k in d.keys() if k.startswith("original_bin_"))
        comp_keys = sorted(k for k in d.keys() if k.startswith("compensated_bin_"))
        if len(orig_keys) == 0 or len(orig_keys) != len(comp_keys):
            raise RuntimeError(f"Unexpected NPZ structure in {allbins_path}")
        var_orig = [float(np.var(d[ok].astype(np.float32))) for ok in orig_keys]
        var_comp = [float(np.var(d[ck].astype(np.float32))) for ck in comp_keys]
        bins = np.arange(len(var_orig))
    elif mode == "recompute":
        # Recompute from full events with chosen bin width using the trainer's model
        if segment_npz is None or params is None:
            raise ValueError("segment_npz and params are required for mode='recompute'")
        # Lazy import to avoid heavy deps unless needed
        import importlib.util
        src = Path(__file__).resolve().parent.parent / "compensate_multiwindow_train_saved_params.py"
        if not src.exists():
            # Fallback: repo root
            src = Path.cwd() / "compensate_multiwindow_train_saved_params.py"
        spec = importlib.util.spec_from_file_location("comp_mod", str(src))
        comp_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(comp_mod)  # type: ignore

        # Load events
        dseg = np.load(segment_npz)
        x = dseg["x"].astype(np.float32)
        y = dseg["y"].astype(np.float32)
        t = dseg["t"].astype(np.float32)
        p = dseg["p"].astype(np.float32)
        if p.min() >= 0 and p.max() <= 1:
            p = (p - 0.5) * 2

        # Build model and load parameters
        num_params = int(params.get("num_params", len(params["a_params"])))
        model = comp_mod.ScanCompensation(
            duration=float(t.max() - t.min()),
            num_params=num_params,
            device=comp_mod.device,
            a_fixed=True,
            b_fixed=False,
            boundary_trainable=False,
            a_default=0.0,
            b_default=-76.0,
            temperature=float(params.get("temperature", 5000.0)),
            debug=False,
        )
        model.load_parameters(params["a_params"], params["b_params"], debug=False)

        # Compute event tensors for original and compensated at requested bin width
        t0 = comp_mod.torch.tensor(float(t.min()), device=comp_mod.device, dtype=comp_mod.torch.float32)
        t1 = comp_mod.torch.tensor(float(t.max()), device=comp_mod.device, dtype=comp_mod.torch.float32)
        Eo = comp_mod.create_event_frames(model, x, y, t, p, sensor_h, sensor_w, var_bin_us, t0, t1, compensated=False)
        Ec = comp_mod.create_event_frames(model, x, y, t, p, sensor_h, sensor_w, var_bin_us, t0, t1, compensated=True)
        # Variances per bin exactly as trainer: torch.var over flattened spatial dims
        with comp_mod.torch.no_grad():
            nb, H, W = Eo.shape
            var_o_t = comp_mod.torch.var(Eo.view(nb, -1), dim=1)
            var_c_t = comp_mod.torch.var(Ec.view(nb, -1), dim=1)
            var_orig = [float(v.item()) for v in var_o_t]
            var_comp = [float(v.item()) for v in var_c_t]
        bins = np.arange(len(var_orig))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Make panel (b) ~1.5× taller than the previous compact height for readability
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    ax.plot(bins, var_orig, color="#7f7f7f", linewidth=1.4, label="Original")
    ax.plot(bins, var_comp, color="#1f77b4", linewidth=1.4, label="Compensate")
    ax.set_xlabel("Time Bin")
    # Set fixed headroom up to 1.3 and show ticks every 0.2, but omit the
    # top-most 1.3 tick label to avoid overlap with panel letter (b).
    top = 1.3 + 1e-6
    ax.set_ylim(0.0, top)
    ax.set_yticks(np.arange(0.0, 1.3, 0.2))
    ax.set_ylabel("Variance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Place legend at the top-right with adequate headroom
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    im1 = ax1.imshow(orig, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title(f"Original – Bin {idx}")
    ax1.set_xlabel("X (px)")
    ax1.set_ylabel("Y (px)")
    # Ensure left y ticks/labels explicitly visible
    ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax1.spines["left"].set_visible(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    im2 = ax2.imshow(comp, cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(f"Compensated – Bin {idx}")
    ax2.set_xlabel("X (px)")
    # Hide right panel's y-label and tick labels (shared y-axis)
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Shared colorbar at the right side (outside of image area)
    fig.subplots_adjust(right=0.86, left=0.08, top=0.94, bottom=0.12, wspace=0.06)
    cax = fig.add_axes([0.88, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.ax.set_ylabel("Value", rotation=90)

    out_path = out_dir / "figure03_c_bin50ms.pdf"
    fig.savefig(out_path, dpi=400)
    fig.savefig(out_dir / "figure03_c_bin50ms.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3: multi-window compensation panels")
    parser.add_argument(
        "segment_npz",
        type=Path,
        help="Path to Scan_*_events.npz OR a dataset directory (the script will search recursively for a forward segment)",
    )
    parser.add_argument("--sensor_width", type=int, default=1280)
    parser.add_argument("--sensor_height", type=int, default=720)
    parser.add_argument("--variance_mode", choices=["npz", "recompute"], default="recompute",
                        help="Source for panel (b) variance: 'npz' uses saved time_binned_frames; 'recompute' rebuilds from full events")
    parser.add_argument("--var_bin_us", type=int, default=5000,
                        help="Bin width in microseconds for variance when variance_mode=recompute (default: 5000us = 5ms)")
    parser.add_argument("--sample", type=float, default=0.05, help="Event sampling fraction for panel (a)")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()

    setup_style()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow passing a dataset directory; search for a forward segment
    input_path = args.segment_npz.resolve()
    if input_path.is_dir():
        candidates = sorted(
            input_path.rglob("*_segments/Scan_*_Forward_events.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No forward segment NPZ found under {input_path}")
        # Prefer candidates that have time-binned CSV/NPZ available
        chosen = None
        for cand in candidates:
            seg_dir = cand.parent
            base_try = cand.stem
            try:
                find_timebin_csv_and_npz(seg_dir, base_try)
                chosen = cand
                break
            except Exception:
                continue
        if chosen is None:
            chosen = candidates[0]
        print(f"Using forward segment: {chosen}")
        segment_path = chosen
    else:
        segment_path = input_path

    segments_dir = segment_path.parent
    base = segment_path.stem
    param_path = find_param_file(segment_path)
    params = load_params_npz(param_path)

    render_panel_a(segment_path, params, args.sensor_width, args.sensor_height, args.sample, out_dir)
    render_panel_b(
        segments_dir,
        base,
        out_dir,
        mode=args.variance_mode,
        var_bin_us=args.var_bin_us,
        segment_npz=segment_path,
        params=params,
        sensor_w=args.sensor_width,
        sensor_h=args.sensor_height,
    )
    render_panel_c(segments_dir, base, out_dir)


if __name__ == "__main__":
    main()
