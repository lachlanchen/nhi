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
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
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


def find_timebin_npz(segment_npz: Path) -> Optional[Path]:
    """Locate the trainer-saved time-binned NPZ next to the segment.

    Looks in `<segment_dir>/time_binned_frames/` for
    `Scan_*_chunked_processing_all_time_bins_data_*.npz` and returns the newest.
    """
    seg_dir = segment_npz.parent
    tb_dir = seg_dir / "time_binned_frames"
    if not tb_dir.exists():
        return None
    base = segment_npz.stem
    patterns = [
        f"{base}_chunked_processing_all_time_bins_data_*.npz",
        f"{base}*all_time_bins_data*.npz",
        "*_all_time_bins_data_*.npz",
    ]
    cands = []
    for pat in patterns:
        cands.extend(sorted(tb_dir.glob(pat)))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


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


def plot_cloud(
    ax,
    x,
    y,
    t_ms,
    p,
    title: str,
    time_scale: float,
    *,
    overlay_img: Optional[np.ndarray] = None,
    overlay_time_ms: Optional[float] = None,
    overlay_cmap: str = "magma",
    overlay_alpha: float = 0.75,
    overlay_stride: int = 6,
    overlay_flipud: bool = False,
    overlay_span: str = "axis",
    view_elev: float = 18.0,
    view_azim: float = -30.0,
):
    pos = p > 0
    neg = p <= 0
    colors = {"pos": "#ffbb78", "neg": "#aec7e8"}
    # Compute tight extents first (so overlay can span full X/Z area)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    t_min, t_max = float(t_ms.min()), float(t_ms.max())
    pad_x = 0.02 * (x_max - x_min + 1)
    pad_y = 0.02 * (y_max - y_min + 1)
    pad_t = 0.02 * (t_max - t_min + 1e-6)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_zlim(y_min - pad_y, y_max + pad_y)
    ax.set_ylim(time_scale * (t_min - pad_t), time_scale * (t_max + pad_t))

    ax.set_xlabel("X (px)", labelpad=1)
    ax.set_ylabel("Time (ms)", labelpad=1)
    ax.set_zlabel("Y (px)", labelpad=1)
    ax.set_title(title, pad=2)
    # View with time on Y, spatial on X/Z; match no-image defaults
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_box_aspect([1, 0.9 * time_scale, 1])
    # Stretch time dimension (Y) similar to legacy EVK visualizer
    x_scale, y_scale, z_scale = 1.0, 1.6, 1.0
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale / scale.max()
    scale[3, 3] = 1.0
    orig_proj = ax.get_proj

    def short_proj():
        return np.dot(orig_proj(), scale)

    ax.get_proj = short_proj
    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    # Optional overlay plane at a given time (Y axis)
    if overlay_img is not None and overlay_time_ms is not None:
        img = overlay_img
        if overlay_flipud:
            img = np.flipud(img)
        H, W = img.shape[:2]
        # Map image extents
        if overlay_span == 'events':
            # Event extents (no pad)
            x0, x1 = float(x_min), float(x_max)
            z0, z1 = float(y_min), float(y_max)
        else:
            # Current axis limits (includes small pad) — safer for consistent cropping
            x0, x1 = ax.get_xlim()
            z0, z1 = ax.get_zlim()
        xs = np.linspace(x0, x1, W)
        zs = np.linspace(z0, z1, H)
        Xg, Zg = np.meshgrid(xs, zs)
        s = max(1, int(overlay_stride))
        Xg_s = Xg[::s, ::s]
        Zg_s = Zg[::s, ::s]
        Yg_s = np.full_like(Xg_s, time_scale * overlay_time_ms)
        img_s = img[::s, ::s].astype(np.float32)
        vmin = float(np.percentile(img_s, 1.0))
        vmax = float(np.percentile(img_s, 99.0))
        if np.isclose(vmin, vmax):
            vmin = float(np.min(img_s))
            vmax = float(np.max(img_s) + 1e-6)
        norm = np.clip((img_s - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
        cmap = plt.get_cmap(overlay_cmap)
        facecolors = cmap(norm)
        facecolors[..., -1] = overlay_alpha
        surf = ax.plot_surface(
            Xg_s,
            Yg_s,
            Zg_s,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            shade=False,
            linewidth=0,
            antialiased=False,
        )
        # Prefer front-to-back sorting to keep plane visually behind points
        try:
            surf.set_zsort('min')
        except Exception:
            pass

    # Now draw the event points on top for clarity
    ax.scatter(
        x[pos],
        time_scale * t_ms[pos],
        y[pos],
        c=colors["pos"],
        s=0.1,
        alpha=0.55,
        marker=".",
        rasterized=True,
        depthshade=False,
    )
    ax.scatter(
        x[neg],
        time_scale * t_ms[neg],
        y[neg],
        c=colors["neg"],
        s=0.1,
        alpha=0.55,
        marker=".",
        rasterized=True,
        depthshade=False,
    )


def main():
    ap = argparse.ArgumentParser(description="3D event clouds before/after compensation")
    ap.add_argument("segment_npz", type=Path, help="Path to Scan_*_events.npz")
    ap.add_argument("--sample", type=float, default=0.02, help="Fraction of events to plot (default: 0.02)")
    ap.add_argument("--time-scale", type=float, default=1.5, help="Stretch factor for time axis (default: 1.5)")
    ap.add_argument("--chunk-size", type=int, default=400000, help="Chunk size for compensation (default: 400000)")
    # Overlay options
    ap.add_argument("--overlay", action="store_true", help="Overlay a time-bin image plane onto the 3D cloud")
    ap.add_argument("--overlay-from-segment", type=Path, default=None,
                    help="Optional: path to a different Scan_*_events.npz whose time_binned_frames will supply the overlay image")
    ap.add_argument("--overlay-bins-npz", type=Path, default=None,
                    help="Optional: direct path to a *_all_time_bins_data_*.npz to supply the overlay image (takes precedence)")
    ap.add_argument("--overlay-bin-index", type=int, default=18, help="Time-bin index to overlay (default: 18)")
    ap.add_argument("--overlay-bin-us", type=int, default=50000, help="Bin width in microseconds (default: 50000 = 50ms)")
    ap.add_argument("--overlay-alpha", type=float, default=0.75, help="Overlay plane alpha (default: 0.75)")
    ap.add_argument("--overlay-cmap", type=str, default="magma", help="Overlay colormap (default: magma)")
    ap.add_argument("--overlay-stride", type=int, default=6, help="Downsample stride for overlay plane (default: 6)")
    # Alias: --plot-image-time for readability in figure scripts
    ap.add_argument("--overlay-time-ms", "--plot-image-time", dest="overlay_time_ms", type=float, default=None,
                    help="Place the image plane at this time (ms). Useful to show a bin image at a different visual time.")
    ap.add_argument("--overlay-time-offset-ms", type=float, default=0.0,
                    help="Optional small offset added to the overlay plane time (ms) to reduce visual occlusion.")
    ap.add_argument("--output-dir", type=Path, default=Path("publication_code/figures"), help="Output directory")
    ap.add_argument("--view-elev", type=float, default=18.0, help="3D view elevation (default: 18)")
    ap.add_argument("--view-azim", type=float, default=-30.0, help="3D view azimuth (default: -30)")
    ap.add_argument("--overlay-span", choices=["axis", "events"], default="axis",
                    help="Span overlay plane across current axis limits (axis) or event extents (events). Default: axis")
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

    # Prepare optional overlay planes
    overlay_before = None
    overlay_after = None
    overlay_t_ms: Optional[float] = None
    if args.overlay:
        tb_npz: Optional[Path]
        if args.overlay_bins_npz is not None:
            tb_npz = args.overlay_bins_npz
        elif args.overlay_from_segment is not None:
            tb_npz = find_timebin_npz(args.overlay_from_segment)
        else:
            tb_npz = find_timebin_npz(args.segment_npz)
        if tb_npz is not None:
            with np.load(tb_npz, allow_pickle=False) as d:
                ob_key = f"original_bin_{args.overlay_bin_index}"
                cb_key = f"compensated_bin_{args.overlay_bin_index}"
                if ob_key in d and cb_key in d:
                    overlay_before = d[ob_key]
                    overlay_after = d[cb_key]
                    if args.overlay_time_ms is not None:
                        overlay_t_ms = float(args.overlay_time_ms)
                    else:
                        overlay_t_ms = (args.overlay_bin_index + 0.5) * (args.overlay_bin_us / 1000.0)
                    overlay_t_ms = overlay_t_ms + float(args.overlay_time_offset_ms)
                else:
                    print(f"[warn] overlay bin keys not found in {tb_npz}: {ob_key}, {cb_key}")
        else:
            print("[warn] no time-binned NPZ found; skipping overlay")

    # Save panels separately to avoid overflow
    fig1 = plt.figure(figsize=(4.4, 3.3))
    ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
    plot_cloud(
        ax1,
        xs,
        ys,
        ts_ms,
        ps,
        "",
        args.time_scale,
        overlay_img=overlay_before,
        overlay_time_ms=overlay_t_ms,
        overlay_cmap=args.overlay_cmap,
        overlay_alpha=args.overlay_alpha,
        overlay_stride=args.overlay_stride,
        overlay_flipud=False,
        overlay_span=args.overlay_span,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_tight_3d(fig1, ax1, out_dir / "event_cloud_before.pdf", dpi=400, pad_inches=0.0, extra_pad=0.01)
    _save_tight_3d(fig1, ax1, out_dir / "event_cloud_before.png", dpi=300, pad_inches=0.0, extra_pad=0.01)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(4.4, 3.3))
    ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
    plot_cloud(
        ax2,
        xs_w,
        ys_w,
        ts_w_ms,
        ps_w,
        "",
        args.time_scale,
        overlay_img=overlay_after,
        overlay_time_ms=overlay_t_ms,
        overlay_cmap=args.overlay_cmap,
        overlay_alpha=args.overlay_alpha,
        overlay_stride=args.overlay_stride,
        overlay_flipud=False,
        overlay_span=args.overlay_span,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_tight_3d(fig2, ax2, out_dir / "event_cloud_after.pdf", dpi=400, pad_inches=0.0, extra_pad=0.01)
    _save_tight_3d(fig2, ax2, out_dir / "event_cloud_after.png", dpi=300, pad_inches=0.0, extra_pad=0.01)
    plt.close(fig2)

    print(f"Saved 3D clouds to {out_dir}")


def _save_tight_3d(fig: plt.Figure, ax: plt.Axes, path: Path, dpi: int = 300, pad_inches: float = 0.0, extra_pad: float = 0.01):
    """Save a 3D axes figure tightly cropped.

    For reliability across backends (and to avoid pathological bbox from 3D),
    save with bbox_inches='tight' then post-crop PDFs with `pdfcrop`.
    PNGs are saved with bbox_inches='tight'.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        tmp = path.with_suffix(".uncropped.pdf")
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        # Post-crop using pdfcrop (present with TeX Live). Keep a tiny margin to avoid clipping.
        import shutil, subprocess
        pdfcrop = shutil.which("pdfcrop")
        if pdfcrop is not None:
            try:
                subprocess.run([pdfcrop, "--margins", "1", str(tmp), str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                tmp.unlink(missing_ok=True)
            except Exception:
                # If cropping fails, fall back to the uncropped file
                tmp.rename(path)
        else:
            tmp.rename(path)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)


if __name__ == "__main__":
    main()
