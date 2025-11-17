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
    overlay_alpha: float = 0.85,
    overlay_span: str = "axis",
    overlay_flipud: bool = False,
    overlay_stride: int = 8,
    fixed_xlim: Optional[Tuple[float, float]] = None,
    fixed_ylim: Optional[Tuple[float, float]] = None,
    fixed_zlim: Optional[Tuple[float, float]] = None,
):
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
    # Tight axis limits with small margins to reduce whitespace
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    t_min, t_max = float(t_ms.min()), float(t_ms.max())
    pad_x = 0.02 * (x_max - x_min + 1)
    pad_y = 0.02 * (y_max - y_min + 1)
    pad_t = 0.02 * (t_max - t_min + 1e-6)
    xlo, xhi = (x_min - pad_x, x_max + pad_x) if fixed_xlim is None else fixed_xlim
    zlo, zhi = (y_min - pad_y, y_max + pad_y) if fixed_zlim is None else fixed_zlim
    ylo, yhi = (time_scale * (t_min - pad_t), time_scale * (t_max + pad_t)) if fixed_ylim is None else fixed_ylim
    ax.set_xlim(xlo, xhi)
    ax.set_zlim(zlo, zhi)
    ax.set_ylim(ylo, yhi)

    ax.set_xlabel("X (px)", labelpad=1)
    ax.set_ylabel("Time (ms)", labelpad=1)
    ax.set_zlabel("Y (px)", labelpad=1)
    ax.set_title(title, pad=2)
    # View with time on Y, spatial on X/Z; gentle perspective from front-left
    ax.view_init(elev=25, azim=-35)
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

    # Optional overlay plane of an image at a fixed time (in ms)
    if overlay_img is not None and overlay_time_ms is not None:
        img = overlay_img
        if overlay_flipud:
            img = np.flipud(img)
        H, W = img.shape[:2]
        # Extents either from axis limits (keeps crop) or event extents
        if overlay_span == "events":
            x0, x1 = float(x.min()), float(x.max())
            z0, z1 = float(y.min()), float(y.max())
        else:
            x0, x1 = ax.get_xlim()
            z0, z1 = ax.get_zlim()
        s = max(1, int(overlay_stride))
        xs = np.linspace(x0, x1, W)[::s]
        zs = np.linspace(z0, z1, H)[::s]
        Xg, Zg = np.meshgrid(xs, zs)
        Yg = np.full_like(Xg, time_scale * overlay_time_ms)
        # Build facecolors
        fc = img.astype(np.float32)[::s, ::s, ...] if img.ndim == 3 else img.astype(np.float32)[::s, ::s]
        if fc.max() > 1.0:
            fc = fc / 255.0
        if fc.ndim == 2:
            # grayscale: map to RGBA magma
            cm = plt.get_cmap("magma")
            fc = cm((fc - fc.min()) / (fc.max() - fc.min() + 1e-12))
        if fc.shape[-1] == 3:
            alpha = np.full(fc.shape[:2] + (1,), overlay_alpha, dtype=fc.dtype)
            fc = np.concatenate([fc, alpha], axis=-1)
        else:
            fc[..., -1] = fc[..., -1] * overlay_alpha
        surf = ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1, facecolors=fc, shade=False, linewidth=0, antialiased=False)
        try:
            surf.set_rasterized(True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="3D event clouds before/after compensation (optional image overlay)")
    ap.add_argument("segment_npz", type=Path, help="Path to Scan_*_events.npz")
    ap.add_argument("--sample", type=float, default=0.02, help="Fraction of events to plot (default: 0.02)")
    ap.add_argument("--time-scale", type=float, default=1.5, help="Stretch factor for time axis (default: 1.5)")
    ap.add_argument("--chunk-size", type=int, default=400000, help="Chunk size for compensation (default: 400000)")
    ap.add_argument("--output-dir", type=Path, default=Path("publication_code/figures"), help="Output directory")
    # Overlay options (using prepared plain images)
    ap.add_argument("--overlay-image-before", type=Path, default=None, help="PNG/JPG to overlay on BEFORE cloud")
    ap.add_argument("--overlay-image-after", type=Path, default=None, help="PNG/JPG to overlay on AFTER cloud")
    ap.add_argument("--overlay-time-ms", type=float, default=None, help="Time (ms) to place overlay plane (e.g., 50)")
    ap.add_argument("--overlay-alpha", type=float, default=1.0, help="Overlay plane alpha (default 1.0)")
    ap.add_argument("--overlay-span", choices=["axis", "events"], default="axis", help="Map image over axis or event extents")
    ap.add_argument("--overlay-flipud", action="store_true", help="Flip overlay vertically to match event Y")
    ap.add_argument("--overlay-stride", type=int, default=8, help="Downsample factor for overlay plane to keep PDF lightweight (default 8)")
    ap.add_argument("--lock-time-axis", action="store_true", default=True, help="Use identical time axis limits for before/after panels")
    ap.add_argument("--lock-spatial-axis", action="store_true", default=True, help="Use identical X/Z limits for before/after panels")
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

    # Load overlay images if provided
    def _read_img(pth: Optional[Path]):
        if pth is None:
            return None
        import matplotlib.image as mpimg
        if not pth.exists():
            print(f"[warn] overlay image not found: {pth}")
            return None
        img = mpimg.imread(pth)
        return img

    ov_before = _read_img(args.overlay_image_before)
    ov_after = _read_img(args.overlay_image_after)
    ov_t_ms: Optional[float] = None
    if (ov_before is not None or ov_after is not None):
        ov_t_ms = args.overlay_time_ms if args.overlay_time_ms is not None else 50.0

    # Compute fixed axes if requested
    fixed_xlim = fixed_zlim = fixed_ylim = None
    if args.lock_spatial_axis:
        x_min = float(min(xs.min(), xs_w.min()))
        x_max = float(max(xs.max(), xs_w.max()))
        y_min = float(min(ys.min(), ys_w.min()))
        y_max = float(max(ys.max(), ys_w.max()))
        pad_x = 0.02 * (x_max - x_min + 1)
        pad_y = 0.02 * (y_max - y_min + 1)
        fixed_xlim = (x_min - pad_x, x_max + pad_x)
        fixed_zlim = (y_min - pad_y, y_max + pad_y)
    if args.lock_time_axis:
        tmin = float(min(ts_ms.min(), ts_w_ms.min()))
        tmax = float(max(ts_ms.max(), ts_w_ms.max()))
        pad_t = 0.02 * (tmax - tmin + 1e-6)
        fixed_ylim = (args.time_scale * (tmin - pad_t), args.time_scale * (tmax + pad_t))

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
        overlay_img=ov_before,
        overlay_time_ms=ov_t_ms,
        overlay_alpha=args.overlay_alpha,
        overlay_span=args.overlay_span,
        overlay_flipud=args.overlay_flipud,
        overlay_stride=args.overlay_stride,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
        fixed_zlim=fixed_zlim,
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
        overlay_img=ov_after,
        overlay_time_ms=ov_t_ms,
        overlay_alpha=args.overlay_alpha,
        overlay_span=args.overlay_span,
        overlay_flipud=args.overlay_flipud,
        overlay_stride=args.overlay_stride,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
        fixed_zlim=fixed_zlim,
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
