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
import matplotlib.patheffects as pe


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11.7,
            "axes.labelsize": 11.7,
            "axes.titlesize": 11.7,
            "axes.linewidth": 1.04,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.9,
            "ytick.major.size": 3.9,
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


def render_panel_a(segment_npz: Path, params: dict, sensor_w: int, sensor_h: int, sample: float, out_dir: Path, suffix: str = "") -> None:
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

    # Use constrained_layout so labels (e.g., 'Time (ms)') are fully visible
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True, constrained_layout=True)
    label_font = 16
    tick_font = 12

    # X–T (align with Fig. 4 but use less-saturated tints)
    # Light tints matching Tableau family: light blue / light orange
    POS_COLOR = "#ffbb78"  # light orange
    NEG_COLOR = "#aec7e8"  # light blue
    BND_COLOR = "#4C4C4C"
    if np.any(pos):
        ax1.scatter(x[pos], t_norm_ms[pos], s=0.25, c=POS_COLOR, alpha=0.7, rasterized=True)
    if np.any(neg):
        ax1.scatter(x[neg], t_norm_ms[neg], s=0.25, c=NEG_COLOR, alpha=0.7, rasterized=True)
    for ln in x_lines:
        # Thinner, dashed boundary overlay
        ax1.plot(
            xs,
            ln / 1000.0,
            color=BND_COLOR,
            linewidth=1.0,
            alpha=0.9,
            linestyle="--",
            solid_capstyle="round",
        )
    ax1.set_xlabel("X (px)", fontsize=label_font)
    ax1.set_ylabel("Time (ms)", fontsize=label_font)
    # Ensure left y ticks/labels explicitly visible
    ax1.tick_params(axis='both', labelsize=tick_font)
    ax1.spines["left"].set_visible(True)
    duration_ms_full = duration_us / 1000.0
    # Add headroom for outside legend
    ax1.set_ylim(0.0, duration_ms_full + 0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Y–T
    if np.any(pos):
        ax2.scatter(y[pos], t_norm_ms[pos], s=0.25, c=POS_COLOR, alpha=0.7, rasterized=True)
    if np.any(neg):
        ax2.scatter(y[neg], t_norm_ms[neg], s=0.25, c=NEG_COLOR, alpha=0.7, rasterized=True)
    for ln in y_lines:
        # Thinner, dashed boundary overlay
        ax2.plot(
            ys,
            ln / 1000.0,
            color=BND_COLOR,
            linewidth=1.0,
            alpha=0.9,
            linestyle="--",
            solid_capstyle="round",
        )
    ax2.set_xlabel("Y (px)", fontsize=label_font)
    # Share y-axis with left panel; hide label and tick labels on the right
    ax2.set_ylabel("")
    ax2.tick_params(axis='both', labelsize=tick_font)
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2.set_ylim(0.0, duration_ms_full + 0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Small legend with color dots (Neg = light blue, Pos = light orange), outside top-right
    import matplotlib.lines as mlines
    # Use short labels to match paper style
    neg_dot = mlines.Line2D([], [], color=NEG_COLOR, marker='o', linestyle='None', markersize=4, label='Neg')
    pos_dot = mlines.Line2D([], [], color=POS_COLOR, marker='o', linestyle='None', markersize=4, label='Pos')
    legend = fig.legend(
        handles=[neg_dot, pos_dot],
        loc='upper right',
        bbox_to_anchor=(1.12, 0.98),
        borderaxespad=0.0,
        fontsize=13,
        framealpha=0.0,
    )

    # No manual subplots_adjust to avoid clipping labels; constrained_layout handles spacing
    stem = f"multiwindow_events{suffix}.pdf"
    out_path = out_dir / stem
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / stem.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.01)
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
                   var_ylim: float | None = None,
                   segment_npz: Path | None = None,
                   params: dict | None = None,
                   sensor_w: int = 1280, sensor_h: int = 720,
                   suffix: str = "") -> None:
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

    # Make panel (b) readable with compact size
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(bins, var_orig, color="#7f7f7f", linewidth=1.6, label="Original")
    ax.plot(bins, var_comp, color="#1f77b4", linewidth=1.6, label="Compensate")
    ax.set_xlabel("Time Bin", fontsize=12)
    # add headroom above max variance (user requested max + 0.2 to fit legend)
    data_top = max(float(np.max(var_orig) if len(var_orig) else 0.0),
                   float(np.max(var_comp) if len(var_comp) else 0.0))
    default_top = data_top + 0.2
    if var_ylim is not None:
        top = max(float(var_ylim), default_top)
    else:
        top = default_top
    ax.set_ylim(0.0, top)
    # Choose tick spacing heuristically based on range
    try:
        span = top
        if span <= 0.5:
            step = 0.05
        elif span <= 1.0:
            step = 0.1
        else:
            step = 0.2
        ticks = np.arange(0.0, top + 1e-9, step)
        ax.set_yticks(ticks)
    except Exception:
        pass
    ax.set_ylabel("Variance", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Place legend inside top-right with slight transparency for readability
    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    stem = f"multiwindow_variance{suffix}.pdf"
    out_path = out_dir / stem
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / stem.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_panel_c_separates(orig: np.ndarray, comp: np.ndarray, vmin: float, vmax: float, idx: int, out_dir: Path, suffix: str) -> None:
    def _one(img: np.ndarray, title: str, fname_pdf: str):
        f = plt.figure(figsize=(6.8, 3.2))
        ax = f.add_subplot(1, 1, 1)
        im = ax.imshow(img, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
        label_font = 12
        tick_font = 10
        ax.set_title(title, fontsize=18, pad=8)
        ax.set_xlabel("X (px)", fontsize=label_font)
        ax.set_ylabel("Y (px)", fontsize=label_font)
        ax.tick_params(axis='both', labelsize=tick_font)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="2.5%", height="85%", loc="center right", borderpad=1.3)
        cb = f.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=tick_font)
        cb.set_label("Value", rotation=90, fontsize=label_font)
        f.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.18)
        f.savefig(out_dir / fname_pdf, dpi=400, bbox_inches="tight", pad_inches=0.01)
        f.savefig(out_dir / fname_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(f)

    _one(orig, f"Original – Bin {idx}", f"multiwindow_bin50ms_original{suffix}.pdf")
    _one(comp, f"Compensated – Bin {idx}", f"multiwindow_bin50ms_compensated{suffix}.pdf")


def save_panel_c_plain(
    orig: np.ndarray,
    comp: np.ndarray,
    vmin: float,
    vmax: float,
    out_dir: Path,
    suffix: str,
    add_blue_variant: bool = False,
    add_blue_minmax_variant: bool = False,
) -> None:
    """Save plain images (no title, ticks, axes, or colorbar).

    If add_blue_variant is True, also save duplicates with center-zero mapping
    (TwoSlopeNorm, coolwarm) so negatives map to blue→white. Originals unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # PNG via imsave (no axes)
    import matplotlib.pyplot as plt
    png_o = out_dir / f"multiwindow_bin50ms_original_plain{suffix}.png"
    png_c = out_dir / f"multiwindow_bin50ms_compensated_plain{suffix}.png"
    plt.imsave(png_o, orig, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.imsave(png_c, comp, cmap="coolwarm", vmin=vmin, vmax=vmax)
    # PDF: embed raster without axes (original behavior: use vmin/vmax and coolwarm)
    def _pdf(img: np.ndarray, stem: str):
        f = plt.figure(figsize=(6, 3))
        ax = f.add_subplot(1, 1, 1)
        ax.imshow(img, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
        ax.axis("off")
        f.savefig(out_dir / stem, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(f)
    _pdf(orig, f"multiwindow_bin50ms_original_plain{suffix}.pdf")
    _pdf(comp, f"multiwindow_bin50ms_compensated_plain{suffix}.pdf")

    if add_blue_variant:
        from matplotlib.colors import TwoSlopeNorm
        def _pdf_blue(img: np.ndarray, stem: str):
            f = plt.figure(figsize=(6, 3))
            ax = f.add_subplot(1, 1, 1)
            # Revert to proven mapping: shared center-zero with reversed coolwarm
            A = max(abs(float(vmin)), abs(float(vmax)), 1e-6)
            norm = TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)
            ax.imshow(img, cmap="coolwarm_r", norm=norm, aspect="equal")
            ax.axis("off")
            f.savefig(out_dir / stem, dpi=300, bbox_inches="tight", pad_inches=0.0)
            if stem.endswith('.pdf'):
                f.savefig(out_dir / stem.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.0)
            plt.close(f)
        _pdf_blue(orig, f"multiwindow_bin50ms_original_plain_blue{suffix}.pdf")
        _pdf_blue(comp, f"multiwindow_bin50ms_compensated_plain_blue{suffix}.pdf")
        # Also save PNGs for convenience
        _pdf_blue(orig, f"multiwindow_bin50ms_original_plain_blue{suffix}.png")
        _pdf_blue(comp, f"multiwindow_bin50ms_compensated_plain_blue{suffix}.png")

    # Optional: blue-only min->max mapping (no red):
    # low (vmin_img) -> deep blue, high (vmax_img) -> white
    if add_blue_minmax_variant:
        from matplotlib.colors import ListedColormap, Normalize
        import matplotlib.pyplot as _plt

        base = _plt.get_cmap("coolwarm")
        blue_half = base(np.linspace(0.0, 0.5, 256))  # deep blue -> white
        blue_cmap = ListedColormap(blue_half)

        def _pdf_blue_minmax(img: np.ndarray, stem: str):
            f = plt.figure(figsize=(6, 3))
            ax = f.add_subplot(1, 1, 1)
            vals = img[np.isfinite(img)]
            if vals.size == 0:
                vmin_img, vmax_img = -1.0, 0.0
            else:
                vmin_img = float(np.min(vals))
                vmax_img = float(np.max(vals))
                if vmin_img == vmax_img:
                    vmax_img = vmin_img + 1e-6
            norm = Normalize(vmin=vmin_img, vmax=vmax_img)
            ax.imshow(img, cmap=blue_cmap, norm=norm, aspect="equal")
            ax.axis("off")
            f.savefig(out_dir / stem, dpi=300, bbox_inches="tight", pad_inches=0.0)
            if stem.endswith('.pdf'):
                f.savefig(out_dir / stem.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.0)
            plt.close(f)

        _pdf_blue_minmax(orig, f"multiwindow_bin50ms_original_plain_blueminmax{suffix}.pdf")
        _pdf_blue_minmax(comp, f"multiwindow_bin50ms_compensated_plain_blueminmax{suffix}.pdf")


def render_panel_c(
    segments_dir: Path,
    base: str,
    out_dir: Path,
    choose: str = "best",
    suffix: str = "",
    add_blue_variant: bool = False,
    add_blue_minmax_variant: bool = False,
) -> None:
    # Use the aggregated 50 ms bin NPZ for clean single-bin images
    _, allbins_path = find_timebin_csv_and_npz(segments_dir, base)
    with np.load(allbins_path, allow_pickle=False) as d:
        orig_keys = sorted(k for k in d.keys() if k.startswith("original_bin_"))
        comp_keys = sorted(k for k in d.keys() if k.startswith("compensated_bin_"))
        if not orig_keys or len(orig_keys) != len(comp_keys):
            raise RuntimeError(f"Unexpected NPZ structure in {allbins_path}")

        var_orig = np.array([
            float(np.var(d[k].astype(np.float32))) for k in orig_keys
        ], dtype=np.float32)
        var_comp = np.array([
            float(np.var(d[k].astype(np.float32))) for k in comp_keys
        ], dtype=np.float32)
        diff = var_orig - var_comp

        if choose == "bin4":
            idx = min(4, len(orig_keys) - 1)
        else:
            idx = int(np.argmax(diff))
            if diff[idx] <= 0:
                idx = int(np.argmax(var_orig))

        orig = d[orig_keys[idx]]
        comp = d[comp_keys[idx]]

    vmin = float(min(orig.min(), comp.min()))
    vmax = float(max(orig.max(), comp.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    label_font = 12
    tick_font = 10
    im1 = ax1.imshow(orig, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
    ax1.set_title(f"Original – Bin {idx}", fontsize=label_font)
    ax1.set_xlabel("X (px)", fontsize=label_font)
    ax1.set_ylabel("Y (px)", fontsize=label_font)
    # Ensure left y ticks/labels explicitly visible
    ax1.tick_params(axis='both', labelsize=tick_font)
    ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax1.spines["left"].set_visible(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    im2 = ax2.imshow(comp, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
    ax2.set_title(f"Compensated – Bin {idx}", fontsize=label_font)
    ax2.set_xlabel("X (px)", fontsize=label_font)
    # Hide right panel's y-label and tick labels (shared y-axis)
    ax2.set_ylabel("")
    ax2.tick_params(axis='both', labelsize=tick_font)
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Shared colorbar at the right side (outside of image area)
    fig.subplots_adjust(right=0.86, left=0.11, top=0.94, bottom=0.12, wspace=0.06)
    # Match colorbar height to axes height by using add_axes with normalized coords
    ax_pos = ax1.get_position()
    cbar_width = 0.02
    cbar_left = 0.86 + 0.02
    cbar_bottom = ax_pos.y0
    cbar_height = ax_pos.height
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.ax.set_ylabel("Value", rotation=90, fontsize=label_font)
    cbar.ax.tick_params(labelsize=tick_font)

    stem = f"multiwindow_bin50ms{suffix}.pdf"
    out_path = out_dir / stem
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / stem.replace('.pdf', '.png'), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")
    # Also save separate single panels using same vmin/vmax
    save_panel_c_separates(orig, comp, vmin, vmax, idx, out_dir, suffix)
    # And save plain images without titles/ticks/colorbar
    save_panel_c_plain(
        orig,
        comp,
        vmin,
        vmax,
        out_dir,
        suffix,
        add_blue_variant=add_blue_variant,
        add_blue_minmax_variant=add_blue_minmax_variant,
    )


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
    parser.add_argument("--var_bin_us", type=int, default=50000,
                        help="Bin width in microseconds for variance when variance_mode=recompute (default: 50000us = 50ms)")
    parser.add_argument("--var_ylim", type=float, default=None, help="Upper y-limit for panel (b) variance (e.g., 1.0)")
    parser.add_argument("--sample", type=float, default=0.05, help="Event sampling fraction for panel (a)")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Optional suffix appended to output filenames, e.g. 'blank' or 'sanqin'.")
    parser.add_argument("--panel_c_choice", choices=["best", "bin4"], default="best",
                        help="Select which time bin to visualize in panel (c); default 'best' chooses max std diff, 'bin4' forces bin 4.")
    parser.add_argument("--plain_blue_extra", action="store_true",
                        help="Save extra blue/white duplicates for plain images (keeps originals unchanged)")
    parser.add_argument("--plain_blue_minmax_extra", action="store_true",
                        help="Save blue-only min->max extras (min=deep blue, max=white) for both original and compensated")
    args = parser.parse_args()

    setup_style()
    import datetime
    out_root = args.output_dir.resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (out_root / f"multiwindow_compensation_{timestamp}").resolve()
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

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    render_panel_a(segment_path, params, args.sensor_width, args.sensor_height, args.sample, out_dir, suffix=suffix)
    render_panel_b(
        segments_dir,
        base,
        out_dir,
        mode=args.variance_mode,
        var_bin_us=args.var_bin_us,
        var_ylim=args.var_ylim,
        segment_npz=segment_path,
        params=params,
        sensor_w=args.sensor_width,
        sensor_h=args.sensor_height,
        suffix=suffix,
    )
    render_panel_c(
        segments_dir,
        base,
        out_dir,
        choose=args.panel_c_choice,
        suffix=suffix,
        add_blue_variant=bool(args.plain_blue_extra),
        add_blue_minmax_variant=bool(args.plain_blue_minmax_extra),
    )


if __name__ == "__main__":
    main()
