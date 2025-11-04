#!/usr/bin/env python3
"""
Figure 4 variant: Orig., Comp., Diff., and Ref. (gradient 20 nm images) + color bar.

Rows:
  - Row 1: Original 50 ms frames
  - Row 2: Compensated (weighted, smoothed, background-subtracted) 50 ms frames
  - Row 3: Diff = Comp - Orig
  - Row 4: Ref. images from a gradient 20 nm folder (match by start wavelength)
  - Bottom: thin color bar spanning available reference wavelengths

This script reuses the core accumulation and alignment logic from the Figure 4 tools.

Example:
  python publication_code/figure04_allinone_gradref.py \
    --segment scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
    --gt-dir groundtruth_spectrum_2835 \
    --grad20-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched_gradient_20nm \
    --bin-width-us 50000 --start-bin 3 --end-bin 15 --save-png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Colormap

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figure04_rescaled import (
    DEFAULT_SHARED_COLORMAP,
    RAW_LIGHTEN_FRACTION,
    COMP_LIGHTEN_FRACTION,
    RESCALE_FINE_STEP_US,
    load_segment_events,
    find_param_file,
    load_params,
    prepare_colormap,
    accumulate_bin,
    smooth_volume_3d,
    subtract_background,
    setup_style,
)
from publication_code.figure04_rescaled_allinone import align_with_groundtruth
from scripts.hs_to_rgb import load_cie_cmf, xyz_to_srgb
import re


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Figure 4 (Orig/Comp/Diff/Ref) with gradient-20nm references")
    ap.add_argument("--segment", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=Path("groundtruth_spectrum_2835"))
    ap.add_argument("--grad20-dir", type=Path, required=True, help="Path to gradient_20nm reference frames")
    ap.add_argument("--bin-width-us", type=float, default=50000.0)
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--neg-scale", type=float, default=1.5)
    ap.add_argument("--start-bin", type=int, default=3)
    ap.add_argument("--end-bin", type=int, default=15)
    ap.add_argument("--colormap", default=None)
    ap.add_argument("--raw-colormap", default=None)
    ap.add_argument("--comp-colormap", default=None)
    ap.add_argument("--smooth", action="store_true", default=True)
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def nm_to_rgb_color(nm: float, repo_root: Path) -> np.ndarray:
    wl, xbar, ybar, zbar = load_cie_cmf(repo_root)
    x = float(np.interp(nm, wl, xbar))
    y = float(np.interp(nm, wl, ybar))
    z = float(np.interp(nm, wl, zbar))
    XYZ = np.array([x, y, z], dtype=np.float32)
    scale = 1.0 / XYZ[1] if XYZ[1] > 0 else 1.0
    rgb = xyz_to_srgb(XYZ[None, :] * scale)[0]
    return rgb


def build_gradient_bar_from_grad20(frames_dir: Path, height_px: int = 6) -> np.ndarray:
    files = list(sorted(frames_dir.glob("*.png")))
    starts: List[float] = []
    pat = re.compile(r".*_(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)nm\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    for p in files:
        m = pat.match(p.name)
        if not m:
            continue
        try:
            starts.append(float(m.group(1)))
        except Exception:
            continue
    if not starts:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    starts_sorted = sorted(starts)
    cols = []
    for nm in starts_sorted:
        rgb = nm_to_rgb_color(float(nm), REPO_ROOT)
        col = np.tile(rgb[None, None, :], (height_px, 1, 1))
        cols.append(col)
    return np.concatenate(cols, axis=1)


def parse_start_nm_from_grad20_name(name: str) -> float | None:
    m = re.match(r".*_(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)nm\.(?:png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def select_grad20_for_bins(frames_dir: Path, wl_lookup: Dict[int, float], meta: List[Dict[str, float]]) -> List[Path]:
    files = list(sorted(frames_dir.glob("*.png")))
    starts: List[Tuple[float, Path]] = []
    for p in files:
        nm = parse_start_nm_from_grad20_name(p.name)
        if nm is None:
            continue
        starts.append((nm, p))
    starts.sort(key=lambda x: x[0])
    if not starts:
        return []
    nm_array = np.array([s for s, _ in starts], dtype=np.float32)
    out: List[Path] = []
    for m in meta:
        idx = int(m["index"])
        wl = float(wl_lookup.get(idx, np.nan))
        if not np.isfinite(wl):
            out.append(Path())
            continue
        j = int(np.argmin(np.abs(nm_array - wl)))
        out.append(starts[j][1])
    return out


def main() -> None:
    args = parse_args()
    seg_path = args.segment.resolve()
    assert seg_path.exists(), seg_path

    figures_root = Path(__file__).resolve().parent / "figures"
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (figures_root / f"figure04_allinone_gradref_{suffix}").resolve() if args.output_dir is None else args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load events
    x, y, t, p = load_segment_events(seg_path)
    sensor_shape = (args.sensor_height, args.sensor_width)

    # Params
    params_file = find_param_file(seg_path)
    a_params = b_params = None
    a_avg = b_avg = 0.0
    if params_file is not None:
        params = load_params(params_file)
        a_params = params["a_params"]
        b_params = params["b_params"]
        a_avg = float(np.mean(a_params)); b_avg = float(np.mean(b_params))

    # Fine 5 ms series for alignment (neg_scale auto from compensated time)
    if a_params is not None and b_params is not None:
        t_comp_full = t - (a_avg * x + b_avg * y)
    else:
        t_comp_full = t
    from figure04_rescaled import auto_scale_neg_weight
    sensor_area = float(args.sensor_width * args.sensor_height)
    neg_scale, series = auto_scale_neg_weight(
        t_comp_full.astype(np.float32),
        p.astype(np.float32),
        sensor_area=sensor_area,
        step_us=float(RESCALE_FINE_STEP_US),
        pos_scale=float(args.pos_scale),
        return_series=True,
    )  # type: ignore[misc]
    neg_scale = float(neg_scale)

    # Build 50 ms frames (orig + comp weights) and metadata
    t_min = float(np.min(t)); t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / args.bin_width_us))
    originals: List[np.ndarray] = []
    comp_raw_frames: List[np.ndarray] = []
    metadata_bins: List[Dict[str, float]] = []
    t_comp_for_bins = t_comp_full
    for idx in range(num_bins):
        start = t_min + idx * args.bin_width_us
        end = start + args.bin_width_us
        mask_orig = (t >= start) & (t < end)
        mask_comp = (t_comp_for_bins >= start) & (t_comp_for_bins < end)
        originals.append(accumulate_bin(x, y, mask_orig, np.ones_like(p, dtype=np.float32), sensor_shape))
        # comp weights
        weights = np.where(p >= 0, float(args.pos_scale), -neg_scale).astype(np.float32)
        comp_raw_frames.append(accumulate_bin(x, y, mask_comp, weights, sensor_shape).astype(np.float32))
        end_clamped = float(min(end, t_max))
        metadata_bins.append({"index": int(idx), "start_us": float(start), "end_us": end_clamped})

    comp_array = np.stack(comp_raw_frames, axis=0)
    if args.smooth:
        comp_array = smooth_volume_3d(comp_array)
    compensated = [subtract_background(f) for f in comp_array]

    # Align with GT to derive wavelength mapping per displayed bin
    align_payload = align_with_groundtruth(series, args.gt_dir.resolve(), metadata_bins, out_dir, neg_scale, float(RESCALE_FINE_STEP_US), args.save_png)
    wavelength_lookup = {int(item["index"]): float(item["wavelength_nm"]) for item in align_payload["bin_mapping"]}

    # Select the display range
    selected = [ (originals[i], compensated[i], metadata_bins[i]) for i in range(len(metadata_bins)) if args.start_bin <= i <= args.end_bin ]
    if not selected:
        raise RuntimeError("No bins selected for display")

    # Prepare colormaps
    from figure04_rescaled import prepare_colormap
    shared_base = args.colormap or DEFAULT_SHARED_COLORMAP
    raw_cmap = prepare_colormap(args.raw_colormap or shared_base, "min", RAW_LIGHTEN_FRACTION)
    comp_cmap = prepare_colormap(args.comp_colormap or shared_base, "center", COMP_LIGHTEN_FRACTION)

    # Gather gradient-20nm refs for displayed columns matching start wavelength
    grad_refs = select_grad20_for_bins(args.grad20_dir.resolve(), wavelength_lookup, [m for _,_,m in selected])

    # Build color bar from the gradient dir
    bar = build_gradient_bar_from_grad20(args.grad20_dir.resolve(), height_px=4)

    # Render grid: rows Orig, Comp, Diff, Ref + bar
    setup_style()
    num_cols = len(selected)
    has_bar = bar.size > 0
    fig = plt.figure(figsize=(min(0.75 * num_cols, 18), 3.0 * (4 + (1 if has_bar else 0))), constrained_layout=True)
    gs = GridSpec(4 + (1 if has_bar else 0), num_cols + 1, figure=fig, height_ratios=([1.0, 1.0, 1.0, 1.0] + ([0.08] if has_bar else [])))

    # Column labels row
    ax_label = fig.add_subplot(gs[0, 0]); ax_label.axis('off')
    ax_label.text(0.5, 1.05, "", ha='center', va='bottom')
    # Row 0: Orig.
    for c, (orig, _, meta) in enumerate(selected):
        ax = fig.add_subplot(gs[0, c + 1])
        im = ax.imshow(orig, cmap=raw_cmap)
        wl = wavelength_lookup.get(int(meta["index"]))
        if wl is not None:
            ax.set_title(f"{wl:.0f} nm", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Orig.", fontsize=9)

    # Row 1: Comp.
    for c, (_, comp, _) in enumerate(selected):
        ax = fig.add_subplot(gs[1, c + 1])
        ax.imshow(comp, cmap=comp_cmap)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Comp.", fontsize=9)

    # Row 2: Diff = Comp - Orig
    for c, (orig, comp, _) in enumerate(selected):
        ax = fig.add_subplot(gs[2, c + 1])
        diff = (comp.astype(np.float32) - orig.astype(np.float32))
        ax.imshow(diff, cmap=comp_cmap)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Diff.", fontsize=9)

    # Row 3: Ref. from gradient-20nm (match by nearest start nm)
    for c, ref_path in enumerate(grad_refs):
        ax = fig.add_subplot(gs[3, c + 1])
        if ref_path and ref_path.exists():
            import matplotlib.image as mpimg
            try:
                ax.imshow(mpimg.imread(str(ref_path)), cmap='gray')
            except Exception:
                ax.text(0.5, 0.5, "Ref load error", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "Ref missing", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Ref.", fontsize=9)

    # Bar row
    if has_bar:
        ax_bar = fig.add_subplot(gs[4, 1:])
        ax_bar.imshow(bar, origin='lower', aspect='auto')
        ax_bar.axis('off')

    stem = out_dir / f"figure04_gradref_bins_{args.start_bin:02d}_{args.end_bin:02d}"
    fig.savefig(f"{stem}.pdf", dpi=400, bbox_inches='tight')
    if args.save_png:
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save a small JSON manifest
    manifest = {
        "segment": str(seg_path),
        "params_file": str(params_file) if params_file else None,
        "bin_width_us": float(args.bin_width_us),
        "start_bin": int(args.start_bin),
        "end_bin": int(args.end_bin),
        "grad20_dir": str(args.grad20_dir.resolve()),
        "output": str(stem) + ".pdf",
        "bins": [int(m[2]["index"]) for m in selected],
    }
    with (out_dir / "figure04_gradref_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print("Saved:", stem.with_suffix('.pdf'))


if __name__ == "__main__":
    main()

