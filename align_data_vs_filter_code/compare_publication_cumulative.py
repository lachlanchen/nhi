#!/usr/bin/env python3
"""Publication-ready overlays: cumulative reconstruction vs two ground truths.

Generates exactly one figure per provided reconstruction segment. Each figure
contains three lines: the compensated, auto-scaled exponential cumulative
reconstruction and the two spectrometer ground-truth curves. Styling uses
thicker lines and larger fonts suitable for publications. Outputs are saved to
`align_bg_vs_gt_code/publication_<timestamp>/` by default (or under the folder
specified via ``--output_root``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Repo root for importing shared helpers
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse existing utilities
import visualize_cumulative_weighted as vcw  # noqa: E402
from compare_reconstruction_to_gt import (  # noqa: E402
    detect_active_region,
    load_ground_truth,
    moving_average,
    normalise_curve,
)


def build_compensated_cumulative(
    segment_npz: Path,
    step_ms: float,
    pos_scale: float,
    auto_bounds: Tuple[float, float],
    plateau_frac: float,
    sensor_width: int,
    sensor_height: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (time_ms, exp-cumulative series, chosen_neg_scale).

    - Uses learned parameters to compute fast-compensated timestamps.
    - Auto-tunes the negative polarity scale to equalize start/end plateaus.
    - Applies exp transform on the per-pixel cumulative mean.
    """
    x, y, t, p = vcw.load_npz_events(str(segment_npz))
    t_min, t_max = float(np.min(t)), float(np.max(t))

    # Compensation
    param_file = vcw.find_param_file_for_segment(str(segment_npz))
    if param_file is None:
        raise FileNotFoundError("No learned parameter NPZ found next to segment")
    params = vcw.load_parameters_from_npz(param_file)
    t_comp, *_ = vcw.compute_fast_compensated_times(x, y, t, params["a_params"], params["b_params"])

    # Binned sums by polarity (compensated domain)
    step_us = step_ms * 1000.0
    pos_mask = p >= 0
    neg_mask = ~pos_mask
    sums_pos, edges_ms = vcw.base_binned_sums_weighted(
        t_comp[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.float32), t_min, t_max, step_us
    )
    sums_neg, _ = vcw.base_binned_sums_weighted(
        t_comp[neg_mask], np.ones(np.count_nonzero(neg_mask), dtype=np.float32), t_min, t_max, step_us
    )

    hw = float(sensor_width * sensor_height)

    def build_cum(neg_scale: float) -> np.ndarray:
        return np.exp(np.cumsum(pos_scale * sums_pos - neg_scale * sums_neg) / hw)

    # Auto-scale negative weight so the exp-cumulative plateaus match
    neg_min, neg_max = auto_bounds
    def plateau_diff(neg_scale: float) -> float:
        series = build_cum(neg_scale)
        k = len(series)
        if k <= 2:
            return 0.0
        n = max(5, int(plateau_frac * k))
        return float(np.mean(series[-n:]) - np.mean(series[:n]))

    f_min, f_max = plateau_diff(neg_min), plateau_diff(neg_max)
    if f_min * f_max < 0:
        a, b = neg_min, neg_max
        for _ in range(40):
            m = 0.5 * (a + b)
            fm = plateau_diff(m)
            if abs(fm) < 1e-6:
                a = b = m
                break
            if f_min * fm < 0:
                b, f_max = m, fm
            else:
                a, f_min = m, fm
        chosen = 0.5 * (a + b)
    else:
        grid = np.linspace(neg_min, neg_max, 50)
        vals = np.array([abs(plateau_diff(g)) for g in grid])
        chosen = float(grid[int(np.argmin(vals))])

    series = build_cum(chosen)
    time_ms = edges_ms - edges_ms[0]
    return time_ms, series, chosen


def load_gt_curves(gt_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Collect spectrometer curves from a directory, skipping non-spectral txt files.

    The folder may also contain analysis summaries (txt) that do not follow the
    OceanView export format. We attempt to parse each and keep only files that
    yield a non-empty numeric spectrum.
    """
    valid: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for txt in sorted(gt_dir.glob("*.txt")):
        try:
            wl, val = load_ground_truth(txt)
        except Exception:
            continue
        if isinstance(wl, np.ndarray) and wl.size >= 50 and isinstance(val, np.ndarray) and val.size == wl.size:
            valid.append((txt.stem, wl, val))
    if not valid:
        raise FileNotFoundError(
            f"Found no spectrometer files in {gt_dir}; expected at least one (*.txt with spectral data)"
        )
    # Heuristic: prefer files beginning with 'USB' if present
    valid.sort(key=lambda t: (0 if t[0].upper().startswith("USB") else 1, t[0]))
    # Historically we used the first two; callers may also provide explicit files.
    return valid[:2]



def detect_visible_edges(curves: Sequence[Tuple[str, np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    starts, ends = [], []
    for _, wl, val in curves:
        mask = (wl >= 300.0) & (wl <= 900.0)
        wl_sel = wl[mask]
        val_sel = val[mask]
        smooth = moving_average(val_sel, max(21, len(val) // 300))
        region = detect_active_region(smooth)
        starts.append(float(wl_sel[region.start_idx]))
        ends.append(float(wl_sel[region.end_idx]))
    return float(np.mean(starts)), float(np.mean(ends))


def align_series_to_wavelength(
    time_ms: np.ndarray, series: np.ndarray, gt_start_nm: float, gt_end_nm: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    smooth = moving_average(series, max(21, int(len(series) // 200) | 1))
    region = detect_active_region(smooth)
    series_norm = normalise_curve(series, region)
    t0 = float(time_ms[region.start_idx])
    t1 = float(time_ms[region.end_idx])
    if t1 <= t0:
        raise ValueError("Detected non-increasing active region in reconstruction")
    slope = (gt_end_nm - gt_start_nm) / (t1 - t0)
    intercept = gt_start_nm - slope * t0
    wl = slope * time_ms + intercept
    return wl, series_norm, slope, intercept


def publication_style() -> None:
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.5,
    })


def ensure_output_dir(root: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"publication_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    default_segment = (
        REPO_ROOT
        / "scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184724/"
        "angle_20_blank_2835_event_20250925_184724_segments/Scan_1_Forward_events.npz"
    )
    ap = argparse.ArgumentParser(description="Publication-style cumulative vs ground-truth overlays")
    ap.add_argument("--segments", type=Path, nargs="+", default=[default_segment], help="One or more segment NPZ files")
    ap.add_argument("--gt_dir", type=Path, default=REPO_ROOT / "groundtruth_spectrum_2835", help="Folder of .txt spectrometer files")
    ap.add_argument("--step_ms", type=float, default=2.0, help="Cumulative step size (ms)")
    ap.add_argument("--sensor_width", type=int, default=1280)
    ap.add_argument("--sensor_height", type=int, default=720)
    ap.add_argument("--pos_scale", type=float, default=1.0)
    ap.add_argument("--auto_neg_min", type=float, default=0.1)
    ap.add_argument("--auto_neg_max", type=float, default=3.0)
    ap.add_argument("--plateau_frac", type=float, default=0.05)
    ap.add_argument("--xlim", type=float, nargs=2, default=(300.0, 900.0), help="X-axis wavelength limits (nm)")
    ap.add_argument(
        "--gt_files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of spectrometer TXT files to use (overrides --gt_dir)",
    )
    ap.add_argument("--output_root", type=Path, default=REPO_ROOT / "align_bg_vs_gt_code")
    ap.add_argument("--show", action="store_true", help="Display plots interactively")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    publication_style()

    # Load one or more ground-truth curves
    if args.gt_files:
        gt_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for txt in args.gt_files:
            wl, val = load_ground_truth(txt)
            gt_curves.append((txt.stem, wl, val))
    else:
        gt_curves = load_gt_curves(args.gt_dir)
    gt_start_nm, gt_end_nm = detect_visible_edges(gt_curves)

    out_dir = ensure_output_dir(args.output_root)

    # Pre-normalise ground-truth curves for plotting consistency
    gt_norm_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, wl, val in gt_curves:
        mask = (wl >= args.xlim[0]) & (wl <= args.xlim[1])
        wl_sel = wl[mask]
        val_sel = val[mask]
        smooth = moving_average(val_sel, max(21, len(val) // 300))
        region = detect_active_region(smooth)
        norm_val = normalise_curve(smooth, region)
        gt_norm_curves.append((name, wl_sel, norm_val))

    # Process each requested reconstruction
    for seg in args.segments:
        time_ms, series, neg_scale = build_compensated_cumulative(
            seg,
            step_ms=args.step_ms,
            pos_scale=args.pos_scale,
            auto_bounds=(args.auto_neg_min, args.auto_neg_max),
            plateau_frac=args.plateau_frac,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
        )
        wl_recon, recon_norm, slope, intercept = align_series_to_wavelength(
            time_ms, series, gt_start_nm, gt_end_nm
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axvspan(380, 780, color="0.92", zorder=0)
        # Plot two GT curves with short labels: GT 1, GT 2
        colors = ["#1f77b4", "#ff7f0e"]
        for i, ((_, wl, gt), c) in enumerate(zip(gt_norm_curves, colors), start=1):
            ax.plot(wl, gt, color=c, linewidth=3.0, label=f"GT {i}")
        # Plot reconstruction with short label: Recon
        ax.plot(wl_recon, recon_norm, color="#2ca02c", linewidth=3.0, label="Recon")

        ax.set_xlim(*args.xlim)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalised intensity")
        ax.grid(alpha=0.3)
        seg_name = seg.stem.replace("_events", "")
        ax.set_title(seg_name)
        ax.legend(loc="upper right")
        fig.tight_layout()

        out_name = f"publication_cumulative_{seg.parent.parent.name}_{seg_name}.png"
        fig.savefig(out_dir / out_name, dpi=300)

        # Also save a brief mapping for record keeping
        with (out_dir / f"mapping_{seg_name}.txt").open("w", encoding="utf-8") as f:
            f.write(f"Segment: {seg}\n")
            f.write(f"Neg scale: {neg_scale:.6f}\n")
            f.write(f"Mapping: λ(t_ms) = {slope:.6f} * t_ms + {intercept:.3f}\n")

        if args.show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)

    print(f"Saved publication overlays to: {out_dir}")


if __name__ == "__main__":
    main()
