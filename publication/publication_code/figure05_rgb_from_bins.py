#!/usr/bin/env python3
"""
Figure 5: RGB render of time-binned reconstructions using Figure 4's
wavelength mapping.

This script converts original and compensated time-binned frames to RGB by
assigning each bin a single wavelength (from the Figure 4 alignment), then
mapping per-pixel intensity at that wavelength through the CIE 1931 2° color
matching functions (XYZ → sRGB).

It supports two bin widths:
  - 50 ms: prefers existing frames from the compensation pipeline NPZ
  - 5  ms: computed on-the-fly from events (original and compensated times)

Outputs are written to publication_code/figures/figure05_<timestamp>/ with
subfolders for each bin width and modality.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Make repo root importable, then reuse helpers from Figure 4 and CIE utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_code.figure04_rescaled import (  # type: ignore
    load_segment_events,
    find_param_file,
    load_params,
    compute_fast_comp_times,
    accumulate_bin,
)
from scripts.hs_to_rgb import (  # type: ignore
    load_cie_cmf,
    xyz_to_srgb,
)


@dataclass
class Alignment:
    slope_nm_per_ms: float
    intercept_nm: float

    def wavelength_for_ms(self, t_ms: float) -> float:
        return float(self.slope_nm_per_ms * t_ms + self.intercept_nm)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Figure 5: RGB render from binned frames using wavelength mapping")
    ap.add_argument("--segment", type=Path, required=True, help="Path to Scan_*_Forward_events.npz (or backward)")
    ap.add_argument(
        "--alignment-json",
        type=Path,
        default=None,
        help="Path to figure04_rescaled_bg_alignment.json (if omitted, picks the latest under publication_code/figures/figure04_allinone_*/)",
    )
    ap.add_argument("--sensor-width", type=int, default=1280)
    ap.add_argument("--sensor-height", type=int, default=720)
    ap.add_argument("--pos-scale", type=float, default=1.0, help="Positive event weight for compensated accumulation (default 1.0)")
    ap.add_argument("--neg-scale", type=float, default=1.5, help="Initial negative event weight before auto-scaling")
    ap.add_argument(
        "--bin-widths-us",
        type=float,
        nargs="+",
        default=(5000.0, 50000.0),
        help="Bin widths to render in microseconds (default: 5ms and 50ms)",
    )
    ap.add_argument(
        "--per-bin-mode",
        choices=("gray", "rgb"),
        default="gray",
        help="How to save per-bin frames: grayscale (default) or colorized via wavelength.",
    )
    ap.add_argument(
        "--modes",
        choices=("orig", "comp", "both"),
        default="both",
        help="Which modalities to render (original times, compensated times, or both)",
    )
    ap.add_argument("--start-bin", type=int, default=None, help="Optional inclusive start bin index to export")
    ap.add_argument("--end-bin", type=int, default=None, help="Optional inclusive end bin index to export")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory root (default: figures/figure05_<timestamp>)")
    ap.add_argument("--smooth", action="store_true", help="Apply 3x3x3 spatio-temporal mean filter before background removal")
    # Accept both the intended spelling and the earlier typo for convenience
    ap.add_argument("--remove-bg-spectral", dest="remove_bg", action="store_true", help="Subtract per-frame mean (background) after optional smoothing")
    ap.add_argument("--remove-bg-spetral", dest="remove_bg", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--save-png", action="store_true", help="Save PNGs alongside PDFs where applicable")
    return ap.parse_args()


def find_latest_alignment() -> Path:
    base = REPO_ROOT / "publication_code" / "figures"
    if not base.exists():
        raise FileNotFoundError("Figures directory not found for locating alignment JSON.")
    cand: List[Path] = sorted(base.glob("figure04_allinone_*/figure04_rescaled_bg_alignment.json"))
    if not cand:
        raise FileNotFoundError("No figure04_allinone alignment JSON found under publication_code/figures.")
    return cand[-1]


def load_alignment(path: Path) -> Alignment:
    data = json.loads(path.read_text(encoding="utf-8"))
    al = data.get("alignment", {})
    slope = float(al.get("slope_nm_per_ms"))
    intercept = float(al.get("intercept_nm"))
    return Alignment(slope_nm_per_ms=slope, intercept_nm=intercept)


def load_timebinned_npz_if_available(segment_npz: Path) -> Tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """Try to load 50 ms frames saved by the compensation script.

    Returns: (originals [N,H,W], compensated [N,H,W], bin_width_us)
    or (None, None, None) if not found.
    """
    tb_dir = segment_npz.parent / "time_binned_frames"
    if not tb_dir.is_dir():
        return None, None, None
    # Find the most recent all_time_bins_data NPZ for this segment
    pats = sorted(tb_dir.glob(f"{segment_npz.stem}_chunked_processing_all_time_bins_data_*.npz"))
    if not pats:
        return None, None, None
    npz_path = pats[-1]
    npz = np.load(npz_path, allow_pickle=True)
    # Infer num bins from available keys
    keys = [k for k in npz.files if k.startswith("original_bin_")]
    if not keys:
        return None, None, None
    num_bins = max(int(k.split("_")[-1]) for k in keys) + 1
    H, W = npz[keys[0]].shape
    originals = np.empty((num_bins, H, W), dtype=np.float32)
    compensated = np.empty((num_bins, H, W), dtype=np.float32)
    for i in range(num_bins):
        originals[i] = npz[f"original_bin_{i}"]
        compensated[i] = npz[f"compensated_bin_{i}"]
    bin_width_us = float(npz["bin_width_us"]) if "bin_width_us" in npz.files else 50000.0
    return originals, compensated, bin_width_us


def accumulate_bins_from_events(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    t_comp: np.ndarray | None,
    weights: np.ndarray,
    bin_width_us: float,
    sensor_shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Accumulate original (raw time) and compensated (t_comp) counts into bins.

    We use uniform weights (=1) and do not perform background subtraction here,
    because we only need intensity for colorimetric mapping.
    """
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    num_bins = int(np.ceil((t_max - t_min) / bin_width_us))
    H, W = sensor_shape
    compensated = np.zeros((num_bins, H, W), dtype=np.float32)
    if t_comp is None:
        t_comp = t

    metadata: List[Dict[str, float]] = []
    for idx in range(num_bins):
        start = t_min + idx * bin_width_us
        end = start + bin_width_us
        # Last bin includes the max endpoint
        mask_o = (t >= start) & (t < end)
        mask_c = (t_comp >= start) & (t_comp < end)
        if idx == num_bins - 1:
            mask_o |= t == t_max
            mask_c |= t_comp == float(np.max(t_comp))

        compensated[idx] = accumulate_bin(x, y, mask_c, weights, sensor_shape)
        metadata.append(
            {
                "index": idx,
                "start_us": float(start),
                "end_us": float(min(end, t_max)),
                "duration_ms": float((min(end, t_max) - start) / 1000.0),
            }
        )
    return compensated, metadata


VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 680.0


def _clamp_wavelength(wl_nm: float) -> float:
    return float(min(max(wl_nm, VISIBLE_MIN_NM), VISIBLE_MAX_NM))


def scalar_frame_to_rgb(frame: np.ndarray, wl_nm: float, cmf: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Map a single-band frame at wavelength wl_nm to sRGB via CIE 1931 CMFs.

    - Negative intensities are clipped to 0 before mapping.
    - Brightness is normalised per-frame using the 99th percentile of the Y
      channel so that images have good contrast while remaining comparable.
    """
    wl_cmf, xbar, ybar, zbar = cmf
    # Interpolate CMFs at wl_nm
    wl_used = _clamp_wavelength(wl_nm)
    xk = float(np.interp(wl_used, wl_cmf, xbar))
    yk = float(np.interp(wl_used, wl_cmf, ybar))
    zk = float(np.interp(wl_used, wl_cmf, zbar))
    f = np.clip(frame.astype(np.float32), 0.0, None)
    X = f * xk
    Y = f * yk
    Z = f * zk
    XYZ = np.stack([X, Y, Z], axis=-1)
    # Scale to 99th percentile of Y
    y_scale = float(np.percentile(Y, 99.0))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0
    XYZ /= y_scale
    rgb = xyz_to_srgb(XYZ)
    return rgb


def composite_rgb_from_series(
    frames: np.ndarray,  # [N,H,W] float32
    metadata: List[Dict[str, float]],
    alignment: Alignment,
    cmf: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Integrate across wavelengths to form a single RGB image.

    - For each bin i, map its centre time to wavelength λ_i and approximate
      Δλ_i = slope_nm_per_ms * duration_ms.
    - Accumulate X = Σ max(F_i,0) * x̄(λ_i) * Δλ_i (and similarly for Y,Z).
    - Normalise by 99th percentile of Y before XYZ→sRGB.
    """
    if frames.size == 0 or not metadata:
        raise ValueError("Empty frames/metadata for composite RGB rendering.")
    wl_cmf, xbar, ybar, zbar = cmf
    base_start = metadata[0]["start_us"]
    centres_ms = np.array([
        (((m["start_us"] + m["end_us"]) * 0.5) - base_start) / 1000.0 for m in metadata
    ], dtype=np.float32)
    durations_ms = np.array([m["duration_ms"] for m in metadata], dtype=np.float32)
    wavelengths = alignment.slope_nm_per_ms * centres_ms + alignment.intercept_nm
    wavelengths = np.clip(wavelengths, VISIBLE_MIN_NM, VISIBLE_MAX_NM)
    delta_lambda = alignment.slope_nm_per_ms * durations_ms

    xk = np.interp(wavelengths, wl_cmf, xbar).astype(np.float32)
    yk = np.interp(wavelengths, wl_cmf, ybar).astype(np.float32)
    zk = np.interp(wavelengths, wl_cmf, zbar).astype(np.float32)
    # Multiply CMFs by Δλ to approximate integration
    xw = xk * delta_lambda
    yw = yk * delta_lambda
    zw = zk * delta_lambda

    # Clip negatives to zero prior to integration
    F = np.clip(frames.astype(np.float32), 0.0, None)
    # tensordot over bin dimension -> [H,W]
    X = np.tensordot(F, xw, axes=(0, 0))
    Y = np.tensordot(F, yw, axes=(0, 0))
    Z = np.tensordot(F, zw, axes=(0, 0))
    XYZ = np.stack([X, Y, Z], axis=-1)
    # Normalise by 99th percentile of Y
    y_scale = float(np.percentile(Y, 99.0))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0
    XYZ /= y_scale
    rgb = xyz_to_srgb(XYZ)
    return rgb


def ensure_outdir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_rgb_frames(
    frames: np.ndarray,
    metadata: List[Dict[str, float]],
    alignment: Alignment,
    cmf: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_dir: Path,
    prefix: str,
    start_bin: int | None,
    end_bin: int | None,
) -> List[Dict[str, float]]:
    out_meta: List[Dict[str, float]] = []
    if not len(metadata):
        return out_meta
    base_start = metadata[0]["start_us"]
    for m in metadata:
        idx = int(m["index"])
        if start_bin is not None and idx < start_bin:
            continue
        if end_bin is not None and idx > end_bin:
            continue
        centre_ms = (((m["start_us"] + m["end_us"]) * 0.5) - base_start) / 1000.0
        wl_nm = alignment.wavelength_for_ms(centre_ms)
        rgb = scalar_frame_to_rgb(frames[idx], wl_nm, cmf)
        out_path = out_dir / f"{prefix}_bin_{idx:03d}_{wl_nm:.0f}nm.png"
        plt.imsave(str(out_path), rgb)
        out_meta.append({"index": idx, "time_center_ms": float(centre_ms), "wavelength_nm": float(wl_nm), "path": str(out_path)})
    return out_meta


def save_gray_frames(
    frames: np.ndarray,
    metadata: List[Dict[str, float]],
    out_dir: Path,
    prefix: str,
    start_bin: int | None,
    end_bin: int | None,
) -> List[Dict[str, float]]:
    out_meta: List[Dict[str, float]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in metadata:
        idx = int(m["index"])
        if start_bin is not None and idx < start_bin:
            continue
        if end_bin is not None and idx > end_bin:
            continue
        frame = frames[idx].astype(np.float32)
        vmin = float(np.min(frame))
        vmax = float(np.max(frame))
        vabs = max(abs(vmin), abs(vmax))
        if not np.isfinite(vabs) or vabs <= 1e-12:
            img = np.full_like(frame, 0.5, dtype=np.float32)
        else:
            img = (frame + vabs) / (2.0 * vabs)
            img = np.clip(img, 0.0, 1.0)
        out_path = out_dir / f"{prefix}_bin_{idx:03d}.png"
        plt.imsave(str(out_path), img, cmap="gray")
        out_meta.append({"index": idx, "path": str(out_path)})
    return out_meta


def main() -> None:
    args = parse_args()
    segment_path = args.segment.resolve()
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    # Resolve alignment JSON (latest if omitted)
    align_path = args.alignment_json.resolve() if args.alignment_json else find_latest_alignment()
    alignment = load_alignment(align_path)

    # Output root
    figures_root = REPO_ROOT / "publication_code" / "figures"
    if args.output_dir is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = figures_root / f"figure05_{suffix}"
    out_root = ensure_outdir(args.output_dir)

    # Load CMFs once
    cmf = load_cie_cmf(REPO_ROOT)

    # Load events and, if parameters present, compensated times
    x, y, t, p = load_segment_events(segment_path)
    params_file = find_param_file(segment_path)
    if params_file is not None:
        params = load_params(params_file)
        t_comp, _, _ = compute_fast_comp_times(x.astype(np.float32), y.astype(np.float32), t, params)
    else:
        t_comp = t

    sensor_shape = (args.sensor_height, args.sensor_width)
    sensor_area = float(args.sensor_width * args.sensor_height)

    # Learn neg weight on a fine series (like Figure 4) using original times
    from publication_code.figure04_rescaled import auto_scale_neg_weight, RESCALE_FINE_STEP_US, smooth_volume_3d, subtract_background  # type: ignore
    neg_scale, _series = auto_scale_neg_weight(
        t,
        p,
        sensor_area=sensor_area,
        step_us=RESCALE_FINE_STEP_US,
        pos_scale=args.pos_scale,
        neg_scale_init=args.neg_scale,
        return_series=True,
    )
    comp_weights = np.where(p >= 0, args.pos_scale, -float(neg_scale)).astype(np.float32)

    exported: Dict[str, object] = {
        "segment": str(segment_path),
        "alignment_json": str(align_path),
        "alignment": {
            "slope_nm_per_ms": alignment.slope_nm_per_ms,
            "intercept_nm": alignment.intercept_nm,
        },
        "bin_sets": [],
    }

    # We recompute from events to ensure weighting + background removal match Fig. 4
    orig50_npz = comp50_npz = binwidth_npz = None

    for bw_us in args.bin_widths_us:
        bw_ms = bw_us / 1000.0
        set_entry: Dict[str, object] = {"bin_width_us": float(bw_us), "bin_width_ms": float(bw_ms), "modes": {}}

        # Build compensated, polarity-weighted frames per bin
        compensated, metadata = accumulate_bins_from_events(
            x, y, t, t_comp, comp_weights, bw_us, sensor_shape
        )

        # Subset bins if requested
        start_bin = args.start_bin
        end_bin = args.end_bin

        # Prepare out dirs
        bw_dir = ensure_outdir(out_root / ("%dms" % int(round(bw_ms))))
        # Build compensated volume with optional smoothing + background removal
        comp_vol = compensated
        if args.smooth and comp_vol.size:
            comp_vol = smooth_volume_3d(comp_vol)
        if args.remove_bg and comp_vol.size:
            comp_vol = np.stack([subtract_background(f) for f in comp_vol], axis=0)
        out_dir = ensure_outdir(bw_dir / "compensated")
        if args.per_bin_mode == "rgb":
            meta_c = save_rgb_frames(comp_vol, metadata, alignment, cmf, out_dir, "comp", start_bin, end_bin)
        else:
            meta_c = save_gray_frames(comp_vol, metadata, out_dir, "comp", start_bin, end_bin)
        set_entry["modes"]["compensated"] = meta_c

        # Composite overall RGB across all bins for this bin width
        try:
            overall_rgb = composite_rgb_from_series(comp_vol, metadata, alignment, cmf)
            overall_path = bw_dir / f"overall_rgb_{int(round(bw_ms))}ms_compensated.png"
            plt.imsave(str(overall_path), overall_rgb)
            set_entry["overall_rgb_path"] = str(overall_path)
        except Exception as exc:
            set_entry["overall_rgb_error"] = str(exc)

        exported["bin_sets"].append(set_entry)

    # Save summary JSON
    info_path = out_root / "figure05_summary.json"
    with info_path.open("w", encoding="utf-8") as fp:
        json.dump(exported, fp, indent=2)
    print(f"Saved summary: {info_path}")


if __name__ == "__main__":
    main()
