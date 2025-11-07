#!/usr/bin/env python3
"""Crop overlapped regions from reference and overlay images using saved alignment."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class Alignment:
    scale: float
    tx: float
    ty: float
    alpha: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop only the overlapping region using manual alignment data.")
    parser.add_argument("--alignment", type=Path, required=True, help="JSON file produced by manual_match_gui.py")
    parser.add_argument("--ref", type=Path, default=None, help="Reference image override (defaults to JSON path).")
    parser.add_argument("--overlay", type=Path, default=None, help="Overlay image override (defaults to JSON path).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save cropped outputs.")
    parser.add_argument("--prefix", type=str, default="", help="Optional output filename prefix.")
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="Blending alpha for composite preview.")
    return parser.parse_args()


def load_alignment(path: Path) -> Tuple[Alignment, Path, Path]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    transform = data["transform"]
    alignment = Alignment(scale=float(transform["scale"]), tx=float(transform["tx"]), ty=float(transform["ty"]),
                          alpha=float(transform.get("alpha", 1.0)))
    ref_path = Path(data["reference"]["path"])
    overlay_path = Path(data["overlay"]["path"])
    return alignment, ref_path, overlay_path


def load_reference(path: Path) -> np.ndarray:
    ref = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError(path)
    return ref


def load_overlay(path: Path) -> np.ndarray:
    overlay = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(path)

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    elif overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    elif overlay.shape[2] != 4:
        raise ValueError("Unsupported overlay channel count.")
    return overlay


def warp_overlay(overlay: np.ndarray, ref_shape: Tuple[int, int], alignment: Alignment) -> np.ndarray:
    ref_h, ref_w = ref_shape
    M = np.array([[alignment.scale, 0.0, alignment.tx],
                  [0.0, alignment.scale, alignment.ty]], dtype=np.float32)
    warped = cv2.warpAffine(
        overlay,
        M,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def crop_overlap(ref: np.ndarray, warped_overlay: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    alpha_channel = warped_overlay[..., 3]
    ys, xs = np.where(alpha_channel > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("No overlapping region detected after warping; check alignment parameters.")

    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)

    ref_crop = ref[y0:y1, x0:x1]
    overlay_crop = warped_overlay[y0:y1, x0:x1]
    return ref_crop, overlay_crop, (x0, y0, x1, y1)


def save_outputs(ref_crop: np.ndarray, overlay_crop: np.ndarray, bbox: Tuple[int, int, int, int],
                 out_dir: Path, prefix: str, blend_alpha: float) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    ref_path = out_dir / f"{prefix}ref_crop.png"
    overlay_path = out_dir / f"{prefix}overlay_crop.png"
    overlay_bgr = cv2.cvtColor(overlay_crop, cv2.COLOR_BGRA2BGR)
    blend = cv2.addWeighted(ref_crop, float(1 - blend_alpha), overlay_bgr, float(blend_alpha), 0.0)
    blend_path = out_dir / f"{prefix}blend.png"

    meta = {
        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
        "size": {"width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]},
    }

    meta_path = out_dir / f"{prefix}crop_metadata.json"

    cv2.imwrite(str(ref_path), ref_crop)
    cv2.imwrite(str(overlay_path), overlay_crop)
    cv2.imwrite(str(blend_path), blend)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"reference": ref_path, "overlay": overlay_path, "blend": blend_path, "metadata": meta_path}


def main() -> None:
    args = parse_args()
    alignment, ref_default, overlay_default = load_alignment(args.alignment)
    ref_path = args.ref if args.ref else ref_default
    overlay_path = args.overlay if args.overlay else overlay_default

    ref = load_reference(ref_path)
    overlay = load_overlay(overlay_path)
    warped_overlay = warp_overlay(overlay, ref.shape[:2], alignment)
    ref_crop, overlay_crop, bbox = crop_overlap(ref, warped_overlay)
    outputs = save_outputs(ref_crop, overlay_crop, bbox, args.output_dir, args.prefix, args.blend_alpha)

    print("Cropped overlap saved to:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    print(f"Overlap bounding box: {bbox}")



if __name__ == "__main__":
    main()
