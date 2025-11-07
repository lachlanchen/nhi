#!/usr/bin/env python3
"""Crop matched region from reference and template using alignment JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop overlapping region from reference and template images based on saved transform.")
    parser.add_argument("--alignment-json", type=Path, required=True, help="JSON file produced by manual matcher")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save cropped images")
    parser.add_argument("--ref-out", default="ref_crop.png", help="Filename for cropped reference")
    parser.add_argument("--template-out", default="template_crop.png", help="Filename for cropped template")
    parser.add_argument("--flip-ref-vertical", action="store_true", help="Flip reference vertically before cropping (if alignment was done on flipped view)")
    return parser.parse_args()


def load_alignment(path: Path):
    data = json.loads(path.read_text())
    ref = Path(data["reference"]["path"])
    tpl = Path(data["template"]["path"])
    transform = data["transform"]
    return ref, tpl, transform


def crop_images(ref_path: Path, tpl_path: Path, transform: dict, out_dir: Path, ref_out: str, tpl_out: str, flip_ref: bool):
    ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    tpl = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
    if ref is None or tpl is None:
        raise FileNotFoundError("Failed to read input images.")
    if flip_ref:
        ref = cv2.flip(ref, 0)

    h_ref, w_ref = ref.shape[:2]
    h_tpl, w_tpl = tpl.shape[:2]
    scale = float(transform["scale"])
    tx = float(transform["tx"])
    ty = float(transform["ty"])

    x0 = tx
    y0 = ty
    x1 = tx + w_tpl * scale
    y1 = ty + h_tpl * scale

    ix0 = max(0, int(np.floor(x0)))
    iy0 = max(0, int(np.floor(y0)))
    ix1 = min(w_ref, int(np.ceil(x1)))
    iy1 = min(h_ref, int(np.ceil(y1)))
    if ix0 >= ix1 or iy0 >= iy1:
        raise ValueError("No overlap between transformed template and reference.")

    ref_crop = ref[iy0:iy1, ix0:ix1]

    tpl_x0 = max(0, int(np.floor((ix0 - tx) / scale)))
    tpl_y0 = max(0, int(np.floor((iy0 - ty) / scale)))
    tpl_x1 = min(w_tpl, int(np.ceil((ix1 - tx) / scale)))
    tpl_y1 = min(h_tpl, int(np.ceil((iy1 - ty) / scale)))
    tpl_crop = tpl[tpl_y0:tpl_y1, tpl_x0:tpl_x1]

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / ref_out), ref_crop)
    cv2.imwrite(str(out_dir / tpl_out), tpl_crop)

    meta = {
        "ref_crop": {
            "path": str(out_dir / ref_out),
            "x0": ix0,
            "y0": iy0,
            "x1": ix1,
            "y1": iy1,
        },
        "template_crop": {
            "path": str(out_dir / tpl_out),
            "x0": tpl_x0,
            "y0": tpl_y0,
            "x1": tpl_x1,
            "y1": tpl_y1,
        },
    }
    (out_dir / "crop_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Cropped images saved to {out_dir}")


def main() -> None:
    args = parse_args()
    ref_path, tpl_path, transform = load_alignment(args.alignment_json)
    crop_images(ref_path, tpl_path, transform, args.output_dir, args.ref_out, args.template_out, args.flip_ref_vertical)


if __name__ == "__main__":
    main()
