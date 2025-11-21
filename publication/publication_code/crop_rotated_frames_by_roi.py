#!/usr/bin/env python3
"""Crop square ROIs from rotated frames using metadata from Cellpose detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop square ROI from rotated band frames.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing band_*.png frames")
    parser.add_argument("--roi-json", type=Path, required=True, help="Cellpose detection JSON with bbox_xyxy")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for cropped frames")
    parser.add_argument("--suffix", default="_roi", help="Suffix to append to filenames (default: _roi)")
    return parser.parse_args()


def load_square_bbox(meta_path: Path) -> Tuple[int, int, int, int]:
    meta = json.loads(meta_path.read_text())
    xs, ys = [], []
    if "mask" in meta:
        mask = cv2.imread(meta["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask > 0):
            ys, xs = np.where(mask > 0)
    if xs is None or len(xs) == 0:
        if "bbox_xyxy" not in meta:
            raise ValueError("ROI JSON missing 'bbox_xyxy'")
        xmin, ymin, xmax, ymax = map(int, meta["bbox_xyxy"])
    else:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    side = max(width, height)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half = side / 2.0
    x0 = int(np.floor(cx - half))
    y0 = int(np.floor(cy - half))
    x1 = x0 + int(side)
    y1 = y0 + int(side)
    return x0, y0, x1, y1


def crop_frames(frames_dir: Path, bbox: Tuple[int, int, int, int], out_dir: Path, suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = bbox
    for path in sorted(frames_dir.glob("band_*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: skipped unreadable frame {path}")
            continue
        h, w = img.shape[:2]
        xa0 = max(0, min(x0, w - 1))
        ya0 = max(0, min(y0, h - 1))
        xa1 = max(xa0 + 1, min(x1, w))
        ya1 = max(ya0 + 1, min(y1, h))
        crop = img[ya0:ya1, xa0:xa1]
        out_path = out_dir / f"{path.stem}{suffix}{path.suffix}"
        cv2.imwrite(str(out_path), crop)


def main() -> None:
    args = parse_args()
    bbox = load_square_bbox(args.roi_json)
    crop_frames(args.frames_dir, bbox, args.output_dir, args.suffix)
    print(f"Cropped frames saved to {args.output_dir} using bbox {bbox}")


if __name__ == "__main__":
    main()
