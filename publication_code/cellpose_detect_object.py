#!/usr/bin/env python3
"""Locate the dominant object in an image using Cellpose and save mask + metadata."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect dominant object with Cellpose.")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store mask/overlay/JSON")
    parser.add_argument("--model-type", default="cyto2", help="Cellpose model type (default: cyto2)")
    parser.add_argument("--diameter", type=float, nargs="+", default=[None], help="Approximate object diameter(s) in pixels. You can pass multiple values to scan (e.g., --diameter 200 120 80).")
    parser.add_argument("--flow-thresh", type=float, default=0.4, help="Cellpose flow threshold")
    parser.add_argument("--cellprob-thresh", type=float, default=0.0, help="Cellpose cell probability threshold")
    parser.add_argument("--max-side", type=int, default=1024, help="Downscale so max(H,W)<=max_side before running Cellpose")
    parser.add_argument(
        "--selection-mode",
        choices=("largest", "smallest", "nearest"),
        default="largest",
        help="Choose largest area, smallest area, or region nearest to a reference point.",
    )
    parser.add_argument("--ref-x", type=float, default=None, help="Reference X (pixels) when selection mode is 'nearest'")
    parser.add_argument("--ref-y", type=float, default=None, help="Reference Y (pixels) when selection mode is 'nearest'")
    parser.add_argument("--nearest-default-center", action="store_true", help="Use image center when nearest mode lacks ref coords")
    parser.add_argument("--min-area", type=int, default=None, help="Discard detections smaller than this many pixels before selection")
    parser.add_argument("--max-area", type=int, default=None, help="Discard detections larger than this many pixels before selection")
    return parser.parse_args()


def ensure_cellpose():
    try:
        from cellpose import models  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Cellpose is required. Install with `pip install cellpose`.") from exc


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_if_needed(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return img, 1.0
    scale = max_side / max_dim
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def run_cellpose(
    img: np.ndarray,
    model_type: str,
    diameter: float | None,
    flow_thresh: float,
    cellprob_thresh: float,
) -> np.ndarray:
    from cellpose import models  # type: ignore

    model = models.Cellpose(model_type=model_type)
    masks, _, _, _ = model.eval(
        img,
        channels=[0, 0] if img.ndim == 2 else [0, 0],
        diameter=diameter,
        flow_threshold=flow_thresh,
        cellprob_threshold=cellprob_thresh,
    )
    if masks is None:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return masks.astype(np.int32)


def extract_regions(masks: np.ndarray) -> list[dict]:
    labels = np.unique(masks)
    regions = []
    for lab in labels:
        if lab <= 0:
            continue
        ys, xs = np.where(masks == lab)
        if ys.size == 0:
            continue
        centroid = (float(xs.mean()), float(ys.mean()))
        regions.append({"label": int(lab), "area": int(xs.size), "centroid": centroid})
    return regions


def select_region_mask(
    masks: np.ndarray,
    mode: str,
    ref_point: Tuple[float, float] | None,
    min_area: int | None,
    max_area: int | None,
) -> np.ndarray:
    regions_all = extract_regions(masks)
    regions = regions_all
    if min_area is not None:
        regions = [r for r in regions if r["area"] >= min_area]
    if max_area is not None:
        regions = [r for r in regions if r["area"] <= max_area]
    if not regions:
        regions = regions_all
    if not regions:
        return np.zeros_like(masks, dtype=np.uint8)
    if mode == "largest":
        label = max(regions, key=lambda r: r["area"])["label"]
    elif mode == "smallest":
        label = min(regions, key=lambda r: r["area"])["label"]
    else:
        if ref_point is None:
            raise ValueError("Nearest selection requires a reference point.")
        rx, ry = ref_point
        def dist(reg):
            cx, cy = reg["centroid"]
            return (cx - rx) ** 2 + (cy - ry) ** 2
        label = min(regions, key=dist)["label"]
    return (masks == label).astype(np.uint8)


def upscale_mask(mask_small: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    return cv2.resize(mask_small, (tw, th), interpolation=cv2.INTER_NEAREST)


@dataclass
class DetectionMetadata:
    image: str
    mask: str
    overlay: str
    bbox_xyxy: Tuple[int, int, int, int]
    bbox_rel: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    area_px: int


def save_outputs(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    src_path: Path,
) -> DetectionMetadata:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "cellpose_mask.png"
    overlay_path = out_dir / "cellpose_overlay.png"

    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        bbox = (0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1)
        centroid = (rgb.shape[1] / 2.0, rgb.shape[0] / 2.0)
        area = 0
    else:
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        bbox = (xmin, ymin, xmax, ymax)
        centroid = (float(xs.mean()), float(ys.mean()))
        area = int(xs.size)

    bbox_rel = (
        bbox[0] / rgb.shape[1],
        bbox[1] / rgb.shape[0],
        bbox[2] / rgb.shape[1],
        bbox[3] / rgb.shape[0],
    )

    meta = DetectionMetadata(
        image=str(src_path),
        mask=str(mask_path),
        overlay=str(overlay_path),
        bbox_xyxy=bbox,
        bbox_rel=bbox_rel,
        centroid=centroid,
        area_px=area,
    )
    (out_dir / "cellpose_detection.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    ensure_cellpose()

    rgb = load_image(args.image)
    resized, scale = resize_if_needed(rgb, args.max_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    diameters = args.diameter
    all_masks = []
    for d in diameters:
        masks_small = run_cellpose(
            gray,
            model_type=args.model_type,
            diameter=d * scale if (d is not None) else None,
            flow_thresh=args.flow_thresh,
            cellprob_thresh=args.cellprob_thresh,
        )
        if masks_small.max() > 0:
            all_masks.append(masks_small)

    if not all_masks:
        selected_small = np.zeros(gray.shape, dtype=np.uint8)
    else:
        masks_small = max(all_masks, key=lambda m: m.max())
        ref_point = None
        if args.selection_mode == "nearest":
            if args.ref_x is not None and args.ref_y is not None:
                ref_point = (args.ref_x * scale, args.ref_y * scale)
            elif args.nearest_default_center:
                ref_point = (resized.shape[1] / 2.0, resized.shape[0] / 2.0)
            else:
                raise SystemExit("Nearest selection requires --ref-x/--ref-y or --nearest-default-center.")

        selected_small = select_region_mask(
            masks_small,
            mode=args.selection_mode,
            ref_point=ref_point,
            min_area=args.min_area,
            max_area=args.max_area,
        )

    if scale != 1.0:
        mask = upscale_mask(selected_small, (rgb.shape[0], rgb.shape[1]))
    else:
        mask = selected_small

    save_outputs(rgb, mask, args.output_dir, args.image)
    print(f"Detection saved to {args.output_dir}")


if __name__ == "__main__":
    main()
