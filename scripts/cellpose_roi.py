#!/usr/bin/env python3
"""
Derive a reusable ROI via Cellpose from a single RGB image, save ROI mask +
metadata to JSON, and optionally apply the ROI to a folder of frames.

Usage examples
--------------
1) Compute ROI from an RGB and save mask + JSON under an output folder:
   python scripts/cellpose_roi.py \
       --rgb hyperspectral_data_sanqin_gt/test300_rgb_colorimetric_cropped.png \
       --out-dir publication_code/figures/roi_test300

2) Compute ROI once, then apply it to frames in a directory (e.g., Fig. 5):
   python scripts/cellpose_roi.py \
       --rgb hyperspectral_data_sanqin_gt/test300_rgb_colorimetric_cropped.png \
       --out-dir publication_code/figures/roi_test300 \
       --apply --frames-dir publication_code/figures/figure05_20251101_184527/5ms/compensated

Notes
-----
- The ROI mask is the union of all Cellpose masks by default; pass
  --largest-only to keep the largest detected object instead.
- The JSON stores key metadata (image path, bounding box, area, mask path) so
  subsequent pipelines can reuse the ROI without re-running Cellpose.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract Cellpose ROI and optionally apply to frames")
    ap.add_argument("--rgb", type=Path, required=True, help="Path to source RGB image for ROI")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to save ROI mask + JSON (created if missing)")
    ap.add_argument("--model-type", default="cyto2", help="Cellpose model type (default: cyto2)")
    ap.add_argument("--diameter", type=float, default=None, help="Approx cell diameter; None to auto")
    ap.add_argument("--flow-thresh", type=float, default=0.4, help="Flow error threshold")
    ap.add_argument("--cellprob-thresh", type=float, default=0.0, help="Cellprob threshold")
    ap.add_argument("--max-side", type=int, default=1024, help="Downscale input so max(H,W) <= this (speeds/stabilizes Cellpose)")
    ap.add_argument("--largest-only", action="store_true", help="Keep only the largest detected object as ROI")
    ap.add_argument("--square-only", action="store_true", help="Use square bounding box of ROI (no pixel-wise mask)")
    ap.add_argument("--apply", action="store_true", help="Apply ROI to a frames directory (mask/crop frames)")
    ap.add_argument("--frames-dir", type=Path, help="Directory containing frames (PNG) to mask")
    ap.add_argument("--frames-glob", default="*.png", help="Glob to select frames (default: *.png)")
    ap.add_argument("--masked-out", type=Path, help="Output directory for masked/cropped frames; defaults to <out-dir>/masked_frames")
    ap.add_argument("--apply-mode", choices=("mask", "crop"), default="crop", help="Apply mode when using --square-only: crop (default) or mask")
    return ap.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cellpose(img: np.ndarray, model_type: str, diameter: float | None, flow_th: float, cellprob_th: float) -> np.ndarray:
    try:
        from cellpose import models
    except Exception as e:  # pragma: no cover
        raise SystemExit("Cellpose not installed. Install with `pip install cellpose` in your env.") from e

    model = models.Cellpose(model_type=model_type)
    # channels=[0,0]: grayscale; Cellpose will internally handle 2D inputs
    masks, flows, styles, diams = model.eval(img, channels=[0, 0], diameter=diameter, 
                                             flow_threshold=flow_th, cellprob_threshold=cellprob_th)
    if masks is None or masks.size == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return masks.astype(np.int32)


def masks_to_roi(masks: np.ndarray, largest_only: bool) -> np.ndarray:
    H, W = masks.shape
    if masks.max() <= 0:
        return np.zeros((H, W), dtype=np.uint8)
    if largest_only:
        labels, counts = np.unique(masks[masks > 0], return_counts=True)
        lab = int(labels[np.argmax(counts)])
        roi = (masks == lab)
    else:
        roi = masks > 0
    return roi.astype(np.uint8)


def _resize_gray_nn(arr: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize for 2D arrays without adding dependencies."""
    import math
    H, W = arr.shape
    H2, W2 = new_hw
    y_idx = (np.arange(H2) * (H / H2)).astype(int)
    x_idx = (np.arange(W2) * (W / W2)).astype(int)
    y_idx = np.clip(y_idx, 0, H - 1)
    x_idx = np.clip(x_idx, 0, W - 1)
    return arr[y_idx][:, x_idx]


def save_roi(rgb_path: Path, roi_mask: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(out_dir)
    mask_png = out_dir / "roi_mask.png"
    plt.imsave(str(mask_png), roi_mask, cmap="gray")
    # Save outline visualization for sanity check
    overlay_png = out_dir / "roi_overlay.png"
    from matplotlib import patches  # type: ignore
    img = plt.imread(rgb_path)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(np.ma.masked_where(roi_mask == 0, roi_mask), cmap="autumn", alpha=0.35)
    ax.axis("off")
    fig.savefig(overlay_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ys, xs = np.where(roi_mask > 0)
    if xs.size:
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        area = int(xs.size)
    else:
        xmin = ymin = 0
        ymax = roi_mask.shape[0] - 1
        xmax = roi_mask.shape[1] - 1
        area = 0

    # Build a square bbox centered on the ROI bbox
    H, W = roi_mask.shape
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    side = int(max(xmax - xmin + 1, ymax - ymin + 1))
    half = side // 2
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(W - 1, x0 + side - 1)
    y1 = min(H - 1, y0 + side - 1)
    # Adjust start if we clipped the end
    x0 = max(0, x1 - side + 1)
    y0 = max(0, y1 - side + 1)

    # Normalized versions
    bbox_rel = [xmin / float(W), ymin / float(H), xmax / float(W), ymax / float(H)]
    bbox_square_rel = [x0 / float(W), y0 / float(H), x1 / float(W), y1 / float(H)]

    meta = {
        "source_image": str(rgb_path),
        "created": datetime.now().isoformat(),
        "image_shape": [int(roi_mask.shape[0]), int(roi_mask.shape[1])],
        "mask_png": str(mask_png),
        "overlay_png": str(overlay_png),
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_rel": bbox_rel,
        "bbox_square_xyxy": [x0, y0, x1, y1],
        "bbox_square_rel": bbox_square_rel,
        "area_pixels": area,
        "roi_mode": "largest" if area and (roi_mask.max() == 1 and (roi_mask.sum() == area)) else "union",
    }
    json_path = out_dir / "roi.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mask_png, json_path


def apply_roi_to_frames(mask_png: Path, frames_dir: Path, out_dir: Path, pattern: str) -> int:
    ensure_dir(out_dir)
    mask = plt.imread(mask_png)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0.5).astype(np.float32)
    count = 0
    for i, img_path in enumerate(sorted(frames_dir.glob(pattern))):
        img = plt.imread(img_path)
        # Resize mask to frame size on first iteration if shapes mismatch
        if i == 0 and (mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]):
            Ht, Wt = img.shape[0], img.shape[1]
            mask = _resize_gray_nn(mask, (Ht, Wt)).astype(np.float32)
        if img.ndim == 2:
            masked = img * mask
        else:
            masked = img.copy()
            for c in range(img.shape[2]):
                masked[..., c] = img[..., c] * mask
        out_path = out_dir / img_path.name
        plt.imsave(out_path, masked)
        count += 1
    return count


def apply_square_to_frames(meta_json: Path, frames_dir: Path, out_dir: Path, pattern: str, mode: str) -> int:
    data = json.loads(meta_json.read_text(encoding="utf-8"))
    rel = data.get("bbox_square_rel") or data.get("bbox_rel")
    if not rel:
        raise RuntimeError("ROI JSON missing bbox_square_rel/bbox_rel")
    x0r, y0r, x1r, y1r = [float(v) for v in rel]
    ensure_dir(out_dir)
    count = 0
    for img_path in sorted(frames_dir.glob(pattern)):
        img = plt.imread(img_path)
        H, W = img.shape[0], img.shape[1]
        x0 = max(0, min(W - 1, int(round(x0r * W))))
        y0 = max(0, min(H - 1, int(round(y0r * H))))
        x1 = max(0, min(W - 1, int(round(x1r * W))))
        y1 = max(0, min(H - 1, int(round(y1r * H))))
        if mode == "crop":
            cropped = img[y0 : y1 + 1, x0 : x1 + 1]
            out_path = out_dir / img_path.name
            plt.imsave(out_path, cropped)
        else:
            # mask mode: keep inside bbox, zero outside
            mask = np.zeros((H, W), dtype=np.float32)
            mask[y0 : y1 + 1, x0 : x1 + 1] = 1.0
            if img.ndim == 2:
                masked = img * mask
            else:
                masked = img.copy()
                for c in range(img.shape[2]):
                    masked[..., c] = img[..., c] * mask
            out_path = out_dir / img_path.name
            plt.imsave(out_path, masked)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    rgb_path = args.rgb.resolve()
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    out_dir = ensure_dir(args.out_dir.resolve())
    img = plt.imread(rgb_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    # Convert to grayscale for Cellpose input
    if img.ndim == 3:
        gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        gray = img

    # Optional downscale for stability/speed
    orig_H, orig_W = gray.shape
    scale = 1.0
    if args.max_side and max(orig_H, orig_W) > args.max_side:
        s = args.max_side / float(max(orig_H, orig_W))
        H2, W2 = int(round(orig_H * s)), int(round(orig_W * s))
        gray_small = _resize_gray_nn((gray * 255.0).astype(np.uint8), (H2, W2)).astype(np.float32) / 255.0
        scale = s
    else:
        gray_small = gray

    if str(args.model_type).lower() == "none":
        thr = np.percentile(gray_small, 75.0)
        masks_small = (gray_small > thr).astype(np.int32)
    else:
        try:
            masks_small = run_cellpose(gray_small, args.model_type, args.diameter, args.flow_thresh, args.cellprob_thresh)
        except Exception as e:
            # Fallback: simple threshold to produce a conservative ROI
            print(f"Cellpose failed ({e}); falling back to simple threshold segmentation.")
            thr = np.percentile(gray_small, 75.0)
            masks_small = (gray_small > thr).astype(np.int32)

    # Resize masks back to original size if we downscaled
    if scale != 1.0:
        masks = _resize_gray_nn(masks_small.astype(np.uint8), (orig_H, orig_W)).astype(np.int32)
    else:
        masks = masks_small
    roi_mask = masks_to_roi(masks, args.largest_only)
    mask_png, json_path = save_roi(rgb_path, roi_mask, out_dir)
    print(f"ROI saved: {mask_png}\nMetadata: {json_path}")

    if args.apply:
        if not args.frames_dir:
            raise SystemExit("--apply specified but --frames-dir is missing")
        frames_dir = args.frames_dir.resolve()
        if not frames_dir.is_dir():
            raise FileNotFoundError(frames_dir)
        out_apply = args.masked_out.resolve() if args.masked_out else out_dir / ("cropped_frames" if args.square_only and args.apply_mode == "crop" else "masked_frames")
        ensure_dir(out_apply)
        if args.square_only:
            n = apply_square_to_frames(json_path, frames_dir, out_apply, args.frames_glob, args.apply_mode)
        else:
            n = apply_roi_to_frames(mask_png, frames_dir, out_apply, args.frames_glob)
        print(f"Applied ROI to {n} frame(s): {out_apply}")


if __name__ == "__main__":
    main()
