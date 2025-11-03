#!/usr/bin/env python3
"""
Template-match a smaller RGB (template) inside a larger RGB (search image),
derive a square ROI from the best match, save it to JSON, and optionally apply
the crop to a directory of frames.

Example
-------
python scripts/roi_template_match.py \
  --template hyperspectral_data_sanqin_gt/test300_rgb_colorimetric_cropped.png \
  --search   hyperspectral_data_sanqin_gt/test300_roi_square_rgb.png \
  --out-dir  hyperspectral_data_sanqin_gt/roi_match_test300 \
  --scales 1.0,0.95,1.05 \
  --apply-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames \
  --apply-out hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched

Notes
-----
- Uses OpenCV normalized template matching (TM_CCOEFF_NORMED). If OpenCV is not
  installed in your Python env, install it (e.g., `pip install opencv-python-headless`).
- The saved JSON includes absolute pixel bbox and relative bbox fractions so it
  can be reapplied to other images of the same dimensions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Template-match ROI and optionally crop frames")
    ap.add_argument("--template", type=Path, required=True, help="Path to smaller (template) RGB image")
    ap.add_argument("--search", type=Path, required=True, help="Path to larger (search) RGB image")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to save ROI JSON and debug overlay")
    ap.add_argument("--scales", default="1.0", help="Comma-separated template scales to try (e.g., 0.9,1.0,1.1)")
    ap.add_argument("--force-square", action="store_true", help="Make ROI a square centered on matched bbox")
    ap.add_argument("--square-scale", type=float, default=1.0, help="Scale side length of square ROI (default 1.0)")
    ap.add_argument("--margin", type=int, default=0, help="Pixels to expand (+) or shrink (-) bbox before squaring")
    ap.add_argument("--apply-dir", type=Path, help="Optional: directory with frames to crop (PNG/JPG)")
    ap.add_argument("--apply-glob", default="*.png", help="Glob for frames in --apply-dir (default: *.png)")
    ap.add_argument("--apply-out", type=Path, help="Optional: output dir for cropped frames")
    return ap.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread_rgb(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(
            "OpenCV is required. Install with `pip install opencv-python-headless` in your env."
        ) from e
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        # grayscale -> RGB
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    # BGR->RGB
    img = img[:, :, ::-1]
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    # luminance
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray.astype(np.float32)


def match_template(search_gray: np.ndarray, template_gray: np.ndarray, scales: List[float]) -> Tuple[int, int, int, int, float, float]:
    """Return best bbox (x0, y0, w, h), best score, best scale."""
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "OpenCV is required. Install with `pip install opencv-python-headless`."
        ) from e

    Hs, Ws = search_gray.shape
    best = None  # (score, x, y, w, h, scale)
    for s in scales:
        if s == 1.0:
            templ = template_gray
        else:
            Ht, Wt = template_gray.shape
            H2, W2 = max(1, int(round(Ht * s))), max(1, int(round(Wt * s)))
            templ = cv2.resize(template_gray, (W2, H2), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        if templ.shape[0] > Hs or templ.shape[1] > Ws:
            continue
        res = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        x, y = int(maxLoc[0]), int(maxLoc[1])
        w, h = int(templ.shape[1]), int(templ.shape[0])
        cand = (float(maxVal), x, y, w, h, float(s))
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        raise RuntimeError("Template larger than search image at all scales or no match computed.")
    score, x, y, w, h, s = best
    return x, y, w, h, score, s


def clamp_square_bbox(x: int, y: int, w: int, h: int, H: int, W: int, margin: int, force_square: bool, square_scale: float) -> Tuple[int, int, int, int]:
    # expand/shrink with margin
    x0, y0 = x - margin, y - margin
    x1, y1 = x + w - 1 + margin, y + h - 1 + margin
    # ensure order
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    # clamp to image
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W - 1, x1)
    y1 = min(H - 1, y1)

    if not force_square:
        return x0, y0, x1, y1

    # build square centered on current bbox center
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = int(round(max(x1 - x0 + 1, y1 - y0 + 1) * square_scale))
    half = side // 2
    xs = max(0, int(round(cx - half)))
    ys = max(0, int(round(cy - half)))
    xe = min(W - 1, xs + side - 1)
    ye = min(H - 1, ys + side - 1)
    xs = max(0, xe - side + 1)
    ys = max(0, ye - side + 1)
    return xs, ys, xe, ye


def save_debug_and_json(search_rgb: np.ndarray, bbox: Tuple[int, int, int, int], meta: dict, out_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(out_dir)
    x0, y0, x1, y1 = bbox
    H, W = search_rgb.shape[:2]
    rel = [x0 / float(W), y0 / float(H), x1 / float(W), y1 / float(H)]
    meta = dict(meta)
    meta.update(
        bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
        bbox_rel=rel,
        created=datetime.now().isoformat(),
        image_shape=[int(H), int(W)],
    )
    json_path = out_dir / "roi_match.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Debug overlay
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "OpenCV is required. Install with `pip install opencv-python-headless`."
        ) from e
    overlay = np.ascontiguousarray(search_rgb[:, :, ::-1])  # to BGR (contiguous) for cv2
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    dbg_path = out_dir / "roi_match_overlay.png"
    cv2.imwrite(str(dbg_path), overlay)
    return dbg_path, json_path


def apply_crop(bbox: Tuple[int, int, int, int], frames_dir: Path, out_dir: Path, pattern: str) -> int:
    ensure_dir(out_dir)
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(
            "OpenCV is required. Install with `pip install opencv-python-headless`."
        ) from e
    count = 0
    x0, y0, x1, y1 = bbox
    for img_path in sorted(frames_dir.glob(pattern)):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        H, W = img.shape[:2]
        # clamp bbox to current image size
        xx0 = max(0, min(W - 1, x0))
        yy0 = max(0, min(H - 1, y0))
        xx1 = max(0, min(W - 1, x1))
        yy1 = max(0, min(H - 1, y1))
        crop = img[yy0 : yy1 + 1, xx0 : xx1 + 1]
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), crop)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    templ_rgb = imread_rgb(args.template)
    search_rgb = imread_rgb(args.search)
    templ_gray = to_gray(templ_rgb)
    search_gray = to_gray(search_rgb)

    scales = [float(s) for s in str(args.scales).split(',') if s.strip()]
    x, y, w, h, score, used_scale = match_template(search_gray, templ_gray, scales)

    H, W = search_gray.shape
    x0, y0, x1, y1 = clamp_square_bbox(x, y, w, h, H, W, args.margin, args.force_square, args.square_scale)

    meta = {
        "template": str(args.template.resolve()),
        "search": str(args.search.resolve()),
        "match_score": float(score),
        "match_scale": float(used_scale),
        "method": "cv2.TM_CCOEFF_NORMED",
        "force_square": bool(args.force_square),
        "square_scale": float(args.square_scale),
        "margin": int(args.margin),
    }
    dbg_path, json_path = save_debug_and_json(search_rgb, (x0, y0, x1, y1), meta, out_dir)
    print(f"ROI match overlay: {dbg_path}\nROI JSON: {json_path}\nScore: {score:.4f}, scale: {used_scale:.3f}")

    if args.apply_dir and args.apply_out:
        n = apply_crop((x0, y0, x1, y1), args.apply_dir, ensure_dir(args.apply_out), args.apply_glob)
        print(f"Cropped {n} frame(s) into: {args.apply_out}")


if __name__ == "__main__":
    main()
