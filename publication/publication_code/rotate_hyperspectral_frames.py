#!/usr/bin/env python3
"""Rotate an ENVI hyperspectral cube and dump each band to PNG frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


HDR_DTYPE_MAP: Dict[int, np.dtype] = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.uint16,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_hdr(path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith(";") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip().lower()] = value.strip()
    return meta


def parse_int(meta: Dict[str, str], key: str) -> int:
    try:
        return int(meta[key])
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' in header") from exc


def parse_float_list(field: str) -> List[float]:
    stripped = field.strip("{}")
    if not stripped:
        return []
    return [float(item.strip()) for item in stripped.split(",") if item.strip()]


def load_cube(bin_path: Path, meta: Dict[str, str]) -> Tuple[np.ndarray, List[float]]:
    samples = parse_int(meta, "samples")
    lines = parse_int(meta, "lines")
    bands = parse_int(meta, "bands")
    interleave = meta.get("interleave", "bil").lower()
    dtype_code = int(meta.get("data type", "4"))
    dtype = HDR_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported data type code {dtype_code}")

    data = np.fromfile(bin_path, dtype=dtype)
    expected = samples * lines * bands
    if data.size != expected:
        raise ValueError(f"Unexpected file size: expected {expected}, found {data.size}")

    if interleave == "bil":
        cube = data.reshape(lines, bands, samples).transpose(1, 0, 2)
    elif interleave == "bip":
        cube = data.reshape(lines, samples, bands).transpose(2, 0, 1)
    elif interleave == "bsq":
        cube = data.reshape(bands, lines, samples)
    else:
        raise ValueError(f"Unsupported interleave '{interleave}'")

    wavelengths = parse_float_list(meta.get("wavelength", ""))
    if wavelengths and len(wavelengths) != bands:
        raise ValueError("Wavelength list length does not match number of bands")

    return cube.astype(np.float32), wavelengths


def rotate_frame(
    frame: np.ndarray,
    angle_deg: float,
    expand: bool = True,
    crop_valid: bool = True,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    h, w = frame.shape
    cx, cy = center if center is not None else (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    size = (w, h)
    if expand:
        corners = np.array(
            [
                [0.0, 0.0],
                [float(w), 0.0],
                [float(w), float(h)],
                [0.0, float(h)],
            ],
            dtype=np.float32,
        )
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.concatenate([corners, ones], axis=1)
        matrix_full = np.vstack([matrix, [0.0, 0.0, 1.0]])
        transformed = (matrix_full @ corners_h.T).T
        min_x = float(transformed[:, 0].min())
        max_x = float(transformed[:, 0].max())
        min_y = float(transformed[:, 1].min())
        max_y = float(transformed[:, 1].max())
        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))
        matrix[0, 2] -= min_x
        matrix[1, 2] -= min_y
        size = (max(new_w, 1), max(new_h, 1))

    rotated = cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if crop_valid:
        mask = np.ones((h, w), dtype=np.uint8)
        rotated_mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.where(rotated_mask > 0)
        if ys.size and xs.size:
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            rotated = rotated[y0:y1, x0:x1]
    return rotated


def save_frame(path: Path, frame: np.ndarray, normalize: bool = True) -> None:
    data = frame.astype(np.float32)
    if normalize:
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if np.isclose(vmax, vmin):
            scaled = np.zeros_like(data)
        else:
            scaled = (data - vmin) / (vmax - vmin)
        frame_uint16 = np.clip(scaled * np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        frame_uint16 = np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), frame_uint16)


def infer_center_from_roi(roi_path: Path, samples: int, lines: int) -> Tuple[float, float]:
    with roi_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    bbox = payload.get("bbox_xyxy")
    image_shape = payload.get("image_shape")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"ROI JSON {roi_path} missing bbox_xyxy")
    if not image_shape or len(image_shape) != 2:
        raise ValueError(f"ROI JSON {roi_path} missing image_shape")
    img_h, img_w = float(image_shape[0]), float(image_shape[1])
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image_shape in {roi_path}")
    x0, y0, x1, y1 = map(float, bbox)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    scale_x = samples / img_w
    scale_y = lines / img_h
    return cx * scale_x, cy * scale_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate hyperspectral cube and dump PNG frames.")
    parser.add_argument("--hdr", type=Path, required=True, help="Path to ENVI .hdr file")
    parser.add_argument("--binary", type=Path, required=True, help="Path to raw binary (.spe) file")
    parser.add_argument("--angle", type=float, required=True, help="Rotation angle in degrees (CCW)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store rotated frames")
    parser.add_argument("--keep-size", action="store_true", help="Keep original frame size (skip canvas expansion)")
    parser.add_argument("--limit-bands", type=int, default=None, help="Optional limit on number of bands to export")
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-frame min-max scaling")
    parser.add_argument("--no-crop", action="store_true", help="Disable valid-region cropping after rotation")
    parser.add_argument("--center-x", type=float, default=None, help="Custom rotation center X (original pixels)")
    parser.add_argument("--center-y", type=float, default=None, help="Custom rotation center Y (original pixels)")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI match JSON (with bbox_xyxy + image_shape) for center")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = parse_hdr(args.hdr)
    samples = parse_int(meta, "samples")
    lines = parse_int(meta, "lines")
    cube, wavelengths = load_cube(args.binary, meta)
    bands = cube.shape[0]
    num_export = min(bands, args.limit_bands) if args.limit_bands else bands

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize = not args.no_normalize
    crop_valid = (not args.no_crop) and (not args.keep_size)

    center_x = args.center_x
    center_y = args.center_y
    roi_path = args.roi_json
    if (center_x is None or center_y is None) and roi_path is not None:
        cx_roi, cy_roi = infer_center_from_roi(roi_path, samples, lines)
        if center_x is None:
            center_x = cx_roi
        if center_y is None:
            center_y = cy_roi
    center_tuple: Optional[Tuple[float, float]] = None
    if center_x is not None and center_y is not None:
        center_tuple = (float(center_x), float(center_y))

    for band_idx in range(num_export):
        frame = cube[band_idx]
        rotated = rotate_frame(
            frame,
            args.angle,
            expand=not args.keep_size,
            crop_valid=crop_valid,
            center=center_tuple,
        )
        wl = f"_{wavelengths[band_idx]:.0f}nm" if wavelengths and band_idx < len(wavelengths) else ""
        out_name = f"band_{band_idx:03d}{wl}.png"
        save_frame(output_dir / out_name, rotated, normalize=normalize)

    summary = {
        "hdr": str(args.hdr),
        "binary": str(args.binary),
        "angle_deg": args.angle,
        "keep_size": bool(args.keep_size),
        "bands_exported": num_export,
        "output_dir": str(output_dir),
        "normalized": normalize,
        "cropped": crop_valid,
        "rotation_center": {"x": center_tuple[0], "y": center_tuple[1]} if center_tuple else "image_center",
        "roi_json": str(roi_path) if roi_path else None,
    }
    (output_dir / "rotation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rotated and saved {num_export} bands to {output_dir}")


if __name__ == "__main__":
    main()
