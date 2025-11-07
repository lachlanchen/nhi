#!/usr/bin/env python3
"""Replace gradient/reference rows in a spectral reconstruction folder and rerender the figure."""
from __future__ import annotations

import argparse
import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap rows 3-4 with rotated data and rerender figure.")
    parser.add_argument("--base-figure-dir", type=Path, required=True, help="Existing spectral reconstruction folder (with *_used_frames)")
    parser.add_argument("--gradient-dir", type=Path, required=True, help="Directory containing rotated gradient PNGs")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing rotated reference PNGs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder for updated figure")
    parser.add_argument("--save-png", action="store_true", help="Also save PNG when rerendering")
    return parser.parse_args()


PAT_NM = re.compile(r"_(\d+(?:\.\d+)?)nm", re.IGNORECASE)
PAT_BIN = re.compile(r"_(\d+(?:\.\d+)?)[^_]*to(\d+(?:\.\d+)?)", re.IGNORECASE)


def list_targets(orig_used_dir: Path) -> List[Tuple[float, Path]]:
    items = []
    for path in sorted(orig_used_dir.glob("*.png")):
        match = PAT_NM.search(path.stem)
        if match:
            items.append((float(match.group(1)), path))
    return items


def build_gradient_map(folder: Path) -> Dict[float, Path]:
    mapping = {}
    for path in folder.glob("*.png"):
        nm_val = None
        m_nm = PAT_NM.search(path.stem)
        if m_nm:
            nm_val = float(m_nm.group(1))
        else:
            m_bin = PAT_BIN.search(path.stem)
            if m_bin:
                start = float(m_bin.group(1))
                end = float(m_bin.group(2))
                nm_val = (start + end) * 0.5
        if nm_val is not None:
            mapping[nm_val] = path
    return mapping


def build_reference_map(folder: Path) -> Dict[float, Path]:
    mapping = {}
    for path in folder.glob("*.png"):
        m = PAT_NM.search(path.stem)
        if not m:
            continue
        nm = float(m.group(1))
        mapping[nm] = path
    return mapping


def choose_closest(mapping: Dict[float, Path], target_nm: float) -> Path:
    if not mapping:
        raise FileNotFoundError("No files in mapping for selection.")
    keys = np.array(list(mapping.keys()), dtype=np.float32)
    idx = int(np.argmin(np.abs(keys - target_nm)))
    return mapping[float(keys[idx])]


def replace_images(dest_dir: Path, files: List[str], source_path: Path) -> None:
    src_img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if src_img is None:
        raise FileNotFoundError(source_path)
    for dest in files:
        dest_path = dest_dir / dest
        if not dest_path.exists():
            continue
        dst_img = cv2.imread(str(dest_path), cv2.IMREAD_UNCHANGED)
        if dst_img is None:
            continue
        h, w = dst_img.shape[:2]
        resized = cv2.resize(src_img, (w, h), interpolation=cv2.INTER_CUBIC)
        if dst_img.ndim == 2 and resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if dst_img.ndim == 3 and resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dest_path), resized)


def main() -> None:
    args = parse_args()
    base_dir = args.base_figure_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)

    targets = list_targets(out_dir / "orig_used_frames")
    if not targets:
        raise RuntimeError("Could not determine wavelengths from orig_used_frames.")

    gradient_map = build_gradient_map(args.gradient_dir.resolve())
    reference_map = build_reference_map(args.reference_dir.resolve())

    for nm, orig_path in targets:
        basename = orig_path.name
        # gradient replacements
        grad_src = choose_closest(gradient_map, nm)
        replace_images(out_dir / "diff_used_frames", [basename], grad_src)
        replace_images(out_dir / "diff_selected_frames", [basename], grad_src)
        # reference replacements
        ref_src = choose_closest(reference_map, nm)
        replace_images(out_dir / "ref_used_frames", [basename], ref_src)
        replace_images(out_dir / "ref_selected_frames", [basename], ref_src)

    weights_json = out_dir / "figure04_rescaled_weights.json"
    if not weights_json.exists():
        raise FileNotFoundError(weights_json)
    weights = json.loads(weights_json.read_text())
    segment_path = Path(weights["segment"])
    if not segment_path.exists():
        raise FileNotFoundError(segment_path)

    figures_dir = out_dir.parent
    figure_name = f"{out_dir.name}"
    diff_sel = out_dir / "diff_selected_frames"
    ref_sel = out_dir / "ref_selected_frames"

    cmd = [
        "python",
        "publication_code/spectral_reconstruction_figure.py",
        "--segment",
        str(segment_path),
        "--diff-frames-dir",
        str(diff_sel),
        "--ref-frames-dir",
        str(ref_sel),
        "--gt-dir",
        "groundtruth_spectrum_2835",
        "--show-wavelength",
        "--save-png",
        "--figure-name",
        figure_name,
    ]

    import subprocess

    env = dict(**os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)

    generated_dirs = sorted(
        figures_dir.glob(f"{figure_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not generated_dirs:
        raise FileNotFoundError("Could not find generated figure folder.")
    latest = generated_dirs[0]
    stem = Path(figure_name).stem
    src_pdf = latest / f"{stem}.pdf"
    src_png = latest / f"{stem}.png"
    if src_pdf.exists():
        shutil.copy2(src_pdf, out_dir / "spectral_reconstruction_scan.pdf")
    if src_png.exists():
        shutil.copy2(src_png, out_dir / "spectral_reconstruction_scan.png")
    print("Updated figure saved to", out_dir)

if __name__ == "__main__":
    main()
