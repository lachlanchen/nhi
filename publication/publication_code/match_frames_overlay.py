#!/usr/bin/env python3
"""Feature/template alignment between two frames with overlay output."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align two frames (feature or template) and create an overlay")
    parser.add_argument("--comp", type=Path, required=True, help="Path to compensated / target frame")
    parser.add_argument("--ref", type=Path, required=True, help="Path to reference frame")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save visualizations")
    parser.add_argument("--alpha", type=float, default=0.6, help="Blending factor for overlay (default 0.6)")
    parser.add_argument("--min-matches", type=int, default=12, help="Minimum matches required for feature alignment before falling back to template matching")
    return parser.parse_args()


def load_images(comp_path: Path, ref_path: Path):
    comp_color = cv2.imread(str(comp_path), cv2.IMREAD_COLOR)
    ref_color = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if comp_color is None:
        raise FileNotFoundError(comp_path)
    if ref_color is None:
        raise FileNotFoundError(ref_path)
    comp_gray = cv2.cvtColor(comp_color, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    return comp_color, comp_gray, ref_color, ref_gray


def detect_and_match(comp_gray: np.ndarray, ref_gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], int]:
    detector = None
    norm = None
    if hasattr(cv2, "SIFT_create"):
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
            norm = cv2.NORM_L2
        except Exception:
            detector = None
    if detector is None:
        detector = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(comp_gray, None)
    kp2, des2 = detector.detectAndCompute(ref_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return kp1 or [], kp2 or [], [], norm

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return kp1, kp2, good, norm


def feature_alignment(comp_color, ref_color, comp_gray, ref_gray, alpha, min_matches, matches_path, overlay_path):
    kp1, kp2, matches, _ = detect_and_match(comp_gray, ref_gray)
    if len(matches) < min_matches:
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return False

    h, w = comp_gray.shape
    warped_ref = cv2.warpPerspective(ref_color, H, (w, h))
    overlay = cv2.addWeighted(comp_color, 1.0 - alpha, warped_ref, alpha, 0)
    cv2.imwrite(str(overlay_path), overlay)

    matches_vis = cv2.drawMatches(comp_color, kp1, ref_color, kp2, matches, None,
                                  matchesMask=(mask.ravel().tolist() if mask is not None else None),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(matches_path), matches_vis)
    return True


def template_alignment(comp_color, ref_color, comp_gray, ref_gray, alpha, matches_path, overlay_path):
    result = cv2.matchTemplate(comp_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h_ref, w_ref = ref_gray.shape

    overlay = comp_color.copy()
    y0, y1 = top_left[1], top_left[1] + h_ref
    x0, x1 = top_left[0], top_left[0] + w_ref
    if y1 > overlay.shape[0] or x1 > overlay.shape[1]:
        y1 = min(y1, overlay.shape[0])
        x1 = min(x1, overlay.shape[1])
        ref_color = ref_color[: y1 - y0, : x1 - x0]
    roi = overlay[y0:y1, x0:x1]
    blended = cv2.addWeighted(roi, 1.0 - alpha, ref_color, alpha, 0)
    overlay[y0:y1, x0:x1] = blended
    cv2.imwrite(str(overlay_path), overlay)

    matches_vis = comp_color.copy()
    cv2.rectangle(matches_vis, top_left, (x1, y1), (0, 255, 0), 4)
    cv2.putText(matches_vis, f"template score: {max_val:.2f}", (top_left[0], max(0, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(matches_path), matches_vis)


def main() -> None:
    args = parse_args()
    comp_color, comp_gray, ref_color, ref_gray = load_images(args.comp, args.ref)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / f"{args.comp.stem}__{args.ref.stem}_overlay.png"
    matches_path = out_dir / f"{args.comp.stem}__{args.ref.stem}_matches.png"

    success = feature_alignment(comp_color, ref_color, comp_gray, ref_gray,
                                alpha=np.clip(args.alpha, 0.0, 1.0),
                                min_matches=args.min_matches,
                                matches_path=matches_path,
                                overlay_path=overlay_path)
    if success:
        print(f"Saved feature-based overlay to {overlay_path}")
        print(f"Saved match visualization to {matches_path}")
        return

    print("Feature alignment failed; falling back to template matching.")
    template_alignment(comp_color, ref_color, comp_gray, ref_gray,
                       alpha=np.clip(args.alpha, 0.0, 1.0),
                       matches_path=matches_path,
                       overlay_path=overlay_path)
    print(f"Saved template overlay to {overlay_path}")
    print(f"Saved template visualization to {matches_path}")


if __name__ == "__main__":
    main()
