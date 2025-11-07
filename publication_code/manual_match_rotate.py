#!/usr/bin/env python3
"""Interactive rotation-only alignment tool with shared zoom."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RotateState:
    angle_deg: float = 0.0
    alpha: float = 0.6
    view_zoom: float = 1.0
    overlay_on_top: bool = True


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual rotation aligner with unified zoom.")
    parser.add_argument("--ref", type=Path, required=True, help="Reference/background image.")
    parser.add_argument("--overlay", type=Path, required=True, help="Overlay image to rotate.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to save angle/zoom/alpha (press 's' in the UI).",
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="Initial overlay alpha.")
    parser.add_argument("--deg-step", type=float, default=1.0, help="Degrees per scroll tick.")
    parser.add_argument(
        "--zoom-step",
        type=float,
        default=1.2,
        help="Multiplicative zoom step for 'z'/'x' keys (default 1.2).",
    )
    return parser.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return rgb.astype(np.float32) / 255.0

    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    if img.shape[2] == 4:
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        alpha = img[..., 3:].astype(np.float32) / 255.0
        return np.concatenate([rgb.astype(np.float32) / 255.0, alpha], axis=-1)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def ensure_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=-1)
    return image


class RotationAligner:
    """Matplotlib-based interactive rotation aligner."""

    def __init__(self, ref: np.ndarray, overlay: np.ndarray, alpha: float, deg_step: float, zoom_step: float):
        self.ref = ref[..., :3]
        self.overlay_orig = ensure_rgba(overlay)

        self.ref_h, self.ref_w = self.ref.shape[:2]
        self.overlay = self._match_dimensions(self.overlay_orig, (self.ref_h, self.ref_w))
        self.state = RotateState(alpha=clamp(alpha, 0.0, 1.0))

        self.deg_step = deg_step
        self.zoom_step = zoom_step
        self.current_overlay = self.overlay.copy()

        self.fig, self.ax = plt.subplots()
        self._setup_axes()
        self.overlay_artist = self.ax.imshow(
            self._current_overlay_rgb(),
            origin="upper",
            alpha=self.state.alpha,
            interpolation="nearest",
            zorder=2,
        )
        self._connect_events()
        self._update_title()

    @staticmethod
    def _match_dimensions(overlay: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_shape
        h, w = overlay.shape[:2]
        if (h, w) == (target_h, target_w):
            return overlay
        resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _setup_axes(self) -> None:
        self.ref_artist = self.ax.imshow(self.ref, origin="upper", zorder=1)
        self.ax.set_xlim(0, self.ref_w)
        self.ax.set_ylim(self.ref_h, 0)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (pixels)")
        self.ax.set_ylabel("y (pixels)")

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _current_overlay_rgb(self) -> np.ndarray:
        return self.current_overlay[..., :3]

    def _rotate_overlay(self) -> None:
        center = (self.ref_w / 2.0, self.ref_h / 2.0)
        M = cv2.getRotationMatrix2D(center, self.state.angle_deg, 1.0)
        rotated = cv2.warpAffine(
            self.overlay,
            M,
            (self.ref_w, self.ref_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )
        self.current_overlay = rotated

    def _update_axes_limits(self) -> None:
        zoom = clamp(self.state.view_zoom, 1.0, 50.0)
        cx, cy = self.ref_w / 2.0, self.ref_h / 2.0
        half_w = (self.ref_w / 2.0) / zoom
        half_h = (self.ref_h / 2.0) / zoom
        self.ax.set_xlim(cx - half_w, cx + half_w)
        self.ax.set_ylim(cy + half_h, cy - half_h)

    def _update_display(self) -> None:
        self.overlay_artist.set_data(self._current_overlay_rgb())
        self.overlay_artist.set_alpha(self.state.alpha)
        if self.state.overlay_on_top:
            self.overlay_artist.set_zorder(2)
            self.ref_artist.set_zorder(1)
        else:
            self.overlay_artist.set_zorder(1)
            self.ref_artist.set_zorder(2)
        self._update_axes_limits()
        self.fig.canvas.draw_idle()
        self._update_title()

    def _update_title(self) -> None:
        layer = "overlay" if self.state.overlay_on_top else "reference"
        title = (
            "Rotate overlay with mouse wheel | zoom view: z/x | alpha: +/- | toggle layer: t | reset: r | save: s | quit: q\n"
            f"angle={self.state.angle_deg:.2f}°, alpha={self.state.alpha:.2f}, zoom={self.state.view_zoom:.2f}×, top={layer}"
        )
        self.ax.set_title(title)

    def _on_scroll(self, event) -> None:
        direction = np.sign(event.step)
        if direction == 0:
            return
        self.state.angle_deg = (self.state.angle_deg + direction * self.deg_step) % 360.0
        self._rotate_overlay()
        self._update_display()

    def _on_key(self, event) -> None:
        key = event.key.lower()
        if key == "q":
            plt.close(self.fig)
            self._saved = False
            return
        if key == "s":
            plt.close(self.fig)
            self._saved = True
            return
        if key == "r":
            self.state = RotateState(alpha=self.state.alpha)
            self.state.alpha = clamp(self.state.alpha, 0.0, 1.0)
            self._rotate_overlay()
            self._update_display()
            return
        if key in ("+", "="):
            self.state.alpha = clamp(self.state.alpha + 0.05, 0.0, 1.0)
            self._update_display()
            return
        if key in ("-", "_"):
            self.state.alpha = clamp(self.state.alpha - 0.05, 0.0, 1.0)
            self._update_display()
            return
        if key == "z":
            self.state.view_zoom = clamp(self.state.view_zoom * self.zoom_step, 1.0, 50.0)
            self._update_display()
            return
        if key == "x":
            self.state.view_zoom = clamp(self.state.view_zoom / self.zoom_step, 1.0, 50.0)
            self._update_display()
            return
        if key == "t":
            self.state.overlay_on_top = not self.state.overlay_on_top
            self._update_display()
            return

    def run(self) -> RotateState | None:
        self._saved = False
        self._rotate_overlay()
        self._update_display()
        plt.show()
        return self.state if self._saved else None


def main() -> None:
    args = parse_args()
    ref = load_image_rgb(args.ref)
    overlay = load_image_rgb(args.overlay)
    tool = RotationAligner(ref, overlay, alpha=clamp(args.alpha, 0.0, 1.0),
                           deg_step=args.deg_step, zoom_step=args.zoom_step)
    result = tool.run()
    if result is None:
        print("Closed without saving (press 's' to export angle/zoom next time).")
        return

    output = {
        "reference": {"path": str(args.ref), "width": tool.ref_w, "height": tool.ref_h},
        "overlay": {"path": str(args.overlay), "width": tool.ref_w, "height": tool.ref_h},
        "transform": asdict(result),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved rotation parameters to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
