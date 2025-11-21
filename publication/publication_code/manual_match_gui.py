#!/usr/bin/env python3
"""Interactive manual registration tool for two images.

The reference image is displayed as the background. The overlay image can be
dragged and scaled to visually align the two. Press ``s`` to save the current
transform (scale and top-left translation) to JSON for downstream use.
"""
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
class AlignmentState:
    """Holds the interactive transform parameters."""

    scale: float = 1.0
    tx: float = 0.0  # top-left corner x (columns)
    ty: float = 0.0  # top-left corner y (rows)
    alpha: float = 0.6


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB float in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # Preserve user provided alpha channel.
        bgr = img[..., :3]
        alpha = img[..., 3:] / 255.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        return np.concatenate([rgb, alpha], axis=-1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual overlay alignment tool (drag + scale).")
    parser.add_argument("--ref", required=True, type=Path, help="Reference/background image (fixed).")
    parser.add_argument("--overlay", required=True, type=Path, help="Overlay image to drag and scale.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to store alignment parameters when pressing 's'.",
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="Initial overlay alpha (0-1).")
    return parser.parse_args()


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class ManualMatcher:
    """Interactive matplotlib widget that manages overlay alignment."""

    def __init__(self, ref: np.ndarray, overlay: np.ndarray, alpha: float):
        self.ref = ref
        self.overlay_input = overlay
        self.overlay_rgba, self.overlay_dims = self._prepare_overlay(overlay)
        self.state = AlignmentState(scale=1.0, tx=0.0, ty=0.0, alpha=clamp(alpha, 0.0, 1.0))

        self.is_dragging = False
        self.drag_start_mouse: Tuple[float, float] | None = None
        self.drag_start_tx: float = 0.0
        self.drag_start_ty: float = 0.0

        self.fig, self.ax = plt.subplots()
        self._setup_axes()
        self.overlay_artist = self.ax.imshow(
            self.overlay_rgba,
            origin="upper",
            alpha=self.state.alpha,
            extent=self._current_extent(),
            interpolation="nearest",
        )
        self._connect_callbacks()
        self._update_title()

    @staticmethod
    def _prepare_overlay(overlay: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Ensure overlay has an alpha channel and record its original dimensions."""
        if overlay.ndim == 2:
            overlay = np.stack([overlay, overlay, overlay], axis=-1)
        if overlay.shape[2] == 3:
            alpha = np.ones((*overlay.shape[:2], 1), dtype=overlay.dtype)
            overlay_rgba = np.concatenate([overlay, alpha], axis=-1)
        else:
            overlay_rgba = overlay
        h, w = overlay_rgba.shape[:2]
        return overlay_rgba, (w, h)

    def _setup_axes(self) -> None:
        h, w = self.ref.shape[:2]
        self.ax.imshow(self.ref, origin="upper")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (pixels)")
        self.ax.set_ylabel("y (pixels)")

    def _connect_callbacks(self) -> None:
        cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._callback_ids = [cid_press, cid_release, cid_motion, cid_scroll, cid_key]

    def _current_extent(self):
        w, h = self.overlay_dims
        scaled_w = w * self.state.scale
        scaled_h = h * self.state.scale
        x0 = self.state.tx
        x1 = self.state.tx + scaled_w
        y0 = self.state.ty
        y1 = self.state.ty + scaled_h
        # Matplotlib expects [xmin, xmax, ymin, ymax] even with origin="upper".
        return [x0, x1, y1, y0]

    def _update_overlay_artist(self) -> None:
        self.overlay_artist.set_extent(self._current_extent())
        self.overlay_artist.set_alpha(self.state.alpha)
        self.fig.canvas.draw_idle()
        self._update_title()

    def _update_title(self) -> None:
        title = (
            "Manual alignment | drag: left mouse | scale: scroll wheel | "
            "alpha: +/- | reset: r | save: s | quit: q\n"
            f"scale={self.state.scale:.3f}, tx={self.state.tx:.1f}, ty={self.state.ty:.1f}, alpha={self.state.alpha:.2f}"
        )
        self.ax.set_title(title)

    def _in_overlay(self, event) -> bool:
        if event.xdata is None or event.ydata is None:
            return False
        x0, x1, y1, y0 = self._current_extent()
        within = x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1
        return within

    def _on_press(self, event) -> None:
        if event.button != 1 or not self._in_overlay(event):
            return
        self.is_dragging = True
        self.drag_start_mouse = (event.xdata, event.ydata)
        self.drag_start_tx = self.state.tx
        self.drag_start_ty = self.state.ty

    def _on_motion(self, event) -> None:
        if not self.is_dragging or self.drag_start_mouse is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.drag_start_mouse[0]
        dy = event.ydata - self.drag_start_mouse[1]
        self.state.tx = self.drag_start_tx + dx
        self.state.ty = self.drag_start_ty + dy
        self._update_overlay_artist()

    def _on_release(self, event) -> None:
        if event.button == 1:
            self.is_dragging = False
            self.drag_start_mouse = None

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        factor = 1.1 if event.step > 0 else 1 / 1.1
        old_scale = self.state.scale
        new_scale = clamp(old_scale * factor, 0.05, 10.0)
        if np.isclose(new_scale, old_scale):
            return

        # Re-anchor around cursor position for intuitive zoom.
        rel_x = (event.xdata - self.state.tx) / (self.overlay_dims[0] * old_scale)
        rel_y = (event.ydata - self.state.ty) / (self.overlay_dims[1] * old_scale)
        rel_x = np.nan_to_num(rel_x, nan=0.5)
        rel_y = np.nan_to_num(rel_y, nan=0.5)

        self.state.tx = event.xdata - rel_x * self.overlay_dims[0] * new_scale
        self.state.ty = event.ydata - rel_y * self.overlay_dims[1] * new_scale
        self.state.scale = new_scale
        self._update_overlay_artist()

    def _on_key(self, event) -> None:
        if event.key == "q":
            plt.close(self.fig)
            return
        if event.key == "r":
            self.state = AlignmentState(alpha=self.state.alpha)
            self._update_overlay_artist()
            return
        if event.key == "s":
            plt.close(self.fig)
            self._saved = True
            return
        if event.key in ("+", "="):
            self.state.alpha = clamp(self.state.alpha + 0.05, 0.0, 1.0)
            self._update_overlay_artist()
            return
        if event.key in ("-", "_"):
            self.state.alpha = clamp(self.state.alpha - 0.05, 0.0, 1.0)
            self._update_overlay_artist()
            return

    def run(self) -> AlignmentState:
        self._saved = False
        plt.show()
        return self.state if getattr(self, "_saved", False) else None


def main() -> None:
    args = parse_args()
    ref = load_image(args.ref)
    overlay = load_image(args.overlay)

    tool = ManualMatcher(ref, overlay, alpha=args.alpha)
    saved_state = tool.run()

    if saved_state is None:
        print("Alignment window closed without saving (press 's' next time to export parameters).")
        return

    data = {
        "reference": {
            "path": str(args.ref),
            "width": ref.shape[1],
            "height": ref.shape[0],
        },
        "overlay": {
            "path": str(args.overlay),
            "orig_width": overlay.shape[1],
            "orig_height": overlay.shape[0],
        },
        "transform": asdict(saved_state),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved alignment parameters to {args.output_json}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
