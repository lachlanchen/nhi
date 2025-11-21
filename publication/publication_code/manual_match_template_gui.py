#!/usr/bin/env python3
"""Manual alignment tool for overlaying a template image on a larger reference."""
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
class TemplateState:
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    alpha: float = 0.6


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return rgb.astype(np.float32) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive template matching (drag/scale overlay).")
    parser.add_argument("--ref", type=Path, required=True, help="Reference image (fixed background)")
    parser.add_argument("--template", type=Path, required=True, help="Template image to drag/scale")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to save alignment parameters")
    parser.add_argument("--alpha", type=float, default=0.6, help="Initial overlay alpha")
    return parser.parse_args()


class TemplateMatcher:
    def __init__(self, ref_img: np.ndarray, template_img: np.ndarray, alpha: float):
        self.ref = ref_img
        self.template = template_img
        self.state = TemplateState(alpha=np.clip(alpha, 0.0, 1.0))
        self.dragging = False
        self.drag_start = None
        self.drag_tx = 0.0
        self.drag_ty = 0.0

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.ref, origin="upper")
        self.ax.set_xlim(-10, self.ref.shape[1] + 10)
        self.ax.set_ylim(self.ref.shape[0] + 10, -10)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (px)")
        self.ax.set_ylabel("y (px)")

        self.template_artist = self.ax.imshow(
            self.template,
            origin="upper",
            alpha=self.state.alpha,
            extent=self._extent(),
        )
        self._connect()
        self._update_title()

    def _extent(self):
        h, w = self.template.shape[:2]
        scaled_w = w * self.state.scale
        scaled_h = h * self.state.scale
        x0 = self.state.tx
        y0 = self.state.ty
        return [x0, x0 + scaled_w, y0 + scaled_h, y0]

    def _connect(self):
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _in_overlay(self, event) -> bool:
        if event.xdata is None or event.ydata is None:
            return False
        x0, x1, y1, y0 = self._extent()
        return x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1

    def _on_press(self, event):
        if event.button != 1 or not self._in_overlay(event):
            return
        self.dragging = True
        self.drag_start = (event.xdata, event.ydata)
        self.drag_tx = self.state.tx
        self.drag_ty = self.state.ty

    def _on_motion(self, event):
        if not self.dragging or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.drag_start[0]
        dy = event.ydata - self.drag_start[1]
        self.state.tx = self.drag_tx + dx
        self.state.ty = self.drag_ty + dy
        self._update_artist()

    def _on_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.drag_start = None

    def _on_scroll(self, event):
        if event.step == 0:
            return
        factor = 1.1 if event.step > 0 else 1 / 1.1
        old_scale = self.state.scale
        new_scale = np.clip(old_scale * factor, 0.05, 20.0)
        if event.xdata is None or event.ydata is None:
            return
        if np.isclose(new_scale, old_scale):
            return
        x0, x1, y1, y0 = self._extent()
        cx = event.xdata
        cy = event.ydata
        h, w = self.template.shape[:2]
        rel_x = (cx - self.state.tx) / (w * old_scale)
        rel_y = (cy - self.state.ty) / (h * old_scale)
        self.state.tx = cx - rel_x * w * new_scale
        self.state.ty = cy - rel_y * h * new_scale
        self.state.scale = new_scale
        self._update_artist()

    def _on_key(self, event):
        if event.key == "q":
            self.saved = False
            plt.close(self.fig)
        elif event.key == "s":
            self.saved = True
            plt.close(self.fig)
        elif event.key == "r":
            self.state = TemplateState(alpha=self.state.alpha)
            self._update_artist()
        elif event.key in ("+", "="):
            self.state.alpha = np.clip(self.state.alpha + 0.05, 0.0, 1.0)
            self._update_artist()
        elif event.key in ("-", "_"):
            self.state.alpha = np.clip(self.state.alpha - 0.05, 0.0, 1.0)
            self._update_artist()

    def _update_artist(self):
        self.template_artist.set_extent(self._extent())
        self.template_artist.set_alpha(self.state.alpha)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        self.ax.set_title(
            "Drag overlay | scroll=scale | +/- alpha | r reset | s save | q quit\n"
            f"scale={self.state.scale:.3f}, tx={self.state.tx:.1f}, ty={self.state.ty:.1f}, alpha={self.state.alpha:.2f}",
            fontsize=9,
        )

    def run(self) -> TemplateState | None:
        self.saved = False
        plt.show()
        return self.state if self.saved else None


def main() -> None:
    args = parse_args()
    ref = load_image(args.ref)
    template = load_image(args.template)
    matcher = TemplateMatcher(ref, template, alpha=args.alpha)
    state = matcher.run()
    if state is None:
        print("Alignment canceled (no JSON saved).")
        return

    payload = {
        "reference": {
            "path": str(args.ref),
            "width": ref.shape[1],
            "height": ref.shape[0],
        },
        "template": {
            "path": str(args.template),
            "orig_width": template.shape[1],
            "orig_height": template.shape[0],
        },
        "transform": asdict(state),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved alignment to {args.output_json}")


if __name__ == "__main__":
    main()
