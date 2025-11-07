#!/usr/bin/env python3
"""Interactive template-over-reference matcher with draggable/scaleable overlay."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


@dataclass
class Transform:
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    alpha: float = 0.6


class ManualMatcher:
    def __init__(self, ref: np.ndarray, tpl: np.ndarray, alpha: float):
        self.ref = ref
        self.tpl = tpl
        self.state = Transform(alpha=np.clip(alpha, 0.0, 1.0))
        self.dragging = False
        self.drag_start = None
        self.drag_origin = (0.0, 0.0)

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.ref, origin="upper")
        self.ax.set_xlim(-0.5, self.ref.shape[1] - 0.5)
        self.ax.set_ylim(self.ref.shape[0] - 0.5, -0.5)
        self.ax.set_aspect("equal")

        self.overlay = self.ax.imshow(self.tpl, origin="upper", alpha=self.state.alpha, extent=self._extent())
        self._connect()

    def _extent(self):
        h, w = self.tpl.shape[:2]
        sw, sh = w * self.state.scale, h * self.state.scale
        return [self.state.tx, self.state.tx + sw, self.state.ty + sh, self.state.ty]

    def _connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self._press)
        self.fig.canvas.mpl_connect("button_release_event", self._release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._move)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._update_title()

    def _inside(self, event) -> bool:
        if event.xdata is None or event.ydata is None:
            return False
        x0, x1, y1, y0 = self._extent()
        return x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1

    def _press(self, event):
        if event.button != 1 or not self._inside(event):
            return
        self.dragging = True
        self.drag_start = (event.xdata, event.ydata)
        self.drag_origin = (self.state.tx, self.state.ty)

    def _move(self, event):
        if not self.dragging or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.drag_start[0]
        dy = event.ydata - self.drag_start[1]
        self.state.tx = self.drag_origin[0] + dx
        self.state.ty = self.drag_origin[1] + dy
        self._draw()

    def _release(self, event):
        if event.button == 1:
            self.dragging = False

    def _scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        factor = 1.01 if event.step > 0 else 1 / 1.01
        old_scale = self.state.scale
        new_scale = np.clip(old_scale * factor, 0.05, 20.0)
        if np.isclose(old_scale, new_scale):
            return
        h, w = self.tpl.shape[:2]
        rel_x = (event.xdata - self.state.tx) / (w * old_scale)
        rel_y = (event.ydata - self.state.ty) / (h * old_scale)
        self.state.tx = event.xdata - rel_x * w * new_scale
        self.state.ty = event.ydata - rel_y * h * new_scale
        self.state.scale = new_scale
        self._draw()

    def _key(self, event):
        if event.key == "q":
            self.saved = False
            plt.close(self.fig)
        elif event.key == "s":
            self.saved = True
            plt.close(self.fig)
        elif event.key == "r":
            self.state = Transform(alpha=self.state.alpha)
            self._draw()
        elif event.key in ("+", "="):
            self.state.alpha = np.clip(self.state.alpha + 0.05, 0.0, 1.0)
            self._draw()
        elif event.key in ("-", "_"):
            self.state.alpha = np.clip(self.state.alpha - 0.05, 0.0, 1.0)
            self._draw()
        elif event.key == "[":
            self._apply_scale(1 / 1.02)
        elif event.key == "]":
            self._apply_scale(1.02)

    def _apply_scale(self, factor: float):
        old_scale = self.state.scale
        new_scale = np.clip(old_scale * factor, 0.05, 20.0)
        if np.isclose(old_scale, new_scale):
            return
        cx = self.ref.shape[1] / 2.0
        cy = self.ref.shape[0] / 2.0
        h, w = self.tpl.shape[:2]
        rel_x = (cx - self.state.tx) / (w * old_scale)
        rel_y = (cy - self.state.ty) / (h * old_scale)
        self.state.tx = cx - rel_x * w * new_scale
        self.state.ty = cy - rel_y * h * new_scale
        self.state.scale = new_scale
        self._draw()

    def _draw(self):
        self.overlay.set_extent(self._extent())
        self.overlay.set_alpha(self.state.alpha)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        self.ax.set_title(
            "drag=move | scroll=scale | +/- alpha | r reset | s save | q quit\n"
            f"scale={self.state.scale:.3f}, tx={self.state.tx:.1f}, ty={self.state.ty:.1f}, alpha={self.state.alpha:.2f}"
        )

    def run(self) -> Transform | None:
        self.saved = False
        plt.show()
        return self.state if self.saved else None


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual template-to-background alignment tool.")
    parser.add_argument("--ref", required=True, type=Path, help="Reference/background image path")
    parser.add_argument("--template", required=True, type=Path, help="Template (foreground) image path")
    parser.add_argument("--output-json", required=True, type=Path, help="JSON output path")
    parser.add_argument("--alpha", type=float, default=0.6, help="Initial template alpha")
    parser.add_argument("--flip-ref-vertical", action="store_true", help="Flip reference image vertically before display")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    ref = load_rgb(args.ref)
    if args.flip_ref_vertical:
        ref = np.flipud(ref)
    template = load_rgb(args.template)
    matcher = ManualMatcher(ref, template, args.alpha)
    result = matcher.run()
    if result is None:
        print("No alignment saved.")
        return
    payload = {
        "reference": {"path": str(args.ref), "width": ref.shape[1], "height": ref.shape[0]},
        "template": {"path": str(args.template), "orig_width": template.shape[1], "orig_height": template.shape[0]},
        "transform": asdict(result),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Alignment saved to {args.output_json}")


if __name__ == "__main__":
    main()
