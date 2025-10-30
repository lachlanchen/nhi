#!/usr/bin/env python3
"""
Figure 1: Transmission optical path and plug-in system overview

Generates a clean, publication-ready system diagram as a vector PDF that
emphasizes two modular components:
  - Illumination plug-in (drop-in before the sample)
  - Detection add-on (4f relay to event camera, with original camera retained)

Output: publication_code/figures/figure01_overview.pdf

Notes
 - Keep styling minimal: light fills, clear arrows, small fonts.
 - Use dashed rounded rectangles to indicate plug-in modules.
 - Show compatibility with existing microscope (original camera path preserved).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Rectangle


def add_block(ax, xy, text, width=1.6, height=0.7, fc="#f7f7f7", ec="#4d4d4d",
              lw=1.1, radius=0.08, fontsize=8, ha="center"):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + width/2.0 if ha == "center" else x + 0.06, y + height/2.0,
            text, ha=ha, va="center", fontsize=fontsize)
    return box


def add_arrow(ax, x0, y0, x1, y1, lw=1.2, color="#000000"):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0,
                            width=0.01, length_includes_head=True,
                            head_width=0.08, head_length=0.12,
                            linewidth=lw, edgecolor=color, facecolor=color))


def add_plugin_box(ax, x, y, w, h, label, ec="#2b8cbe"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.012,rounding_size=0.08",
                          linewidth=1.1, edgecolor=ec, facecolor="none",
                          linestyle=(0, (4, 3)))
    ax.add_patch(rect)
    ax.text(x + 0.06, y + h + 0.06, label, fontsize=8, color=ec, ha="left", va="bottom")


def add_beamsplitter(ax, cx, cy, size=0.25, ec="#4d4d4d", fc="#e0e0e0"):
    # Draw a small square rotated by 45° to indicate a beamsplitter cube
    s = size
    # As a simple square (no rotation) plus diagonal line
    r = Rectangle((cx - s/2, cy - s/2), s, s, linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(r)
    ax.plot([cx - s/2, cx + s/2], [cy - s/2, cy + s/2], color=ec, linewidth=1.0)
    return r


def render(out_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.linewidth": 0.8,
    })

    fig_w, fig_h = 7.0, 3.6  # inches (wider to avoid crowding)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Coordinates (rough grid):
    # Illumination row (top): y = 4.5
    # Sample row: y = 3.2
    # Imaging row: y = 2.0
    # Camera row: y = 0.8

    # Illumination chain (left to right)
    src = add_block(ax, (0.4, 4.15), "Broadband\nsource", 1.6)
    slit = add_block(ax, (2.3, 4.15), "Slit", 1.2)
    grat = add_block(ax, (3.9, 4.15), "Grating", 1.4)
    fold = add_block(ax, (5.6, 4.15), "Fold\nmirror", 1.4)

    add_arrow(ax, 0.4 + 1.6, 4.5, 2.3, 4.5)
    add_arrow(ax, 2.3 + 1.2, 4.5, 3.9, 4.5)
    add_arrow(ax, 3.9 + 1.4, 4.5, 5.6, 4.5)

    # Into sample (down)
    add_arrow(ax, 6.3, 4.15, 6.3, 3.4)

    # Sample and objective
    samp = add_block(ax, (5.6, 3.0), "Sample\n(transmission)", 1.4)
    obj = add_block(ax, (7.4, 3.0), "Objective", 1.4)
    add_arrow(ax, 6.3 + 0.7, 3.35, 7.4, 3.35)

    # Tube lens and beamsplitter node
    tube = add_block(ax, (9.0, 3.0), "Tube lens", 1.4)
    add_arrow(ax, 7.4 + 1.4, 3.35, 9.0, 3.35)

    bs_cx, bs_cy = 10.7, 3.35
    add_arrow(ax, 9.0 + 1.4, 3.35, bs_cx - 0.15, 3.35)
    add_beamsplitter(ax, bs_cx, bs_cy, size=0.28)

    # Branch 1: existing microscope camera (straight ahead)
    add_arrow(ax, bs_cx + 0.15, 3.35, 11.8, 3.35)
    add_block(ax, (11.8, 3.0), "Original\nmicroscope camera", 1.4)

    # Branch 2: to 4f relay + event camera (downwards)
    add_arrow(ax, bs_cx, bs_cy - 0.14, bs_cx, 2.2)
    relay = add_block(ax, (10.0, 1.8), "4f relay", 1.4)
    add_arrow(ax, 10.7, 1.8, 10.7, 1.2)
    evt = add_block(ax, (9.5, 0.6), "Event +\nframe sensor", 2.2)

    # Plugin boxes
    # Illumination plug-in around source->fold
    add_plugin_box(ax, 0.25, 4.0, 6.95, 1.2, label="Illumination plug-in (drop-in before sample)")
    # Detection add-on around relay->event
    add_plugin_box(ax, 9.2, 0.45, 3.3, 2.0, label="Detection add-on (4f relay; keeps original camera)", ec="#41ab5d")
    # Existing microscope frame around sample->tube lens (+ original camera)
    add_plugin_box(ax, 5.4, 2.65, 8.0, 1.8, label="Existing microscope (sample, objective, tube lens)", ec="#969696")

    # Small annotations
    ax.text(6.3, 4.9, "Dispersed illumination", fontsize=8, ha="center")
    ax.text(10.7, 2.6, "Optional splitter/port", fontsize=8, ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1: system overview diagram")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "figures" / "figure01_overview.pdf",
                        help="Output PDF path")
    args = parser.parse_args()
    render(args.output.resolve())


if __name__ == "__main__":
    main()

