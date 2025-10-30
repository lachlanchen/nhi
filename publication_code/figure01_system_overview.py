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
    # Place label inside top-left with a light background to avoid dashed overlap
    ax.text(
        x + 0.1, y + h - 0.1, label,
        fontsize=8, color=ec, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=ec, linewidth=0.6, alpha=0.95),
    )


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

    fig_w, fig_h = 7.6, 3.6  # slightly wider and taller to avoid crowding
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # Coordinates (rough grid):
    # Illumination row (top): y = 4.5
    # Sample row: y = 3.2
    # Imaging row: y = 2.0
    # Camera row: y = 0.8

    # Illumination chain (left to right)
    src = add_block(ax, (0.6, 4.6), "Source", 1.5)
    slit = add_block(ax, (2.4, 4.6), "Slit", 1.2)
    grat = add_block(ax, (4.0, 4.6), "Grating", 1.3)
    fold = add_block(ax, (5.7, 4.6), "Fold", 1.0)

    add_arrow(ax, 0.6 + 1.5, 4.95, 2.4, 4.95)
    add_arrow(ax, 2.4 + 1.2, 4.95, 4.0, 4.95)
    add_arrow(ax, 4.0 + 1.3, 4.95, 5.7, 4.95)

    # Into sample (down)
    add_arrow(ax, 6.2, 4.6, 6.2, 3.7)

    # Sample and objective
    samp = add_block(ax, (5.3, 3.3), "Sample\n(trans)", 1.6)
    obj = add_block(ax, (7.3, 3.3), "Objective", 1.4)
    add_arrow(ax, 6.2 + 0.6, 3.65, 7.3, 3.65)

    # Tube lens and beamsplitter node
    tube = add_block(ax, (9.0, 3.3), "Tube lens", 1.4)
    add_arrow(ax, 7.3 + 1.4, 3.65, 9.0, 3.65)

    bs_cx, bs_cy = 10.5, 3.65
    add_arrow(ax, 9.0 + 1.4, 3.65, bs_cx - 0.15, 3.65)
    add_beamsplitter(ax, bs_cx, bs_cy, size=0.28)

    # Branch 1: existing microscope camera (straight ahead)
    add_arrow(ax, bs_cx + 0.15, 3.65, 12.6, 3.65)
    add_block(ax, (12.6, 3.3), "Microscope\ncam", 1.4)

    # Branch 2: to 4f relay + event camera (downwards)
    add_arrow(ax, bs_cx, bs_cy - 0.14, bs_cx, 2.4)
    relay = add_block(ax, (9.1, 2.0), "4f relay", 1.3)
    add_arrow(ax, bs_cx, 2.0, bs_cx, 1.2)
    evt = add_block(ax, (8.6, 0.7), "Event+\nFrame", 2.1)

    # Plugin boxes
    # Illumination plug-in around source->fold
    add_plugin_box(ax, 0.4, 4.25, 6.6, 1.35, label="Illumination plug-in")
    # Detection add-on around relay->event
    add_plugin_box(ax, 8.2, 0.5, 4.4, 2.3, label="Detection add-on (4f)", ec="#41ab5d")
    # Existing microscope frame around sample->tube lens (+ original camera)
    add_plugin_box(ax, 5.0, 2.95, 8.6, 2.1, label="Existing microscope", ec="#969696")

    # Keep diagram clean: avoid extra annotations that can overlap

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
