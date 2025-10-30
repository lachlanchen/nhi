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
from matplotlib.patches import FancyBboxPatch, FancyArrow


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


def add_arrow(ax, x0, y0, x1, y1, lw=1.2, color="#000000", z=1.0):
    arr = FancyArrow(
        x0,
        y0,
        x1 - x0,
        y1 - y0,
        width=0.01,
        length_includes_head=True,
        head_width=0.08,
        head_length=0.12,
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
    )
    arr.set_zorder(z)
    ax.add_patch(arr)


def add_plugin_box(ax, x, y, w, h, label, ec="#2b8cbe", label_pos: str = "inside-tl"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.012,rounding_size=0.08",
                          linewidth=1.1, edgecolor=ec, facecolor="none",
                          linestyle=(0, (4, 3)))
    ax.add_patch(rect)

    # Compute label anchor by position keyword
    if label_pos == "inside-tl":
        lx, ly, ha, va = x + 0.1, y + h - 0.1, "left", "top"
    elif label_pos == "outside-tl":
        lx, ly, ha, va = x - 0.1, y + h + 0.12, "left", "bottom"
    elif label_pos == "outside-tr":
        lx, ly, ha, va = x + w + 0.1, y + h + 0.12, "right", "bottom"
    elif label_pos == "outside-bl":
        lx, ly, ha, va = x - 0.1, y - 0.12, "left", "top"
    elif label_pos == "outside-br":
        lx, ly, ha, va = x + w + 0.1, y - 0.12, "right", "top"
    elif label_pos == "outside-bc":
        lx, ly, ha, va = x + w / 2.0, y - 0.14, "center", "top"
    else:
        lx, ly, ha, va = x + 0.1, y + h - 0.1, "left", "top"

    ax.text(
        lx, ly, label,
        fontsize=8, color=ec, ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=ec, linewidth=0.6, alpha=0.95),
    )


def add_split_block(ax, xy, width, height, text_upper="4f", text_lower="relay",
                    fc="#f7f7f7", ec="#4d4d4d", lw=1.1, radius=0.08, fontsize=8):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    # Diagonal split to indicate combined beamsplitter + relay optics
    ax.plot([x, x + width], [y + height, y], color=ec, linewidth=lw * 0.9)
    pad_x = width * 0.08
    pad_y = height * 0.08
    ax.text(x + width - pad_x, y + height - pad_y, text_upper,
            ha="right", va="top", fontsize=fontsize)
    ax.text(x + pad_x, y + pad_y, text_lower,
            ha="left", va="bottom", fontsize=fontsize)
    return box


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

    # Into sample (down) – updated later to point to sample center once sample_x is known
    # Placeholder; will draw actual arrow after computing sample positions

    # Evenly spaced optical chain within the existing microscope:
    # choose equal edge-to-edge spacing L between Sample -> Objective -> Tube lens -> 4f relay block
    bs_cx = 10.0
    L = 0.40
    sample_w, obj_w, tube_w = 1.6, 1.4, 1.2  # tighten Tube lens width to remove slack
    sample_y = 3.3
    block_h = 0.7
    evt_w = 1.4
    evt_h = 0.7
    evt_y = 2.2
    evt_top = evt_y + evt_h

    # Combined reflector + 4f relay block, shared between the frame and event paths
    relay_block_w = block_h * 1.5  # wider footprint to balance the diagonal labels
    relay_block_h = block_h  # keep consistent height with microscope elements
    relay_block_y = sample_y
    relay_block_top = relay_block_y + relay_block_h
    relay_block_x = bs_cx - relay_block_w / 2.0
    relay_block_right = relay_block_x + relay_block_w

    # Solve positions from right to left so gaps are equal (edge-to-edge):
    tube_x = relay_block_x - L - tube_w
    objective_x = (tube_x - L) - obj_w
    sample_x = (objective_x - L) - sample_w

    # Blocks
    samp = add_block(ax, (sample_x, sample_y), "Sample\n(trans)", sample_w)
    obj = add_block(ax, (objective_x, sample_y), "Objective", obj_w)
    tube = add_block(ax, (tube_x, sample_y), "Tube\nlens", tube_w)
    relay_block = add_split_block(ax, (relay_block_x, relay_block_y), relay_block_w, relay_block_h)

    # Arrows between blocks (edge to edge), equalised length L
    chain_y = sample_y + block_h / 2.0
    add_arrow(ax, sample_x + sample_w, chain_y, objective_x, chain_y)
    add_arrow(ax, objective_x + obj_w, chain_y, tube_x, chain_y)
    # tube lens to beamsplitter left edge, drawn above elements
    add_arrow(ax, tube_x + tube_w, chain_y, relay_block_x, chain_y, lw=1.4, z=10)

    # Now draw the arrow from Fold to the center top of Sample box
    fold_cx = 5.7 + 0.5  # fold x + width/2
    sample_cx = sample_x + sample_w / 2.0
    add_arrow(ax, fold_cx, 4.6, sample_cx, sample_y + block_h)

    # Branch 1: top row to frame camera directly from the 4f relay block
    # Frame camera to the right
    cam_w = 1.7  # keep chain tight while staying inside figure bounds
    cam_x = relay_block_right + L
    cam_y = 3.3
    add_arrow(ax, relay_block_right, chain_y, cam_x, chain_y)
    add_block(ax, (cam_x, cam_y), "Frame\nCamera", cam_w)

    # Branch 2: down to the event sensor
    evt_x = bs_cx - evt_w / 2.0
    # From relay bottom edge to event top edge (shorter arrows that just touch boxes)
    add_arrow(ax, bs_cx, relay_block_y, bs_cx, evt_top, z=10)
    evt = add_block(ax, (evt_x, evt_y), "Event", evt_w)

    # Plugin boxes
    # Illumination plug-in around source->fold (label outside top-left)
    add_plugin_box(ax, 0.4, 4.35, 6.6, 1.2, label="Illumination plug-in", label_pos="outside-tl")
    # Detection add-on around the combined 4f relay and camera/event chain
    det_margin_x = 0.15
    det_margin_y = 0.2
    det_left = min(relay_block_x, evt_x) - det_margin_x
    det_right = cam_x + cam_w + det_margin_x
    det_bottom = evt_y - det_margin_y
    det_top = relay_block_top + det_margin_y
    add_plugin_box(ax, det_left, det_bottom, det_right - det_left, det_top - det_bottom,
                   label="Detection add-on (4f)", ec="#41ab5d", label_pos="outside-bc")
    # Existing microscope frame around sample->tube lens + frame camera (dynamic width), label outside top-right
    mic_margin = 0.07
    mic_left = sample_x - mic_margin
    mic_right = cam_x + cam_w + mic_margin
    add_plugin_box(ax, mic_left, 3.1, mic_right - mic_left, 1.05,
                   label="Existing microscope", ec="#969696", label_pos="outside-tr")

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
