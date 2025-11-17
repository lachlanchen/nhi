# 3D Event Cloud Export (Tight, No Overflow)

This note documents how we render and save tightly cropped 3D event clouds (before/after multi‑window compensation) without overflow or excessive whitespace.

## Script and Inputs
- Script: `publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py`
- Input segment: `Scan_*_Forward_events.npz` (or Backward).
- Learned params NPZ are auto‑discovered next to the segment (pattern `*learned_params*.npz`).

## Environment
- Use the HAL/RAW environment `nhi_test`:
  - Python: `/home/lachlan/miniconda3/envs/nhi_test/bin/python`

## Basic Usage
```bash
~/miniconda3/envs/nhi_test/bin/python \
  publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  <path/to/Scan_*_events.npz> \
  --sample 0.02 \
  --time-scale 1.5 \
  --output-dir publication_code/figures
```

Outputs are written to a timestamped folder:
`publication_code/figures/multiwindow_compensation_figure_3d_cloud_with_image_<YYYYMMDD_HHMMSS>/`

Files: `event_cloud_before.{pdf,png}`, `event_cloud_after.{pdf,png}`.

## Key Options
- `--sample` (float): Fraction of events plotted (default 0.02). Increase for denser clouds; reduce for performance.
- `--time-scale` (float): Stretch factor along the time axis (Y) to improve visibility; default 1.5.
- `--chunk-size` (int): Batch size for compensation computation; default 400000.

## Tight Cropping and No Overflow
3D axes often produce excess whitespace or clipping with standard `bbox_inches="tight"`. We implemented `_save_tight_3d(...)` which:

1. Saves the figure with `bbox_inches='tight'`.
2. If the output is PDF, post‑processes it with `pdfcrop --margins 1` to remove remaining whitespace while preserving labels/ticks.

Notes:
- `pdfcrop` is part of TeX Live (we have `/home/lachlan/bin/pdfcrop`). If it is missing, the script falls back to the uncropped PDF.
- PNGs are saved with `bbox_inches='tight'` and no post‑crop. If you need further trimming, use ImageMagick’s `convert -trim` externally.

## View and Aspect
- Time is on the Y axis; spatial X/Z are X/Z axes.
- View defaults to `view_init(elev=25, azim=-35)` and a gentle perspective.
- The time axis is stretched (`box_aspect([1, 0.9*time_scale, 1])`) and a small projection scale is applied to improve readability.

You can change perspective by editing `view_init(...)` and `box_aspect(...)` in the script for custom angles.

## Example Commands
- Blank scan (Forward):
```bash
~/miniconda3/envs/nhi_test/bin/python \
  publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz \
  --sample 0.02 --time-scale 1.5 --output-dir publication_code/figures
```
- Sanqin scan (Forward):
```bash
~/miniconda3/envs/nhi_test/bin/python \
  publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --sample 0.02 --time-scale 1.5 --output-dir publication_code/figures
```

## Troubleshooting
- ValueError “Image size ... is too large”: this can occur when tight bboxes explode on some backends. The script now saves normally and then crops PDFs with `pdfcrop`, which avoids this. If it still occurs:
  - Lower `--sample`, or reduce DPI (e.g., 300), or slightly increase `--margins` in `_save_tight_3d`.
- Excess whitespace or slight clipping: adjust the `--margins` passed to `pdfcrop` (default is `1` pt). Increase to `2–3` to be safe.
- `pdfcrop` not found:
  - Install TeX Live’s `pdfcrop` (`texlive-extra-utils` on Debian/Ubuntu), or run `pdfcrop` manually later.
- Performance: plotting 3D clouds is rasterized for speed; `--sample` is the main knob.

## Using in Manuscript
Copy the final PDFs into `publication/self_calibrated_event_spectrum/figures/` and reference them from LaTeX. The pages are tightly cropped, so they integrate cleanly without overflow.

