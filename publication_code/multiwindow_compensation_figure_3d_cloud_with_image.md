# 3D Event Cloud Figure — Usage & Notes

This script renders before/after 3D event clouds and can overlay a 2D bin image as a semi-transparent plane at a chosen time. It saves tightly cropped PDFs/PNGs while keeping axis labels visible.

Script: `publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py`

## Environment
- Python: `/home/lachlan/miniconda3/envs/nhi_test/bin/python`
- Depends on: numpy, matplotlib, torch (CPU OK), pdfcrop (from TeX Live) for post-cropping PDFs.

## Key Behavior
- Axes: X (px), Y (time in ms), Z (px). Time is visually stretched but tick labels show true ms.
- View: `elev=25, azim=-35` so the cloud runs SW→NE.
- Z span is compressed to reduce vertical footprint.
- Points are rasterized for small PDFs; overlays are downsampled to keep vector size reasonable.
- PDF saving avoids `bbox_inches='tight'` (unreliable for 3D); instead saves full and `pdfcrop --margins 6` trims safely.

## Common Commands

Blank cloud with Sanqin overlays at 900 ms:

```
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --sample 0.02 --time-scale 1.5 \
  --overlay-image-before publication_code/figures/multiwindow_compensation_20251118_164334/multiwindow_bin50ms_original_plain_sanqin.png \
  --overlay-image-after  publication_code/figures/multiwindow_compensation_20251118_164334/multiwindow_bin50ms_compensated_plain_sanqin.png \
  --overlay-time-ms 900 --overlay-alpha 1.0 --overlay-span axis --overlay-flipud \
  --overlay-stride 8 --lock-time-axis --lock-spatial-axis \
  --output-dir publication_code/figures
```

Same, but place the image at 50 ms:

```
... --overlay-time-ms 50 ...
```

No overlay (just clouds):

```
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  <path>/Scan_1_Forward_events.npz --sample 0.02 --time-scale 1.5 \
  --output-dir publication_code/figures
```

## Important Flags
- `--sample FLOAT` — fraction of events to plot (default 0.02).
- `--time-scale FLOAT` — visual stretch of time axis (labels remain true ms).
- `--overlay-image-before/--overlay-image-after PATH` — PNG/JPG overlays (plain images recommended).
- `--overlay-time-ms FLOAT` — time position (ms) to place the overlay plane.
- `--overlay-alpha FLOAT` — overlay opacity (1.0 = opaque).
- `--overlay-span {axis,events}` — map overlay to current axis limits (keeps crop) or to event extents.
- `--overlay-flipud` — flip overlay vertically to match event Y convention.
- `--overlay-stride INT` — mesh downsampling for overlay; larger = lighter PDFs.
- `--lock-time-axis/--lock-spatial-axis` — use identical limits for before/after so planes align.

The dashed overlay box is enabled by default in this script invocation:
- Before: red; After: green.

## Avoiding Overflow/Clipping
- PDFs are saved un-tight and post-cropped via `pdfcrop --margins 6`. If labels still look close to the edge, increase to `--margins 8`.
- Subplot margins are set to `left=0.06, bottom=0.06`. Nudge slightly (e.g., 0.07) if needed.
- Tick density is limited via `MaxNLocator` to reduce crowding.

## Outputs
- Saved into `publication_code/figures/multiwindow_compensation_figure_3d_cloud_with_image_YYYYMMDD_HHMMSS/`:
  - `event_cloud_before.pdf/.png`
  - `event_cloud_after.pdf/.png`

For manuscript, copy the PDFs to:
- `publication/self_calibrated_event_spectrum/figures/event_cloud_before.pdf`
- `publication/self_calibrated_event_spectrum/figures/event_cloud_after.pdf`

## Notes & Tips
- Y ticks display real milliseconds even when the time axis is visually stretched.
- If overlays appear to occlude the cloud, reduce `--overlay-alpha` (e.g., 0.6–0.8) or draw overlays at a time outside the densest region.
- If files get large, increase `--overlay-stride` (e.g., 10–14) and keep the point `--sample` small.
- To change the perceived vertical height, adjust the internal `set_box_aspect` or the Z compression (see code comments).

