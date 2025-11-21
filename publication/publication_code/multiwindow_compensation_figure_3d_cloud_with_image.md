3D Event Cloud (Before/After Compensation) — Run Notes

Script
- `publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py`

Purpose
- Renders 3D event clouds (X vs Time vs Y) before and after multi‑window compensation.
- Optionally overlays plain 2D images as a plane at a given time and draws dashed boxes at chosen times.

Key Flags
- Input:
  - `segment_npz` (positional): `Scan_*_Forward_events.npz`
- Sampling & device:
  - `--sample 0.02` (fraction of events)
  - `--chunk-size 400000`
- Axis & grid:
  - `--pin-grid --sensor-width 1280 --sensor-height 720` (pin ticks to sensor/time grid)
  - `--time-segments 3` (0, 1/3, 2/3, 1 ticks)
  - `--lock-time-axis --lock-spatial-axis` (consistent extents before/after)
  - `--axis-font-scale 0.9` (default: 0.9) — scales axis label fonts
- Overlay images:
  - `--overlay-image-before <png/pdf>` and `--overlay-image-after <png/pdf>`
  - `--overlay-time-ms 900` (time of overlay plane)
  - `--overlay-box-times 850,900` (draw dashed boxes at multiple times)
  - `--overlay-span axis|events` (map plane to full axis extents or event extents)
  - `--overlay-flipud` (flip overlay to match event Y)
  - `--overlay-stride 8` (downsample plane for light PDFs)
- Output:
  - `--output-dir publication_code/figures`

Examples
1) Clouds with dual dashed boxes at 850/900 ms and blueminmax overlays
```
~/miniconda3/envs/nhi_test/bin/python publication_code/multiwindow_compensation_figure_3d_cloud_with_image.py \
  scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/\
  angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --sample 0.02 --sensor-width 1280 --sensor-height 720 \
  --time-segments 3 --pin-grid --lock-time-axis --lock-spatial-axis \
  --overlay-image-before publication_code/figures/multiwindow_compensation_20251119_231734/multiwindow_bin50ms_original_plain_blueminmax_sanqin.png \
  --overlay-image-after  publication_code/figures/multiwindow_compensation_20251119_231734/multiwindow_bin50ms_compensated_plain_blueminmax_sanqin.png \
  --overlay-time-ms 900 --overlay-box-times 850,900 \
  --overlay-alpha 1.0 --overlay-span axis --overlay-flipud --overlay-stride 8 \
  --axis-font-scale 0.9 \
  --output-dir publication_code/figures
```

Outputs
- Timestamped folder: `publication_code/figures/multiwindow_compensation_figure_3d_cloud_with_image_<timestamp>/`
  - `event_cloud_before.{pdf,png}`
  - `event_cloud_after.{pdf,png}`
  - `overlay_image_before_plain.{pdf,png}` (if overlay specified)
  - `overlay_image_after_plain.{pdf,png}`  (if overlay specified)

Notes
- Tick labels remain at 8 pt; `--axis-font-scale` affects the axis label fonts only.
- Use `--overlay-span events` if you prefer mapping overlays to the actual event extents.
- Boxes adopt panel colors: red for BEFORE, green for AFTER.

