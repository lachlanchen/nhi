Spectral Reconstruction (Cropped) — Edges‑Only Mapping

Overview
- Script: `publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py`
- Purpose: Produce the 4‑row grid (Original, Comp., Gradient, Reference) using the edges‑only time→wavelength mapping, plus the full Figure‑04 background artifact set.
- Key features:
  - Column selection by wavelength (fixed range or explicit list)
  - Spectrum bar locked to desired nm span
  - Tight control of row/column gaps
  - Correct frame selection for each row

Quick Presets
- No-flip, equal aspect for all rows (consistent pixel aspect):
  - Skip both `--flip-row12` and `--flip-row34`.
  - Pass `--image-aspect12 equal --image-aspect34 equal`.
  - Typical spacing: `--col-gap 0.045 --row-gap 0.045`.
- No-flip, tighter external rows (pack gradient/reference):
  - `--image-aspect12 equal --image-aspect34 auto`.
  - Use smaller `--row-gap` (e.g., 0.006) if needed.
- Zero-gap stacked layout (shared colorbars, min margins):
  - `--row-gap 0.0`, `--column-step 2`, `--row12-shared-cbar`, `--row34-colorbar`, `--cbar-ratio 0.15`
  - Figure saving uses tight bounding box; final gap controlled entirely by `--row-gap`.

Essential Flags
- Input
  - `--segment` Scan NPZ (e.g., `.../Scan_1_Forward_events.npz`)
  - `--gt-dir` ground truth SPD dir (e.g., `groundtruth_spectrum_2835`)
  - `--diff-frames-dir` gradient frames (e.g., `..._gradient_20nm`)
  - `--ref-frames-dir` reference ROI frames (single‑nm)
- Alignment & binning
  - `--bin-width-us` (default 50000), `--fine-step-us` (default from Figure 04)
  - `--edge-quantile` (default 0.05)
- Column selection by wavelength
  - Range: `--wl-min 400 --wl-max 700 --wl-step 20`
  - List: `--wl-list 400,450,500,550,600,650,700`
- Spectrum bar bounds
  - `--bar-wl-min 400 --bar-wl-max 700`
- Layout/spacings
  - `--col-gap` (default 0.045)
  - `--row-gap` (default 0.35 × col-gap). Use smaller values to compress rows; set near 0 for minimal spacing.
  - `--image-aspect12 {equal,auto}`: aspect for rows 1–2 (Original/Comp.). Use `equal` to preserve sensor pixel aspect.
  - `--image-aspect34 {equal,auto}`: aspect for rows 3–4 (Gradient/Reference). Use `auto` to pack vertically if desired.
  - `--unified-row12-scales`: enforce a single global scale per row (Original counts + Comp. Δ) so comparisons across columns stay consistent.
  - `--raw-global-vmin/--raw-global-vmax`, `--comp-global-abs`: manually override the sensor row ranges.
  - `--cbar-ratio` (default 0.15): width of each colorbar column, as a fraction of a data column.
  - `--row34-colorbar`: add intensity colorbars for Gradient/Reference rows (per-row bars stacked in the rightmost column).
- Crops/flip
  - `--crop-json` (sensor rows, used for Original/Comp.)
  - `--external-crop-json` (external rows, used for Gradient/Reference)
  - `--flip-row12` (flip Original/Comp.) and/or `--flip-row34`
- Output
  - `--figure-name` base name (timestamped folder under `publication_code/figures/`)
  - `--save-png` alongside PDF

Correct Row/Frame Sources
- Row 1: Original (sensor crop)
- Row 2: Comp. (sensor crop)
- Row 3: Gradient (range contains mapped nm; else nearest; external crop)
- Row 4: Reference (nearest single‑nm ROI; external crop)
- Spectrum bar: ticks centered under each kept column, labels = mapped wavelength for that column.

Example Commands
1) Typical run (400→700 nm, 20-nm step), tighter rows
```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python \
  publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment /home/lachlan/ProjectsLFS/nhi_reconstruction/scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/groundtruth_spectrum_2835 \
  --diff-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --edge-quantile 0.05 \
  --wl-min 400 --wl-max 700 --wl-step 20 \
  --bar-wl-min 400 --bar-wl-max 700 \
  --col-gap 0.045 --row-gap 0.006 --image-aspect12 equal --image-aspect34 auto \
  --flip-row12 \
  --crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --external-crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700 \
  --save-png
```

2) Explicit nm list (custom columns), row gap even smaller
```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment <.../Scan_1_Forward_events.npz> \
  --gt-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/groundtruth_spectrum_2835 \
  --diff-frames-dir <gradient_20nm_dir> \
  --ref-frames-dir <roi_dir> \
  --wl-list 400,450,500,550,600,650,700 \
  --bar-wl-min 400 --bar-wl-max 700 \
  --col-gap 0.045 --row-gap 0.006 --image-aspect12 equal --image-aspect34 auto \
  --flip-row12 --save-png
```

3) No-flip, equal aspect for all rows (crop applied to both sensor/external rows)
```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment /home/lachlan/ProjectsLFS/nhi_reconstruction/scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/groundtruth_spectrum_2835 \
  --diff-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --edge-quantile 0.05 \
  --wl-min 400 --wl-max 700 --wl-step 20 \
  --bar-wl-min 400 --bar-wl-max 700 \
  --col-gap 0.045 --row-gap 0.045 \
  --image-aspect12 equal --image-aspect34 equal \
  --crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --external-crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --unified-row12-scales --row34-colorbar --cbar-ratio 0.2 \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700 \
  --save-png
4) Zero-gap shared-colorbar layout (7 columns, minimal margins)
```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment /home/lachlan/ProjectsLFS/nhi_reconstruction/scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/groundtruth_spectrum_2835 \
  --diff-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --edge-quantile 0.05 \
  --wl-min 400 --wl-max 700 --wl-step 20 \
  --bar-wl-min 400 --bar-wl-max 700 \
  --col-gap 0.045 --row-gap 0.0 \
  --image-aspect12 equal --image-aspect34 equal \
  --crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --external-crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --unified-row12-scales --row12-shared-cbar --row34-colorbar --cbar-ratio 0.15 --column-step 2 \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700 \
  --save-png
```
```

Outputs
- Timestamped folder under `publication_code/figures/<figure-name>_<timestamp>/` containing:
  - `spectral_reconstruction_scan_rotated_cropped_400_700.pdf|png`
  - Used/selected frames per row: `orig_used_frames/`, `comp_used_frames/`, `diff_used_frames/`, `ref_used_frames/`, plus `diff_selected_frames/`, `gt_selected_frames/`, `ref_selected_frames/`
  - Figure‑04 background artifacts: `figure04_rescaled_bg_alignment.json`, `figure04_rescaled_bg_series.npz`, `figure04_rescaled_bg_gt_third_only.pdf|png`, `figure04_edges_only_third.pdf|png`, `figure04_rescaled_weights.json`

Notes
- If you prefer fixed anchor ticks (e.g., 400/450/…/700) independent of columns, we can add an option; currently ticks are per‑column labels and centered under each column.
- For tighter layouts, start with `--row-gap 0.008` or `0.006` and adjust `--col-gap` accordingly.
