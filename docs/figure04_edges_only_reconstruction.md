# Figure 4 — Edges‑Only Cropped Reconstruction (Run Notes)

This document records the exact changes, commands, and outputs for the
edges‑only alignment + cropped spectral reconstruction figure, so future
runs can be reproduced and the changes are traceable to commits.

## What Changed

- Wavelength ticks
  - The spectrum bar’s x‑ticks are centered under each kept column, with labels
    taken from the edges‑only wavelength mapping for that column. This avoids
    uneven spacing and places labels directly below the columns.
- Row spacing
  - Row gap is smaller than column gap to tighten vertical spacing:
    `wspace = 0.045`, `hspace = max(0.015, 0.35 * wspace)`.
- Cropping/rows and selection logic
  - Row 1: Original
  - Row 2: Comp.
  - Row 3: Gradient (selected by mapped wavelength; uses the bin whose nm range
    contains the mapped wavelength; otherwise nearest bin center)
  - Row 4: Reference (nearest single‑nm ROI to the mapped wavelength)
  - Sensor crop (`ref_crop`) applied to rows 1–2; external crop (`template_crop`)
    applied to rows 3–4.
  - Wavelength mapping is used for used/selected filenames:
    `bin_XX_YYYnm.png`.

Affected file: `publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py`

## Command Used

```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python \
  publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment /home/lachlan/ProjectsLFS/nhi_reconstruction/scan_angle_20_led_2835b/\
           angle_20_sanqin_2835_20250925_184638/\
           angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/groundtruth_spectrum_2835 \
  --diff-frames-dir /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/\
                    test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir  /home/lachlan/ProjectsLFS/nhi_reconstruction/hyperspectral_data_sanqin_gt/\
                    test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --start-bin 3 --end-bin 15 --downsample-rate 3 \
  --edge-quantile 0.05 \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700 \
  --flip-row12 \
  --crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --external-crop-json /home/lachlan/ProjectsLFS/nhi_reconstruction/alignment/crops/crop_metadata.json \
  --save-png
```

## Latest Output

- Folder: `publication_code/figures/spectral_reconstruction_scan_rotated_cropped_400_700_20251112_235544`
  - `spectral_reconstruction_scan_rotated_cropped_400_700.pdf|png`
  - `orig_used_frames/`, `comp_used_frames/`
  - `diff_used_frames/`, `ref_used_frames/`
  - `diff_selected_frames/`, `gt_selected_frames/`, `ref_selected_frames/`
  - `figure04_rescaled_bg_alignment.json`, `figure04_rescaled_bg_series.npz`
  - `figure04_rescaled_bg_gt_third_only.pdf|png`, `figure04_edges_only_third.pdf|png`
  - `figure04_rescaled_weights.json`

## Links to Commits (publication branch)

These commits implement the features documented above:

- 08baafa6 — feat(fig04-edges): add edges‑only script
- f8d593d0 — feat(fig04-edges): emit full all‑in‑one artifacts
- d548bbcc — fix(figure): choose correct crop boxes (sensor=ref_crop, external=template_crop)
- 87409080 — fix(figure): select Gradient/Reference per mapped wavelength (containment/nearest)
- e6886e3b — fix(figure): correct `render_spectral_grid` call
- 64d49367 — feat(figure): add wavelength tick labels under spectrum bar
- 728a167b — chore(figure): reduce row gap to half of column gap
- 155214f7 — fix(figure): center wavelength ticks under each displayed column
- bc814cec — chore(figure): equalize row and column gaps (later superseded by smaller row gap)
- baa13538 — chore(figure): rename row label “Compensated” → “Comp.”

Tip: Use `git show <sha>` to inspect the diff of an individual change.

## Quick Re‑Run Checklist

- Segment NPZ exists and has learned params next to it.
- Gradient folder contains `*_AtoBnm.png` bins.
- Reference folder contains `*_XXXnm.png` single‑nm ROI frames.
- Crops JSON present at `alignment/crops/crop_metadata.json` with keys
  `ref_crop` and `template_crop`.
- Use `--flip-row12` for row 1–2 if required by your dataset orientation.
- Use `--edge-quantile` to adjust edge sensitivity (default 0.05).

