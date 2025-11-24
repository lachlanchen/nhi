Run Notes — spectral_reconstruction_figure_cropped_rescaled_edges_only.py

Purpose
- Renders the 4-row spectral grid (Raw, Comp., Diff., Reference) with wavelength bar.
- Uses “edges-only” time→wavelength alignment and applies consistent crops to all rows.
- Draws a dashed rectangle around rows 2–3 (Comp. and Diff.).

Environment
- Python: ~/miniconda3/envs/nhi_test/bin/python
- Segment: scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz
- Crops JSON (both sensor + external): alignment/crops/crop_metadata.json
  - Keys: ref_crop (sensor rows 1–2), template_crop (external rows 3–4)
- External rows (rotated, ROI-cropped):
  - Diff:  hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm
  - Ref:   hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops
- Rows 3–4 (Diff/Ref) are vertically flipped in the script to match the desired orientation.

Command (final, working)
```
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/\
           angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir groundtruth_spectrum_2835 \
  --diff-frames-dir hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir  hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --start-bin 3 --end-bin 15 \
  --wl-list 394,446,497,549,600,651,703 \
  --edge-quantile 0.05 \
  --flip-row12 \
  --show-wavelength --single-colorbar \
  --col-gap 0.045 --row-gap 0.006 \
  --crop-json alignment/crops/crop_metadata.json \
  --external-crop-json alignment/crops/crop_metadata.json \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700_wllist_signedraw_q95_sharednorm_fullbar_ticks_labels_v3 \
  --save-png
```

Outputs
- Timestamped folder: publication_code/figures/spectral_reconstruction_scan_rotated_cropped_400_700_wllist_signedraw_q95_sharednorm_fullbar_ticks_labels_v3_<timestamp>/
- Main PDF/PNG: spectral_reconstruction_scan_rotated_cropped_400_700_wllist_signedraw_q95_sharednorm_fullbar_ticks_labels_v3.{pdf,png}
- Copy the PDF to: publication/self_calibrated_event_spectrum/figures/

Dash box alignment tips (no code change)
- Ensure both crops are passed: --crop-json and --external-crop-json (same file).
- Use the rotated ROI directories listed above for Diff/Ref rows.
- Keep gaps: --col-gap 0.045 --row-gap 0.006. Small row-gap helps the box align with rows 2–3.
- If alignment still looks off, re-run with the same flags; do not modify the script.

Variant — Single Coolwarm Bar (Rows 1–3), Row 4 Gray
- Colorbars:
  - One shared coolwarm colorbar spanning rows 1–3 only (Raw, Comp., Diff.).
  - Row 4 (Reference) is grayscale with no extra bar.
- Command:
```
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  publication_code/spectral_reconstruction_figure_cropped_rescaled_edges_only.py \
  --segment scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/\
           angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
  --gt-dir groundtruth_spectrum_2835 \
  --diff-frames-dir hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops_gradient_20nm \
  --ref-frames-dir  hyperspectral_data_sanqin_gt/test300_rotated_frames_137d37_roi_crops \
  --bin-width-us 50000 --fine-step-us 5000 \
  --sensor-width 1280 --sensor-height 720 \
  --start-bin 3 --end-bin 15 \
  --wl-list 394,446,497,549,600,651,703 \
  --edge-quantile 0.05 \
  --flip-row12 \
  --show-wavelength \
  --row123-shared-cbar \
  --single-colorbar \
  --col-gap 0.045 --row-gap 0.006 \
  --crop-json alignment/crops/crop_metadata.json \
  --external-crop-json alignment/crops/crop_metadata.json \
  --figure-name spectral_reconstruction_scan_rotated_cropped_400_700_wllist_signedraw_q95_sharednorm_fullbar_ticks_labels_v3 \
  --save-png
```
