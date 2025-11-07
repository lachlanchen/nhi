# Figure Data Sources and Repro Commands

This document records the primary data, derived artifacts, and commands used to generate each publication figure. Use the `nhi_test` environment python: `~/miniconda3/envs/nhi_test/bin/python`.

## Figure 1 — Overview
- Assets: `publication_code/figures/figure01_overview.{pdf,png}`
- Source: Schematic/illustration (no dataset dependency).

## Figure 2 — Self‑synchronized scan segmentation
- Raw events: `scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw`
- Segments dir: `scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments`
- Outputs: `publication_code/figures/figure02_{activity,correlation,duration,eventrate}.{pdf,png}`
- Command:
  ```bash
  ~/miniconda3/envs/nhi_test/bin/python publication_code/figure02_scan_segmentation.py \
    --dataset-path scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments \
    --raw-file     scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw \
    --time-bin-us 1000 \
    --activity-fraction 0.90 \
    --save-png
  ```

## Figure 3 — Multi‑window compensation diagnostics
- Segment NPZ: `scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz`
- Learned params: `.../Scan_1_Forward_events_chunked_processing_learned_params_n13.npz`
- Time‑binned artifacts (panel b/c):
  - NPZ: `.../time_binned_frames/Scan_1_Forward_events_chunked_processing_all_time_bins_data_multiwindow_chunked_processing_atrain_-0.5005_14.1464_btrain_-60.3067_-54.8752.npz`
  - CSV: `.../time_binned_frames/Scan_1_Forward_events_chunked_processing_time_bin_statistics_multiwindow_chunked_processing_atrain_-0.5005_14.1464_btrain_-60.3067_-54.8752.csv`
- Outputs: 
  - `publication_code/figures/multiwindow_compensation_20251106_121021/{multiwindow_events_blank,multiwindow_variance_blank}.{pdf,png}`
  - `publication_code/figures/multiwindow_compensation_20251106_121034/multiwindow_bin50ms_sanqin.{pdf,png}`
- Command:
  ```bash
  ~/miniconda3/envs/nhi_test/bin/python publication_code/multiwindow_compensation_figure.py \
    --variance_mode recompute --var_bin_us 50000 \
    --sensor_width 1280 --sensor_height 720 \
    --sample 0.05 \
    --output_dir publication_code/figures \
    --output_suffix blank \
    scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz
  ~/miniconda3/envs/nhi_test/bin/python publication_code/multiwindow_compensation_figure.py \
    --variance_mode recompute --var_bin_us 50000 \
    --sensor_width 1280 --sensor_height 720 \
    --sample 0.05 \
    --output_dir publication_code/figures \
    --output_suffix sanqin \
    scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz
  ```

## Figure 4 — Spectral reconstruction (grid + GT)
- Segment NPZ: `scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz`
- Ground truth (spectrometer): directory `groundtruth_spectrum_2835`
- GT ROI frames / gradients:
  - `hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched`
  - `hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched_gradient_20nm`
- Outputs (timestamped): `publication_code/figures/spectral_reconstruction_scan_<timestamp>/`
  - `spectral_reconstruction_scan.{pdf,png}` (Orig., Comp., gradient, reference rows + annotated wavelength bar)
  - `gt_selected_frames/`, `ref_selected_frames/`, `diff_selected_frames/`
- Command:
  ```bash
  ~/miniconda3/envs/nhi_test/bin/python publication_code/spectral_reconstruction_figure.py \
    --segment scan_angle_20_led_2835b/angle_20_sanqin_2835_20250925_184638/angle_20_sanqin_2835_event_20250925_184638_segments/Scan_1_Forward_events.npz \
    --gt-dir groundtruth_spectrum_2835 \
    --diff-frames-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched_gradient_20nm \
    --ref-frames-dir hyperspectral_data_sanqin_gt/test300_roi_square_frames_matched \
    --bin-width-us 50000 --start-bin 3 --end-bin 15 \
    --downsample-rate 3 --show-wavelength \
    --external-crop-json <hyperspectral ROI bbox json> \
    --figure-name spectral_reconstruction_scan \
    --save-png
  ```
  - Provide `--external-crop-json` whenever the hyperspectral PNGs still span the full field so rows 3–4 can be trimmed to the matched ROI.

## Figure 5 — RGB render from Figure 4 mapping (if used)
- Alignment JSON: `publication_code/figures/figure04_allinone_*/figure04_rescaled_bg_alignment.json`
- Segment NPZ: same as Figure 4
- Command (example):
  ```bash
  ~/miniconda3/envs/nhi_test/bin/python publication_code/figure05_rgb_from_bins.py \
    --segment <Scan_*_events.npz> \
    --alignment-json <path to figure04_rescaled_bg_alignment.json> \
    --sensor-width 1280 --sensor-height 720 \
    --bin-widths-us 5000 50000 --modes both --per-bin-mode gray --save-png
  ```

Notes
- Keep large outputs untracked. Timestamped Figure 4 folders contain all rendered panels and provenance (e.g., `gt_selected_frames/`).
- Use the exact `nhi_test` interpreter path above for RAW/Metavision workflows.
