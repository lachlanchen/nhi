# Dataset–ground truth mappings

This folder hosts small utilities to align event‑camera reconstructions with
spectrometer ground‑truth curves. The table below records which segment NPZ is
paired with which spectrometer TXT files, so figures can be reproduced later.

## 1. 2835 LED (scan_angle_20_led_2835b)

- **Segment NPZ**  
  `scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz`
- **RAW + segmentation diagnostics**  
  - RAW: `scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747.raw`  
  - Segmentation analysis: `groundtruth_spectrum_2835/angle_20_blank_2835_event_20250925_184747_scanning_analysis.png`
- **Spectrometer ground truth (SPD)**  
  Directory: `groundtruth_spectrum_2835/`  
  Files:  
  - `USB2F042671_16-05-20-488.txt`  
  - `USB2F042671_16-05-22-288.txt`
- **Three‑panel overlay** (SPD vs events, log, derivative)  
  ```bash
  /home/lachlan/miniconda3/envs/nhi_test/bin/python \
    align_bg_vs_gt_code/compare_publication_three_panel.py \
    --segment scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz \
    --gt_files groundtruth_spectrum_2835/USB2F042671_16-05-20-488.txt \
    --step_ms 2.0 --bin_ms 5.0 \
    --sensor_width 1280 --sensor_height 720 \
    --suffix 2835 \
    --output_root align_bg_vs_gt_code
  ```
- **Current output (with suffix)**  
  - `align_bg_vs_gt_code/publication_20251121_182257/three_panel_5ms_Scan_1_Forward_2835.png`

## 2. Lumileds LED (scan_angle_20_lumileds)

- **Segment NPZ**  
  `scan_angle_20_lumileds/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz`
- **RAW + segmentation diagnostics**  
  - RAW: `scan_angle_20_lumileds/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw`  
  - Segmentation analysis: `scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_scanning_analysis.png`
- **Spectrometer ground truth (SPD)**  
  Directory: `groundtruth_spectrum_lumileds/`  
  Files:  
  - `USB2F042671_16-04-36-993.txt`  
  - `USB2F042671_16-04-56-391.txt`
- **Three‑panel overlay** (SPD vs events, log, derivative)  
  ```bash
  /home/lachlan/miniconda3/envs/nhi_test/bin/python \
    align_bg_vs_gt_code/compare_publication_three_panel.py \
    --segment scan_angle_20_lumileds/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz \
    --gt_files groundtruth_spectrum_lumileds/USB2F042671_16-04-56-391.txt \
    --step_ms 2.0 --bin_ms 5.0 \
    --sensor_width 1280 --sensor_height 720 \
    --suffix lumileds \
    --output_root align_bg_vs_gt_code
  ```
- **Current output (with suffix)**  
  - `align_bg_vs_gt_code/publication_20251121_182345/three_panel_5ms_Scan_1_Forward_lumileds.png`

