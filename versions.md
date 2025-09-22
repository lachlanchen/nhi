Repository Versions and Layout

This project evolved through distinct stages. Code and data are grouped by era in versions/, while the current pipeline scripts remain at repo root.

Eras

1) 01_lensless_linear_motor
- Early experiments with a linear motor and lensless imaging.
- Contains EVK/EVK5 datasets and early frame/spectrum utilities and outputs.
- Path: versions/01_lensless_linear_motor/

2) 02_microscopy_reflective
- Microscopy (reflective mode) data.
- Path: versions/02_microscopy_reflective/

3) 03_microscopy_transmissive
- Microscopy (transmissive mode) datasets including the LED 12V series.
- Paths are preserved; LED datasets also have root-level symlinks to avoid breaking scripts:
  - led_12v -> versions/03_microscopy_transmissive/led_12v
  - led_12v_no_acc -> versions/03_microscopy_transmissive/led_12v_no_acc
  - led_12v_no_acc_glass -> versions/03_microscopy_transmissive/led_12v_no_acc_glass

4) 04_rotation_optimized
- Rotation-mode scanning and optimized multi-window compensation.
- Includes sync_imaging, scan_speed_test runs, and other later datasets.
- Path: versions/04_rotation_optimized/

Current (root) pipeline scripts

- Segmentation:
  - simple_autocorr_analysis_segment_robust_fixed.py

- Compensation:
  - scanning_alignment_with_merge_multi_gpt5_saved_params_comp.py
  - scanning_alignment_with_merge.py
  - scanning_alignment_with_merge_multi.py
  - scanning_alignment_with_merge_multi_with_batch.py
  - scanning_alignment_with_merge_multi_without_batch.py

- Visualization:
  - scanning_alignment_visualization.py
  - scanning_alignment_visualization_cumulative_compare.py
  - scanning_alignment_visualization_save.py (kept for convenience)

- IO helper:
  - simple_raw_reader.py

Archived legacy scripts

- archive_code_variants/ contains old variants and legacy scripts (e.g., *.py.old, *_old.py), preserved for reference without cluttering the root.

Rationale and method

- Grouping used folder mtimes and naming conventions (EVK/evk5 → early, reflective_* → reflective microscopy, led_* → transmissive microscopy, sync_imaging/scan_speed_test/sanqin/pass_* → latest rotation/optimized).
- The goal is to improve discoverability without breaking existing workflows. Root-level symlinks for key LED datasets keep historical commands working.
- No functional code changes were made.

Notes

- The Nature/ symlink is left untouched.
- If you want additional utilities or experiments moved into versions/ subfolders or into archive_*, please specify the files; moves are intentionally conservative.

