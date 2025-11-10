Quick Start

Prerequisites

- Python 3.9+ with numpy, torch, matplotlib installed (GPU optional).
- Your RAW recordings and/or segmented NPZ files accessible locally.

Paths used below are for the led_12v_no_acc_glass dataset you ran. Adjust for your data as needed.

1) Segment RAW Into 6 Scans (Forward/Backward)

Manual period (bins at 1 ms): default 1688 bins used in your run.

```
python segment_robust_fixed.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556.raw \
  --segment_events \
  --output_dir led_12v_no_acc_glass/glass/
```

Outputs an NPZ folder with six scans, e.g.

```
led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/
  Scan_1_Forward_events.npz
  Scan_2_Backward_events.npz
  ...
```

2) Train Multi‑Window Compensation on a Segment

Use the forward segment; tune args as desired (GPU used automatically if available):

```
python compensate_multiwindow_train_saved_params.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/Scan_1_Forward_events.npz \
  --bin_width 50000 \
  --visualize --plot_params --a_trainable \
  --iterations 1000 \
  --b_default 0 \
  --smoothness_weight 0.001
```

This saves learned parameter files (NPZ/JSON/CSV) next to the input segment NPZ.

3) Visualize Learned Boundaries and Time‑Binned Frames

This script auto-loads the most recent learned params in the same folder:

```
python visualize_boundaries_and_frames.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/Scan_1_Forward_events.npz
```

Optionally choose a fixed output folder instead of the timestamped default:

```
python scanning_alignment_visualization_save.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/Scan_1_Forward_events.npz \
  --output_dir led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/FIXED_visualization
```

4) Compare Cumulative 2 ms‑Step Means vs Multi‑Bin Means (2/50/100ms, all 2ms shift)

No frames are saved; this computes per‑pixel means only and plots:

```
python visualize_cumulative_compare.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/Scan_1_Forward_events.npz \
  --sensor_width 1280 --sensor_height 720
```

Optional output directory and label:

```
python scanning_alignment_visualization_cumulative_compare.py \
  led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/Scan_1_Forward_events.npz \
  --sensor_width 1280 --sensor_height 720 \
  --output_dir led_12v_no_acc_glass/glass/sync_recording_12v_led_no_acc_blank_event_20250804_232556_segments/cumulative_vs_bin2ms \
  --sample_label "led_12v_no_acc_glass Scan_1_Forward"
```

Tips

- Reuse trained params: supply `--load_params <params.npz>` to the trainer to skip optimization.
- GPU memory: the trainer uses chunked processing throughout; adjust `--chunk_size` if needed.
- For other datasets, replace the paths above with your *.raw and segment *.npz files.

Turbo multi‑scan workflow

Merge multiple Forward/Backward segments into one stream and run the same trainer:

```
python compensate_multiwindow_turbo.py \
  --segments-dir <path to *_segments> \
  --include all --sort name \
  --bin-width 5000 \
  -- --a_trainable --iterations 1000 --smoothness_weight 0.001 --chunk_size 250000 --visualize --plot_params
```

If the scan runs N× faster than baseline, scale the bin width by 1/N (e.g., 10× faster → `--bin-width 5000`). Train once, then use `--load-params <npz>` to change bin width quickly without retraining.
