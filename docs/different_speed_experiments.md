# Different-Speed Scan Experiments — Segmentation and Compensation Commands

This file records exact commands used to segment and compensate scans collected at different speeds for reproducibility.

Environment: HAL-enabled Conda env `nhi_test`.

General templates

- Segment with manual start shift (ms):

```
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  <PATH-TO>.raw \
  --segment_events \
  --output_dir <OUTPUT-DIR> \
  --auto_calculate_period \
  --manual_start_shift_ms <SHIFT_MS>
```

- Single-scan compensation (save under the segment folder):

```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  <SEGMENTS-DIR>/Scan_1_Forward_events.npz \
  --output_dir <SEGMENTS-DIR>/compensation_<BIN>ms \
  --bin_width <BIN_MS*1000> \
  --a_trainable --a_default 0.0 --b_default <B_DEFAULT> \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

- F+B merge with turbo (fold to 0..period, reverse backward time, flip polarity 0/1→1-p or −1/1→−p):

```
export MPLBACKEND=Agg
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_turbo.py \
  --segments <SEGMENTS-DIR>/Scan_1_Forward_events.npz <SEGMENTS-DIR>/Scan_2_Backward_events.npz \
  --include all --sort name --bin-width <BIN_MS*1000> \
  --output-dir <SEGMENTS-DIR>/compensation_<BIN>ms_fb \
  -- --a_trainable --a_default 0.0 --b_default <B_DEFAULT> \
     --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
     --chunk_size 250000 --visualize --plot_params
```

Recommended bin widths (guideline): speed_30 → 5–10 ms; speed_20 → 10 ms; speed_10 → 20 ms; speed_5 → 50 ms.

## Speed 30 (10× faster)

Segment (auto period detection):

```
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219.raw \
  --segment_events \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219 \
  --auto_calculate_period
```

Single forward compensation (bin width 5 ms; reuse params later for other bin widths):

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_segments/Scan_1_Forward_events.npz \
  --bin_width 5000 --a_trainable --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

F+B merge with turbo wrapper (folded time axis), train at 5 ms:

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_turbo.py \
  --segments \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_segments/Scan_1_Forward_events.npz \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_segments/Scan_2_Backward_events.npz \
  --include all --sort name --bin_width 5000 \
  -- --a_trainable --iterations 1000 --smoothness_weight 0.001 --chunk_size 250000 --visualize --plot_params
```

Manual start shift runs used for tuning:

```
# +35 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219 \
  --auto_calculate_period --manual_start_shift_ms 35

# +40 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219 \
  --auto_calculate_period --manual_start_shift_ms 40

# +50 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219 \
  --auto_calculate_period --manual_start_shift_ms 50

# +55 ms (chosen for compensation examples below)
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219 \
  --auto_calculate_period --manual_start_shift_ms 55
```

Compensation commands (saved under the segment folder):

```
# 5 ms
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/Scan_1_Forward_events.npz \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/compensation_5ms \
  --bin_width 5000 --a_trainable --a_default 0.0 --b_default -3.8 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params

# 10 ms
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/Scan_1_Forward_events.npz \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/compensation_10ms \
  --bin_width 10000 --a_trainable --a_default 0.0 --b_default -3.8 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params

# F+B at 10 ms via turbo
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_turbo.py \
  --segments \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/Scan_1_Forward_events.npz \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/Scan_2_Backward_events.npz \
  --include all --sort name --bin-width 10000 \
  --output-dir scan_angle_20_led_2835_different_speed/position_optimized/speed_30_angle_20_sanqin_opt_20251109_234219/\
  sync_recording_speed_30_angle_20_sanqin_opt_event_20251109_234219_shift_55ms_segments/compensation_10ms_fb \
  -- --a_trainable --a_default 0.0 --b_default -3.8 --learning_rate 0.1 --iterations 1000 \
     --smoothness_weight 0.001 --chunk_size 250000 --visualize --plot_params
```

## Speed 20

Segment (auto period detection):

```
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300.raw \
  --segment_events \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300 \
  --auto_calculate_period
```

Compensation (bin width 10 ms; 10× smaller init b; lr=0.1):

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300_segments/Scan_1_Forward_events.npz \
  --bin_width 10000 --a_trainable --a_default 0.0 --b_default -7.6 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

Manual start shift runs used for tuning:

```
# +60 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300 \
  --auto_calculate_period --manual_start_shift_ms 60

# +65 ms (chosen for compensation example below)
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300 \
  --auto_calculate_period --manual_start_shift_ms 65
```

Compensation saved under the shifted segments folder:

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300_shift_65ms_segments/Scan_1_Forward_events.npz \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_20_angle_20_sanqin_opt_20251109_234300/\
  sync_recording_speed_20_angle_20_sanqin_opt_event_20251109_234300_shift_65ms_segments/compensation_10ms \
  --bin_width 10000 --a_trainable --a_default 0.0 --b_default -7.6 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

## Speed 10

Segment (auto period detection):

```
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348.raw \
  --segment_events \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348 \
  --auto_calculate_period
```

Compensation (bin width 20 ms; moderate init b; lr=0.1):

Manual start shift runs used for tuning:

```
# +200 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348 \
  --auto_calculate_period --manual_start_shift_ms 200

# +150 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348 \
  --auto_calculate_period --manual_start_shift_ms 150

# +125 ms (used for compensation example below)
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348 \
  --auto_calculate_period --manual_start_shift_ms 125

# +115 ms
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348.raw \
  --segment_events --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348 \
  --auto_calculate_period --manual_start_shift_ms 115
```

Compensation saved under the shifted segments folder (20 ms):

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348_shift_125ms_segments/Scan_1_Forward_events.npz \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348_shift_125ms_segments/compensation_20ms \
  --bin_width 20000 --a_trainable --a_default 0.0 --b_default -38.0 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

## Speed 5

Segment with manual shift (best alignment +250 ms):

```
~/miniconda3/envs/nhi_test/bin/python segment_robust_fixed.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_5_angle_20_sanqin_opt_20251109_234440/\
  sync_recording_speed_5_angle_20_sanqin_opt_event_20251109_234440.raw \
  --segment_events \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_5_angle_20_sanqin_opt_20251109_234440 \
  --auto_calculate_period --manual_start_shift_ms 250
```

Compensation saved under the shifted segments folder (50 ms):

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_5_angle_20_sanqin_opt_20251109_234440/\
  sync_recording_speed_5_angle_20_sanqin_opt_event_20251109_234440_shift_250ms_segments/Scan_1_Forward_events.npz \
  --output_dir scan_angle_20_led_2835_different_speed/position_optimized/speed_5_angle_20_sanqin_opt_20251109_234440/\
  sync_recording_speed_5_angle_20_sanqin_opt_event_20251109_234440_shift_250ms_segments/compensation_50ms \
  --bin_width 50000 --a_trainable --a_default 0.0 --b_default -76.0 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

Additional tips

- Set initial values and learning rate explicitly:
  - `--a_default 0.0 --b_default <value> --learning_rate 0.1`
- Reuse saved parameters to skip training:
  - `--load_params <path/to/*_learned_params_n13.npz>`
- All figures are saved headlessly (Agg); use `--output_dir` to keep outputs under the segment folder.

```
~/miniconda3/envs/nhi_test/bin/python compensate_multiwindow_train_saved_params.py \
  scan_angle_20_led_2835_different_speed/position_optimized/speed_10_angle_20_sanqin_opt_20251109_234347_234348/\
  sync_recording_speed_10_angle_20_sanqin_opt_event_20251109_234348_segments/Scan_1_Forward_events.npz \
  --bin_width 20000 --a_trainable --a_default 0.0 --b_default -38.0 \
  --learning_rate 0.1 --iterations 1000 --smoothness_weight 0.001 \
  --chunk_size 250000 --visualize --plot_params
```

## Notes

- Use `--load_params <npz>` to reuse learned parameters and change bin width quickly without retraining.
- Turbo wrapper always folds F/B onto the same 0..period axis and flips backward polarity/time by default.
