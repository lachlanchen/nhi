# Different-Speed Scan Experiments — Segmentation and Compensation Commands

This file records exact commands used to segment and compensate scans collected at different speeds for reproducibility.

Environment: HAL-enabled Conda env `nhi_test`.

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
