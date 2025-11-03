# Repository Guidelines

## Project Structure & Module Organization
- Core pipeline scripts (`segment_robust_fixed.py`, `compensate_multiwindow_train_saved_params.py`, `visualize_*`) live at repo root; update them together so segmentation, compensation, and visualization stay aligned.
- `versions/` captures experiment snapshots; fold validated ideas back into the root scripts and keep prototypes clearly labeled inside `experiments/`.
- Capture and hardware utilities sit under `sync_image_system/` (dual camera recorder) and `rotor/` (Arduino motor control). Keep board-specific changes isolated there.
- Sample datasets (`led_12v*`, `scan_angle_20/`) and generated outputs (`outputs_root/`) are checked in for reference—avoid committing new large binaries without coordinating storage.

## Build, Test, and Development Commands
- Create an isolated env: `python -m venv .venv && source .venv/bin/activate`, then `pip install numpy torch matplotlib` (pin any extra packages you add).
- Segment RAW input: `python segment_robust_fixed.py <recording.raw> --segment_events --output_dir data/segments`.
- Train compensation: `python compensate_multiwindow_train_saved_params.py <segment.npz> --bin_width 50000 --a_trainable --iterations 1000` (tune `--chunk_size` for memory).
- Inspect results: `python visualize_boundaries_and_frames.py <segment.npz>` and `python visualize_cumulative_compare.py <segment.npz> --sensor_width 1280 --sensor_height 720`.

### Environments
- HAL-enabled Conda env for RAW workflows: `nhi_test`
  - Python path: `/home/lachlan/miniconda3/envs/nhi_test/bin/python`
  - Use this env for any command that reads `.raw` (e.g., `segment_robust_fixed.py`) or regenerates Figure 2 from RAW.
  - Example:
    - `conda activate nhi_test`
    - `python segment_robust_fixed.py scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw --segment_events --output_dir scan_angle_20/angle_20_blank_20250922_170433 --auto_calculate_period`

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case modules/functions, CapWords classes, ≤120 char soft wrap, and grouped imports (stdlib → third-party → local).
- Prefer vectorized numpy/torch ops; add concise comments only where tensor math is non-obvious. Use docstrings for every CLI entry point.
- Name new CLI flags with lowercase-hyphen patterns and mirror them in help text; reuse existing parser utilities when possible.

## Testing Guidelines
- No formal unit suite exists; validate changes by replaying the segmentation → compensation → visualization pipeline on a small NPZ and reviewing plots in `outputs_root/`.
- Prototype checks live in `versions/04_rotation_optimized/experiments/`; extend those scripts (e.g., `scanning_alignment_with_merge_multi_test.py`) instead of scattering new test files.
- Log parameter choices and random seeds so others can reproduce your runs.

## Commit & Pull Request Guidelines
- Use Conventional Commit messages (`feat(weighted): …`, `chore: …`) as shown in `git log`; keep the subject imperative and ≤72 characters.
- Run `./scripts/setup_hooks.sh` once to enable the pre-commit hook that blocks >20 MB or disallowed file types.
- PRs should outline datasets/configurations exercised, link to relevant issues, and attach representative output figures or paths. Confirm the core commands above succeed before requesting review.
