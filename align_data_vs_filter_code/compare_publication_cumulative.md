# `compare_publication_cumulative.py`

Publication‑style overlay of the compensated event‑camera cumulative spectrum against two spectrometer ground‑truth curves.

Each run:

- Rebuilds the compensated exponential cumulative series for one or more `Scan_*_Forward_events.npz` segments.
- Auto‑scales the negative polarity weight so the start/end plateaus match.
- Detects the active region on both reconstruction and GT curves and derives a linear mapping
  \(\lambda(t_{\text{ms}}) = \text{slope} \cdot t_{\text{ms}} + \text{intercept}\).
- Plots `GT 1`, `GT 2`, and `Recon` in wavelength space and writes a small mapping report.
- Saves results under a timestamped folder
  `align_bg_vs_gt_code/publication_YYYYMMDD_HHMMSS/`.

## Basic usage (2835 dataset)

From the repo root, in the `nhi_test` environment:

```bash
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  align_bg_vs_gt_code/compare_publication_cumulative.py \
  --segments scan_angle_20_led_2835b/angle_20_blank_2835_20250925_184747/angle_20_blank_2835_event_20250925_184747_segments/Scan_1_Forward_events.npz \
  --gt_dir groundtruth_spectrum_2835 \
  --step_ms 2.0 \
  --sensor_width 1280 --sensor_height 720 \
  --output_root align_bg_vs_gt_code
```

## Lumileds dataset example

```bash
/home/lachlan/miniconda3/envs/nhi_test/bin/python \
  align_bg_vs_gt_code/compare_publication_cumulative.py \
  --segments scan_angle_20_lumileds/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments/Scan_1_Forward_events.npz \
  --gt_dir groundtruth_spectrum_lumileds \
  --step_ms 2.0 \
  --sensor_width 1280 --sensor_height 720 \
  --output_root align_bg_vs_gt_code
```

## Outputs

For each segment, the script creates:

- `publication_cumulative_<segment_parent>_<seg_name>.png` – publication‑ready overlay figure.
- `mapping_<seg_name>.txt` – text summary including the chosen negative scale and
  the mapping formula \(\lambda(t_{\text{ms}})\).

Both live inside `align_bg_vs_gt_code/publication_YYYYMMDD_HHMMSS/` (the timestamp comes from the run time).

## Polarity weights used for cumsum

- Positive events use a fixed weight `pos_scale = 1.0`.
- Negative events use an auto‑tuned weight `neg_scale`, chosen so the start and end
  plateaus of the exponential cumulative are equal.
- For the Lumileds Scan_1_Forward run documented in
  `align_bg_vs_gt_code/archived/publication_20251121_171414/mapping_Scan_1_Forward.txt`,
  the script selected `neg_scale ≈ 1.260171`.
