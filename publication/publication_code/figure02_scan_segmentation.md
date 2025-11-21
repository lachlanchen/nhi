Figure 02 — Scan Segmentation Figures (Correlation + Activity)

Overview
- Script: `publication_code/figure02_scan_segmentation.py`
- Produces publication‑ready PDFs (and optional PNGs):
  - `figure02_activity.pdf` — Activity trace with pre/scan/post shading
  - `figure02_correlation.pdf` — Auto‑/auto‑convolution diagnostics
  - Supporting: `figure02_duration.pdf`, `figure02_eventrate.pdf`

Environment
- Use the HAL‑enabled env: `~/miniconda3/envs/nhi_test/bin/python`
- Inputs:
  - `--dataset-path` → a `*_segments` directory
  - `--raw-file` → matching RAW file for the dataset

Common Flags
- `--time-bin-us` (int): activity bin width (default 1000 µs)
- `--activity-fraction` (float): fraction of events for densest window (default 0.90)
- `--save-png`: also write PNGs alongside the PDFs
- `--font-scale` (float): scales all plot fonts; e.g. `0.9` → 90% of base
- `--output-suffix` (str): optional suffix appended to filenames (e.g. `_fs08`)
- `--output-dir` (path): write figures into this folder

Font Defaults (before scaling)
- Global/ticks: 11.7 pt
- Axis labels: 11.7 pt
- Titles: 11.7 pt
- Legends: 10.4 pt

Examples
1) Generate into a dedicated folder with 0.9 font scale
```
outdir=publication_code/figures/figure02_fs09_$(date +%Y%m%d_%H%M%S)
mkdir -p "$outdir"
~/miniconda3/envs/nhi_test/bin/python publication_code/figure02_scan_segmentation.py \
  --dataset-path scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments \
  --raw-file     scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw \
  --time-bin-us 1000 --activity-fraction 0.90 --save-png \
  --font-scale 0.9 --output-dir "$outdir"
```

2) Generate with suffix `_fs08` and 0.8 font scale (writes into default `publication_code/figures/`)
```
~/miniconda3/envs/nhi_test/bin/python publication_code/figure02_scan_segmentation.py \
  --dataset-path scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433_segments \
  --raw-file     scan_angle_20/angle_20_blank_20250922_170433/angle_20_blank_event_20250922_170433.raw \
  --time-bin-us 1000 --activity-fraction 0.90 --save-png \
  --font-scale 0.8 --output-suffix _fs08
```

Outputs
- PDFs (and PNGs when `--save-png`):
  - `figure02_activity[<suffix>].{pdf,png}`
  - `figure02_correlation[<suffix>].{pdf,png}`
  - `figure02_duration[<suffix>].{pdf,png}`
  - `figure02_eventrate[<suffix>].{pdf,png}`

Notes
- The script auto‑detects multiple `*_segments` under the dataset root to compute supporting plots (durations, event‑rate) while the two main figures use the selected dataset/RAW.
- Panel labels (a)/(b) are not embedded; place them via LaTeX/TikZ in the manuscript.

