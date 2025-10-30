#!/usr/bin/env bash
# Run the end-to-end reconstruction pipeline used for Figure 4 on a single RAW dataset.
# The script executes (in order):
#   1. Event segmentation with automatic scan-period detection.
#   2. Publication diagnostics (Figure 2 activity/correlation panels) for the new dataset.
#   3. Multi-window compensation on the first forward scan with chunked training.
#   4. Boundary/frame visualisations, cumulative comparisons, and weighted cumulative plots.
#
# Usage:
#   scripts/prepare_figure04.sh /path/to/dataset_dir [raw_file]
#
# Arguments:
#   dataset_dir : Directory containing the RAW/AVI pair for the acquisition.
#   raw_file    : (Optional) Explicit RAW path. When omitted, the script picks the first
#                 "*event*.raw" file in the dataset directory.
#
# Environment:
#   The commands rely on the "nhi_test" conda environment (CUDA-enabled PyTorch +
#   Prophesee metavision_hal bindings). Activate it before running this script:
#       conda activate nhi_test
#
# Tunable environment variables:
#   FIG4_ACTIVITY_FRACTION   – Event fraction for densest-window detection (default: 0.90)
#   FIG4_BIN_WIDTH           – Bin width for compensation training (default: 50000)
#   FIG4_SENSOR_WIDTH        – Sensor width in pixels (default: 1280)
#   FIG4_SENSOR_HEIGHT       – Sensor height in pixels (default: 720)
#   FIG4_SAMPLE_RATE         – Subsampling fraction for event visualisations (default: 0.10)
#   FIG4_TIME_BIN_US         – Bin size (µs) for activity trace in diagnostics (default: 1000)
#
# The script is idempotent: re-running will regenerate plots/NPZs with the same paths.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/dataset_dir [raw_file]" >&2
  exit 1
fi

DATASET_DIR="$(realpath "$1")"
RAW_PATH="${2:-}"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

if [[ -z "$RAW_PATH" ]]; then
  mapfile -t RAW_CANDIDATES < <(find "$DATASET_DIR" -maxdepth 1 -type f -name "*event*.raw" | sort)
  if [[ "${#RAW_CANDIDATES[@]}" -eq 0 ]]; then
    echo "Could not locate a RAW file (pattern '*event*.raw') inside $DATASET_DIR" >&2
    exit 1
  fi
  RAW_PATH="${RAW_CANDIDATES[0]}"
fi

RAW_PATH="$(realpath "$RAW_PATH")"
if [[ ! -f "$RAW_PATH" ]]; then
  echo "RAW file not found: $RAW_PATH" >&2
  exit 1
fi

RAW_BASENAME="$(basename "${RAW_PATH}" .raw)"
OUTPUT_DIR="$DATASET_DIR"
SEGMENTS_DIR="$OUTPUT_DIR/${RAW_BASENAME}_segments"
FORWARD_SEGMENT="$SEGMENTS_DIR/Scan_1_Forward_events.npz"

ACTIVITY_FRACTION="${FIG4_ACTIVITY_FRACTION:-0.90}"
BIN_WIDTH="${FIG4_BIN_WIDTH:-50000}"
SENSOR_WIDTH="${FIG4_SENSOR_WIDTH:-1280}"
SENSOR_HEIGHT="${FIG4_SENSOR_HEIGHT:-720}"
SAMPLE_RATE="${FIG4_SAMPLE_RATE:-0.10}"
TIME_BIN_US="${FIG4_TIME_BIN_US:-1000}"

FIGURE_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../publication_code/figures/figure04")"
mkdir -p "$FIGURE_DIR"

echo "=== Figure 4 pipeline ==="
echo "Dataset directory : $DATASET_DIR"
echo "RAW file          : $RAW_PATH"
echo "Segments output   : $SEGMENTS_DIR"
echo "Activity fraction : $ACTIVITY_FRACTION"
echo "Comp bin width    : $BIN_WIDTH"
echo "Sensor (WxH)      : ${SENSOR_WIDTH}x${SENSOR_HEIGHT}"
echo

# ---------------------------------------------------------------------------
# 1. Segment events into forward/backward scans with automatic period detection
# ---------------------------------------------------------------------------
echo "[1/6] Segmenting events..."
python segment_robust_fixed.py \
  "$RAW_PATH" \
  --segment_events \
  --auto_calculate_period \
  --output_dir "$OUTPUT_DIR" \
  --activity_fraction "$ACTIVITY_FRACTION"

if [[ ! -d "$SEGMENTS_DIR" ]]; then
  echo "Segmentation output missing: $SEGMENTS_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Produce diagnostics (activity/correlation + duration/event rate)
# ---------------------------------------------------------------------------
echo "[2/6] Generating segmentation diagnostics (Figure 2 panels for dataset)..."
python publication_code/figure02_scan_segmentation.py \
  --dataset-path "$SEGMENTS_DIR" \
  --raw-file "$RAW_PATH" \
  --time-bin-us "$TIME_BIN_US" \
  --activity-fraction "$ACTIVITY_FRACTION" \
  --output-dir "$FIGURE_DIR" \
  --save-png

# ---------------------------------------------------------------------------
# 3. Train multi-window compensation on the first forward scan segment
# ---------------------------------------------------------------------------
if [[ ! -f "$FORWARD_SEGMENT" ]]; then
  echo "Forward scan segment not found: $FORWARD_SEGMENT" >&2
  exit 1
fi

echo "[3/6] Training multi-window compensation on Scan_1_Forward..."
python compensate_multiwindow_train_saved_params.py \
  "$FORWARD_SEGMENT" \
  --bin_width "$BIN_WIDTH" \
  --visualize \
  --plot_params \
  --a_trainable \
  --iterations 1000 \
  --b_default 0 \
  --smoothness_weight 0.001

# ---------------------------------------------------------------------------
# 4. Boundary and frame visualisations with learned parameters
# ---------------------------------------------------------------------------
echo "[4/6] Rendering boundary/frame visualisations..."
python visualize_boundaries_and_frames.py \
  "$FORWARD_SEGMENT" \
  --sensor_width "$SENSOR_WIDTH" \
  --sensor_height "$SENSOR_HEIGHT" \
  --sample_rate "$SAMPLE_RATE"

# ---------------------------------------------------------------------------
# 5. Cumulative vs. multi-bin means comparison
# ---------------------------------------------------------------------------
echo "[5/6] Plotting cumulative vs. multi-bin means..."
python visualize_cumulative_compare.py \
  "$FORWARD_SEGMENT" \
  --sensor_width "$SENSOR_WIDTH" \
  --sensor_height "$SENSOR_HEIGHT"

# ---------------------------------------------------------------------------
# 6. Weighted cumulative plot (positive/negative balance)
# ---------------------------------------------------------------------------
echo "[6/6] Drawing weighted cumulative curves..."
python visualize_cumulative_weighted.py \
  "$FORWARD_SEGMENT" \
  --sensor_width "$SENSOR_WIDTH" \
  --sensor_height "$SENSOR_HEIGHT" \
  --step_us 10 \
  --auto_scale \
  --exp \
  --ymin 0 \
  --ymax 300

echo
echo "Figure 4 preparation complete."
echo "Key outputs:"
echo "  Segments directory : $SEGMENTS_DIR"
echo "  Diagnostics (PDF/PNG): $FIGURE_DIR"
echo "  Compensation artefacts : $(dirname "$FORWARD_SEGMENT")"
echo "  Additional visualisations : $(dirname "$FORWARD_SEGMENT")/FIXED_visualization_*"
