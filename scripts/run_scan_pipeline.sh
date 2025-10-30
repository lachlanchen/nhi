#!/usr/bin/env bash
# Generic convenience wrapper that runs the standard reconstruction pipeline:
#   1. Event segmentation with segment_robust_fixed.py
#   2. Multi-window compensation on selected scan segments
#   3. Visualisation suite (boundaries/frames + cumulative plots)
#
# Usage:
#   scripts/run_scan_pipeline.sh /path/to/dataset_dir [raw_file]
#
# Arguments:
#   dataset_dir : Directory that holds the RAW / AVI pair for the acquisition.
#   raw_file    : (Optional) Explicit RAW path. If omitted, the first file
#                 matching "*event*.raw" in dataset_dir is used.
#
# By default the script compensates the first forward scan (Scan_1_Forward).
# Override the segment pattern by exporting PIPELINE_SEGMENT_PATTERN, e.g.:
#   export PIPELINE_SEGMENT_PATTERN='Scan_[13]_Forward_events.npz'
#   scripts/run_scan_pipeline.sh ...
#
# Tunable environment knobs (fall back to sensible defaults):
#   PIPELINE_ACTIVITY_FRACTION  (default 0.90)
#   PIPELINE_BIN_WIDTH          (default 50000)
#   PIPELINE_SENSOR_WIDTH       (default 1280)
#   PIPELINE_SENSOR_HEIGHT      (default 720)
#   PIPELINE_SAMPLE_RATE        (default 0.10)
#   PIPELINE_TIME_BIN_US        (default 1000)
#
# Assumes the "nhi_test" conda env (CUDA + metavision) is active.

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

ACTIVITY_FRACTION="${PIPELINE_ACTIVITY_FRACTION:-0.90}"
BIN_WIDTH="${PIPELINE_BIN_WIDTH:-50000}"
SENSOR_WIDTH="${PIPELINE_SENSOR_WIDTH:-1280}"
SENSOR_HEIGHT="${PIPELINE_SENSOR_HEIGHT:-720}"
SAMPLE_RATE="${PIPELINE_SAMPLE_RATE:-0.10}"
TIME_BIN_US="${PIPELINE_TIME_BIN_US:-1000}"
SEGMENT_PATTERN="${PIPELINE_SEGMENT_PATTERN:-Scan_1_Forward_events.npz}"

echo "=== Reconstruction pipeline ==="
echo "Dataset directory : $DATASET_DIR"
echo "RAW file          : $RAW_PATH"
echo "Segments output   : $SEGMENTS_DIR"
echo "Activity fraction : $ACTIVITY_FRACTION"
echo "Bin width         : $BIN_WIDTH"
echo "Segment pattern   : $SEGMENT_PATTERN"
echo

# ---------------------------------------------------------------------------
# 1. Segment events into forward/backward scans
# ---------------------------------------------------------------------------
echo "[1/4] Segmenting events..."
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

mapfile -t SEGMENTS < <(find "$SEGMENTS_DIR" -maxdepth 1 -type f -name "$SEGMENT_PATTERN" | sort)
if [[ "${#SEGMENTS[@]}" -eq 0 ]]; then
  echo "No segments matching pattern '$SEGMENT_PATTERN' in $SEGMENTS_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Loop over selected segments: compensate + visualise
# ---------------------------------------------------------------------------
for SEGMENT in "${SEGMENTS[@]}"; do
  echo
  echo "Processing segment: $(basename "$SEGMENT")"

  echo "[2/4] Training multi-window compensation..."
  python compensate_multiwindow_train_saved_params.py \
    "$SEGMENT" \
    --bin_width "$BIN_WIDTH" \
    --visualize \
    --plot_params \
    --a_trainable \
    --iterations 1000 \
    --b_default 0 \
    --smoothness_weight 0.001

  echo "[3/4] Rendering boundary/frame visualisations..."
  python visualize_boundaries_and_frames.py \
    "$SEGMENT" \
    --sensor_width "$SENSOR_WIDTH" \
    --sensor_height "$SENSOR_HEIGHT" \
    --sample_rate "$SAMPLE_RATE"

  echo "[4/4] Generating cumulative plots..."
  python visualize_cumulative_compare.py \
    "$SEGMENT" \
    --sensor_width "$SENSOR_WIDTH" \
    --sensor_height "$SENSOR_HEIGHT"

  python visualize_cumulative_weighted.py \
    "$SEGMENT" \
    --sensor_width "$SENSOR_WIDTH" \
    --sensor_height "$SENSOR_HEIGHT" \
    --step_us 10 \
    --auto_scale \
    --exp \
    --ymin 0 \
    --ymax 300
done

echo
echo "Pipeline complete."
echo "Outputs:"
echo "  Segments directory : $SEGMENTS_DIR"
echo "  Compensation / plots: $(dirname "${SEGMENTS[0]}")"
