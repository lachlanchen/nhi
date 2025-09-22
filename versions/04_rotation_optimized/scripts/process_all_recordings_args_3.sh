#!/bin/bash

# Enhanced script to process all synchronized recordings with segmentation and scan compensation
# Usage: ./process_all_recordings.sh [OPTIONS]
#
# OPTIONS:
#   -h, --help                      Show this help message
#   -s, --skip-segmentation         Skip the segmentation step
#   -d, --base-dir DIR              Set base directory (default: ./plastics)
#
# SEGMENTATION OPTIONS:
#   --time-bin-us VALUE             Set time bin size in microseconds for segmentation (default: 1000)
#   --max-events VALUE              Maximum events to load for segmentation
#
# SCAN COMPENSATION OPTIONS:
#   -b, --bin-width VALUE           Set bin width in microseconds (default: 50000)
#   -x, --initial-a-x VALUE         Set initial A_x parameter (default: 0.541157)
#   -y, --initial-a-y VALUE         Set initial A_y parameter (default: -75.052391)
#   --sensor-width VALUE            Sensor width in pixels (default: 1280)
#   --sensor-height VALUE           Sensor height in pixels (default: 720)
#   --iterations VALUE              Number of training iterations (default: 1000)
#   --learning-rate VALUE           Learning rate for optimization (default: 1.0)
#   --direct-params                 Use provided a_x and a_y directly without optimization
#   --no-visualize                  Disable visualization in scan compensation
#   --save-frames                   Save basic event frames as images and arrays
#   --save-enhanced-frames          Save enhanced frames with smoothing and shifting
#   --smooth                        Apply 3D smoothing to enhanced frames
#   --use-mean                      Use mean instead of median for frame shifting
#   --max-comparison-frames VALUE   Maximum number of frames to show in comparison plots
#
# Examples:
#   ./process_all_recordings.sh
#   ./process_all_recordings.sh --skip-segmentation
#   ./process_all_recordings.sh -b 75000 -x 0.6 -y -80.0
#   ./process_all_recordings.sh --skip-segmentation --bin-width 100000 --no-visualize
#   ./process_all_recordings.sh --skip-segmentation --bin-width 2500 --no-visualize --use-mean
#   ./process_all_recordings.sh --save-enhanced-frames --smooth --use-mean

# Default values
SKIP_SEGMENTATION=false
INITIAL_A_X="0.541157"
INITIAL_A_Y="-75.052391"
BIN_WIDTH="50000"
ENABLE_VISUALIZE=true
SCRIPT_DIR="$(pwd)"
BASE_DIR_RELATIVE="plastics"

# Segmentation options
TIME_BIN_US="1000"
MAX_EVENTS=""

# Scan compensation options
SENSOR_WIDTH="1280"
SENSOR_HEIGHT="720"
ITERATIONS="1000"
LEARNING_RATE="1.0"
DIRECT_PARAMS=false
SAVE_FRAMES=false
SAVE_ENHANCED_FRAMES=false
SMOOTH=false
USE_MEAN=false
MAX_COMPARISON_FRAMES=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Enhanced script to process all synchronized recordings with segmentation and scan compensation"
    echo ""
    echo "GENERAL OPTIONS:"
    echo "  -h, --help                      Show this help message"
    echo "  -s, --skip-segmentation         Skip the segmentation step"
    echo "  -d, --base-dir DIR              Set base directory (default: $BASE_DIR_RELATIVE)"
    echo ""
    echo "SEGMENTATION OPTIONS:"
    echo "  --time-bin-us VALUE             Set time bin size in microseconds (default: $TIME_BIN_US)"
    echo "  --max-events VALUE              Maximum events to load for segmentation"
    echo ""
    echo "SCAN COMPENSATION OPTIONS:"
    echo "  -b, --bin-width VALUE           Set bin width in microseconds (default: $BIN_WIDTH)"
    echo "  -x, --initial-a-x VALUE         Set initial A_x parameter (default: $INITIAL_A_X)"
    echo "  -y, --initial-a-y VALUE         Set initial A_y parameter (default: $INITIAL_A_Y)"
    echo "  --sensor-width VALUE            Sensor width in pixels (default: $SENSOR_WIDTH)"
    echo "  --sensor-height VALUE           Sensor height in pixels (default: $SENSOR_HEIGHT)"
    echo "  --iterations VALUE              Number of training iterations (default: $ITERATIONS)"
    echo "  --learning-rate VALUE           Learning rate for optimization (default: $LEARNING_RATE)"
    echo "  --direct-params                 Use provided a_x and a_y directly without optimization"
    echo "  --no-visualize                  Disable visualization in scan compensation"
    echo "  --save-frames                   Save basic event frames as images and arrays"
    echo "  --save-enhanced-frames          Save enhanced frames with smoothing and shifting"
    echo "  --smooth                        Apply 3D smoothing to enhanced frames"
    echo "  --use-mean                      Use mean instead of median for frame shifting"
    echo "  --max-comparison-frames VALUE   Maximum number of frames to show in comparison plots"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --skip-segmentation"
    echo "  $0 -b 75000 -x 0.6 -y -80.0"
    echo "  $0 --skip-segmentation --bin-width 100000 --no-visualize"
    echo "  $0 --skip-segmentation --bin-width 2500 --no-visualize --use-mean"
    echo "  $0 --save-enhanced-frames --smooth --use-mean"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--skip-segmentation)
            SKIP_SEGMENTATION=true
            shift
            ;;
        -d|--base-dir)
            if [[ -n "$2" && "$2" != -* ]]; then
                BASE_DIR_RELATIVE="$2"
                shift 2
            else
                echo "Error: --base-dir requires a value"
                exit 1
            fi
            ;;
        --time-bin-us)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                TIME_BIN_US="$2"
                shift 2
            else
                echo "Error: --time-bin-us requires a value"
                exit 1
            fi
            ;;
        --max-events)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                MAX_EVENTS="$2"
                shift 2
            else
                echo "Error: --max-events requires a value"
                exit 1
            fi
            ;;
        -b|--bin-width)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                BIN_WIDTH="$2"
                shift 2
            else
                echo "Error: --bin-width requires a value"
                exit 1
            fi
            ;;
        -x|--initial-a-x)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                INITIAL_A_X="$2"
                shift 2
            else
                echo "Error: --initial-a-x requires a value"
                exit 1
            fi
            ;;
        -y|--initial-a-y)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                INITIAL_A_Y="$2"
                shift 2
            else
                echo "Error: --initial-a-y requires a value"
                exit 1
            fi
            ;;
        --sensor-width)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                SENSOR_WIDTH="$2"
                shift 2
            else
                echo "Error: --sensor-width requires a value"
                exit 1
            fi
            ;;
        --sensor-height)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                SENSOR_HEIGHT="$2"
                shift 2
            else
                echo "Error: --sensor-height requires a value"
                exit 1
            fi
            ;;
        --iterations)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                ITERATIONS="$2"
                shift 2
            else
                echo "Error: --iterations requires a value"
                exit 1
            fi
            ;;
        --learning-rate)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                LEARNING_RATE="$2"
                shift 2
            else
                echo "Error: --learning-rate requires a value"
                exit 1
            fi
            ;;
        --direct-params)
            DIRECT_PARAMS=true
            shift
            ;;
        --no-visualize)
            ENABLE_VISUALIZE=false
            shift
            ;;
        --save-frames)
            SAVE_FRAMES=true
            shift
            ;;
        --save-enhanced-frames)
            SAVE_ENHANCED_FRAMES=true
            shift
            ;;
        --smooth)
            SMOOTH=true
            shift
            ;;
        --use-mean)
            USE_MEAN=true
            shift
            ;;
        --max-comparison-frames)
            if [[ -n "$2" && ( "$2" != -* || "$2" =~ ^-[0-9] ) ]]; then
                MAX_COMPARISON_FRAMES="$2"
                shift 2
            else
                echo "Error: --max-comparison-frames requires a value"
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set the base directory
BASE_DIR="$SCRIPT_DIR/$BASE_DIR_RELATIVE"

echo "Script directory: $SCRIPT_DIR"
echo "Processing recordings in: $BASE_DIR"
echo "Configuration:"
echo "  Skip segmentation: $SKIP_SEGMENTATION"
echo ""
echo "Segmentation settings:"
echo "  Time bin (μs): $TIME_BIN_US"
echo "  Max events: ${MAX_EVENTS:-unlimited}"
echo ""
echo "Scan compensation settings:"
echo "  Bin width (μs): $BIN_WIDTH"
echo "  Initial A_x: $INITIAL_A_X"
echo "  Initial A_y: $INITIAL_A_Y"
echo "  Sensor size: ${SENSOR_WIDTH}x${SENSOR_HEIGHT}"
echo "  Iterations: $ITERATIONS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Direct params: $DIRECT_PARAMS"
echo "  Enable visualization: $ENABLE_VISUALIZE"
echo "  Save frames: $SAVE_FRAMES"
echo "  Save enhanced frames: $SAVE_ENHANCED_FRAMES"
echo "  Apply smoothing: $SMOOTH"
echo "  Use mean (vs median): $USE_MEAN"
echo "  Max comparison frames: ${MAX_COMPARISON_FRAMES:-unlimited}"

# Python environment activation (adjust if needed)
# source activate nhi_test  # Uncomment if you need to activate conda environment

# Script paths (in the same directory as this script)
SEGMENTATION_SCRIPT="$SCRIPT_DIR/simple_autocorr_analysis_segment_robust.py"
ALIGNMENT_SCRIPT="$SCRIPT_DIR/scanning_alignment_with_merge.py"

# Check if scripts exist
if [[ "$SKIP_SEGMENTATION" == false && ! -f "$SEGMENTATION_SCRIPT" ]]; then
    echo "Error: Segmentation script not found at $SEGMENTATION_SCRIPT"
    echo "Please adjust SEGMENTATION_SCRIPT path in this script or use --skip-segmentation"
    exit 1
fi

if [[ ! -f "$ALIGNMENT_SCRIPT" ]]; then
    echo "Error: Alignment script not found at $ALIGNMENT_SCRIPT"
    echo "Please adjust ALIGNMENT_SCRIPT path in this script"
    exit 1
fi

# Function to process a single recording
process_recording() {
    local folder="$1"
    local raw_file="$2"
    local base_name="$3"
    
    echo ""
    echo "=" $(printf '%.0s' {1..80})
    echo "PROCESSING: $folder"
    echo "=" $(printf '%.0s' {1..80})
    echo "RAW file: $raw_file"
    echo "Base name: $base_name"
    
    # Get absolute path for the raw file
    raw_file_abs="$(pwd)/$raw_file"
    
    # Step 1: Run segmentation analysis (if not skipped)
    if [[ "$SKIP_SEGMENTATION" == false ]]; then
        echo ""
        echo "Step 1: Running segmentation analysis..."
        
        # Build segmentation command
        SEG_CMD="python \"$SEGMENTATION_SCRIPT\" \"$raw_file_abs\" --segment_events --time_bin_us $TIME_BIN_US"
        
        # Add max_events if specified
        if [[ -n "$MAX_EVENTS" ]]; then
            SEG_CMD="$SEG_CMD --max_events $MAX_EVENTS"
        fi
        
        echo "Command: $SEG_CMD"
        
        if eval $SEG_CMD; then
            echo "✓ Segmentation completed successfully"
        else
            echo "✗ Segmentation failed for $folder"
            return 1
        fi
    else
        echo ""
        echo "Step 1: Skipping segmentation analysis (--skip-segmentation flag used)"
    fi
    
    # Step 2: Check if segments folder exists
    segments_folder="$(pwd)/${folder}/${base_name}_segments"
    if [[ ! -d "$segments_folder" ]]; then
        if [[ "$SKIP_SEGMENTATION" == false ]]; then
            echo "✗ Segments folder not found: $segments_folder"
            return 1
        else
            echo "✗ Segments folder not found: $segments_folder"
            echo "  Since segmentation was skipped, the segments folder must already exist"
            return 1
        fi
    fi
    
    echo "✓ Found segments folder: $segments_folder"
    
    # Step 3: Run scan compensation with merge
    echo ""
    if [[ "$SKIP_SEGMENTATION" == false ]]; then
        echo "Step 2: Running scan compensation with merge..."
    else
        echo "Step 1: Running scan compensation with merge..."
    fi
    
    # Build scan compensation command
    ALIGN_CMD="python \"$ALIGNMENT_SCRIPT\" \"$segments_folder\" --merge"
    ALIGN_CMD="$ALIGN_CMD --bin_width $BIN_WIDTH"
    ALIGN_CMD="$ALIGN_CMD --sensor_width $SENSOR_WIDTH"
    ALIGN_CMD="$ALIGN_CMD --sensor_height $SENSOR_HEIGHT"
    ALIGN_CMD="$ALIGN_CMD --iterations $ITERATIONS"
    ALIGN_CMD="$ALIGN_CMD --learning_rate $LEARNING_RATE"
    ALIGN_CMD="$ALIGN_CMD --initial_a_x $INITIAL_A_X"
    ALIGN_CMD="$ALIGN_CMD --initial_a_y $INITIAL_A_Y"
    
    # Add conditional flags
    if [[ "$ENABLE_VISUALIZE" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --visualize"
    fi
    
    if [[ "$DIRECT_PARAMS" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --direct_params"
    fi
    
    if [[ "$SAVE_FRAMES" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --save_frames"
    fi
    
    if [[ "$SAVE_ENHANCED_FRAMES" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --save_enhanced_frames"
    fi
    
    if [[ "$SMOOTH" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --smooth"
    fi
    
    if [[ "$USE_MEAN" == true ]]; then
        ALIGN_CMD="$ALIGN_CMD --use_mean"
    fi
    
    if [[ -n "$MAX_COMPARISON_FRAMES" ]]; then
        ALIGN_CMD="$ALIGN_CMD --max_comparison_frames $MAX_COMPARISON_FRAMES"
    fi
    
    echo "Command: $ALIGN_CMD"
    
    # Execute command
    if eval $ALIGN_CMD; then
        echo "✓ Scan compensation completed successfully"
    else
        echo "✗ Scan compensation failed for $folder"
        return 1
    fi
    
    echo "✓ Processing completed for $folder"
    return 0
}

# Find all folders and process them
total_folders=0
successful=0
failed=0
failed_folders=()

# Change to the base directory
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory not found at $BASE_DIR"
    echo "You can specify a different directory with --base-dir"
    exit 1
fi

cd "$BASE_DIR"
echo "Changed to directory: $(pwd)"

echo "Scanning for recording folders..."

# Process each subdirectory that contains .raw files
for folder in */; do
    # Remove trailing slash
    folder="${folder%/}"
    
    if [[ -d "$folder" ]]; then
        echo "Checking folder: $folder"
        
        # Find the .raw file in this folder
        raw_files=("$folder"/*.raw)
        
        if [[ ${#raw_files[@]} -eq 0 ]] || [[ ! -f "${raw_files[0]}" ]]; then
            echo "  No .raw file found in $folder, skipping"
            continue
        fi
        
        if [[ ${#raw_files[@]} -gt 1 ]]; then
            echo "  ⚠ Multiple .raw files found in $folder, using first one"
        fi
        
        raw_file="${raw_files[0]}"
        raw_file_name=$(basename "$raw_file")
        base_name=$(basename "$raw_file" .raw)
        
        echo "  Found .raw file: $raw_file_name"
        
        ((total_folders++))
        
        # Process this recording
        if process_recording "$folder" "$raw_file" "$base_name"; then
            ((successful++))
        else
            ((failed++))
            failed_folders+=("$folder")
        fi
    fi
done

# Summary
echo ""
echo "=" $(printf '%.0s' {1..80})
echo "PROCESSING SUMMARY"
echo "=" $(printf '%.0s' {1..80})
echo "Total folders processed: $total_folders"
echo "Successful: $successful"
echo "Failed: $failed"
echo ""
echo "Configuration used:"
echo "  Skip segmentation: $SKIP_SEGMENTATION"
echo "  Time bin (μs): $TIME_BIN_US"
echo "  Max events: ${MAX_EVENTS:-unlimited}"
echo "  Bin width (μs): $BIN_WIDTH"
echo "  Initial A_x: $INITIAL_A_X"
echo "  Initial A_y: $INITIAL_A_Y"
echo "  Sensor size: ${SENSOR_WIDTH}x${SENSOR_HEIGHT}"
echo "  Iterations: $ITERATIONS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Direct params: $DIRECT_PARAMS"
echo "  Enable visualization: $ENABLE_VISUALIZE"
echo "  Save frames: $SAVE_FRAMES"
echo "  Save enhanced frames: $SAVE_ENHANCED_FRAMES"
echo "  Apply smoothing: $SMOOTH"
echo "  Use mean: $USE_MEAN"
echo "  Max comparison frames: ${MAX_COMPARISON_FRAMES:-unlimited}"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "Failed folders:"
    for folder in "${failed_folders[@]}"; do
        echo "  - $folder"
    done
fi

echo ""
echo "Processing complete!"

# Exit with appropriate code
if [[ $failed -eq 0 ]]; then
    exit 0
else
    exit 1
fi