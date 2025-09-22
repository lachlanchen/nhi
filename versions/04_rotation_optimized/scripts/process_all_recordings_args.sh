#!/bin/bash

# Script to process all synchronized recordings with segmentation and scan compensation
# Usage: ./process_all_recordings.sh [OPTIONS]
#
# OPTIONS:
#   -h, --help              Show this help message
#   -s, --skip-segmentation Skip the segmentation step
#   -b, --bin-width VALUE   Set bin width (default: 50000)
#   -x, --initial-a-x VALUE Set initial A_x parameter (default: 0.541157)
#   -y, --initial-a-y VALUE Set initial A_y parameter (default: -75.052391)
#   -d, --base-dir DIR      Set base directory (default: ./plastics)
#   --no-visualize          Disable visualization in scan compensation
#
# Examples:
#   ./process_all_recordings.sh
#   ./process_all_recordings.sh --skip-segmentation
#   ./process_all_recordings.sh -b 75000 -x 0.6 -y -80.0
#   ./process_all_recordings.sh --skip-segmentation --bin-width 100000 --no-visualize

# Default values
SKIP_SEGMENTATION=false
INITIAL_A_X="0.541157"
INITIAL_A_Y="-75.052391"
BIN_WIDTH="50000"
ENABLE_VISUALIZE=true
SCRIPT_DIR="$(pwd)"
BASE_DIR_RELATIVE="plastics"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Process all synchronized recordings with segmentation and scan compensation"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --skip-segmentation Skip the segmentation step"
    echo "  -b, --bin-width VALUE   Set bin width (default: $BIN_WIDTH)"
    echo "  -x, --initial-a-x VALUE Set initial A_x parameter (default: $INITIAL_A_X)"
    echo "  -y, --initial-a-y VALUE Set initial A_y parameter (default: $INITIAL_A_Y)"
    echo "  -d, --base-dir DIR      Set base directory (default: $BASE_DIR_RELATIVE)"
    echo "  --no-visualize          Disable visualization in scan compensation"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --skip-segmentation"
    echo "  $0 -b 75000 -x 0.6 -y -80.0"
    echo "  $0 --skip-segmentation --bin-width 100000 --no-visualize"
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
        -b|--bin-width)
            if [[ -n "$2" && "$2" != -* ]]; then
                BIN_WIDTH="$2"
                shift 2
            else
                echo "Error: --bin-width requires a value"
                exit 1
            fi
            ;;
        -x|--initial-a-x)
            if [[ -n "$2" && "$2" != -* ]]; then
                INITIAL_A_X="$2"
                shift 2
            else
                echo "Error: --initial-a-x requires a value"
                exit 1
            fi
            ;;
        -y|--initial-a-y)
            if [[ -n "$2" && "$2" != -* ]]; then
                INITIAL_A_Y="$2"
                shift 2
            else
                echo "Error: --initial-a-y requires a value"
                exit 1
            fi
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
        --no-visualize)
            ENABLE_VISUALIZE=false
            shift
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
echo "  Bin width: $BIN_WIDTH"
echo "  Initial A_x: $INITIAL_A_X"
echo "  Initial A_y: $INITIAL_A_Y"
echo "  Enable visualization: $ENABLE_VISUALIZE"

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
        echo "Command: python \"$SEGMENTATION_SCRIPT\" \"$raw_file_abs\" --segment_events"
        
        if python "$SEGMENTATION_SCRIPT" "$raw_file_abs" --segment_events; then
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
    
    # Build command with conditional visualization
    VISUALIZE_FLAG=""
    if [[ "$ENABLE_VISUALIZE" == true ]]; then
        VISUALIZE_FLAG="--visualize"
    fi
    
    echo "Command: python \"$ALIGNMENT_SCRIPT\" \"$segments_folder\" --merge $VISUALIZE_FLAG --bin_width $BIN_WIDTH --save_frames --save_enhanced_frames --direct_params --initial_a_x $INITIAL_A_X --initial_a_y $INITIAL_A_Y"
    
    # Execute command with conditional visualization
    if [[ "$ENABLE_VISUALIZE" == true ]]; then
        python "$ALIGNMENT_SCRIPT" "$segments_folder" \
            --merge \
            --visualize \
            --bin_width "$BIN_WIDTH" \
            --save_frames \
            --save_enhanced_frames \
            --direct_params \
            --initial_a_x "$INITIAL_A_X" \
            --initial_a_y "$INITIAL_A_Y"
    else
        python "$ALIGNMENT_SCRIPT" "$segments_folder" \
            --merge \
            --bin_width "$BIN_WIDTH" \
            --save_frames \
            --save_enhanced_frames \
            --direct_params \
            --initial_a_x "$INITIAL_A_X" \
            --initial_a_y "$INITIAL_A_Y"
    fi
    
    if [[ $? -eq 0 ]]; then
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

# Process each folder
for folder in glass right1-* right2-* right3-*; do
    if [[ -d "$folder" ]]; then
        echo "Found folder: $folder"
        
        # Find the .raw file in this folder
        raw_files=("$folder"/*.raw)
        
        if [[ ${#raw_files[@]} -eq 0 ]] || [[ ! -f "${raw_files[0]}" ]]; then
            echo "✗ No .raw file found in $folder"
            continue
        fi
        
        if [[ ${#raw_files[@]} -gt 1 ]]; then
            echo "⚠ Multiple .raw files found in $folder, using first one"
        fi
        
        raw_file="${raw_files[0]}"
        raw_file_name=$(basename "$raw_file")
        base_name=$(basename "$raw_file" .raw)
        
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
echo "  Bin width: $BIN_WIDTH"
echo "  Initial A_x: $INITIAL_A_X"
echo "  Initial A_y: $INITIAL_A_Y"
echo "  Enable visualization: $ENABLE_VISUALIZE"

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