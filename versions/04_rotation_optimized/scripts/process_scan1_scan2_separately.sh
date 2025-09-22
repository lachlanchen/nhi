#!/bin/bash

# Script to process all synchronized recordings with segmentation and individual scan compensation
# Processes only Scan 1 and Scan 2 separately (no merge, no visualize)
# Usage: ./process_scan1_scan2_separately.sh [OPTIONS]
#
# Options:
#   --initial_a_x VALUE    Set initial a_x parameter (default: 2.763)
#   --initial_a_y VALUE    Set initial a_y parameter (default: -64.519)
#   --bin_width VALUE      Set bin width in microseconds (default: 50000)
#   --skip_segmentation    Skip step 1 (segmentation) - assumes segments already exist
#   --help                 Show this help message

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Process all synchronized recordings with segmentation and individual scan compensation."
    echo "Processes only Scan 1 and Scan 2 separately (no merge, no visualize)."
    echo ""
    echo "Options:"
    echo "  --initial_a_x VALUE    Set initial a_x parameter (default: 2.763)"
    echo "  --initial_a_y VALUE    Set initial a_y parameter (default: -64.519)"
    echo "  --bin_width VALUE      Set bin width in microseconds (default: 50000)"
    echo "  --skip_segmentation    Skip step 1 (segmentation) - assumes segments already exist"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default parameters"
    echo "  $0 --initial_a_x 0.5 --initial_a_y -75"
    echo "  $0 --bin_width 100000 --skip_segmentation"
    exit 0
}

# Default parameters
DEFAULT_A_X="2.763"
DEFAULT_A_Y="-64.519"
DEFAULT_BIN_WIDTH="50000"
SKIP_SEGMENTATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --initial_a_x)
            INITIAL_A_X="$2"
            shift 2
            ;;
        --initial_a_y)
            INITIAL_A_Y="$2"
            shift 2
            ;;
        --bin_width)
            BIN_WIDTH="$2"
            shift 2
            ;;
        --skip_segmentation)
            SKIP_SEGMENTATION=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
INITIAL_A_X="${INITIAL_A_X:-$DEFAULT_A_X}"
INITIAL_A_Y="${INITIAL_A_Y:-$DEFAULT_A_Y}"
BIN_WIDTH="${BIN_WIDTH:-$DEFAULT_BIN_WIDTH}"

# Validate numerical parameters (basic validation)
if ! [[ "$INITIAL_A_X" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: initial_a_x must be a number, got: $INITIAL_A_X"
    exit 1
fi

if ! [[ "$INITIAL_A_Y" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: initial_a_y must be a number, got: $INITIAL_A_Y"
    exit 1
fi

if ! [[ "$BIN_WIDTH" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: bin_width must be a positive number, got: $BIN_WIDTH"
    exit 1
fi

# Set the base directory to the plastics subdirectory
SCRIPT_DIR="$(pwd)"
BASE_DIR="$SCRIPT_DIR/plastics"
echo "Script directory: $SCRIPT_DIR"
echo "Processing recordings in: $BASE_DIR"
echo ""
echo "Parameters:"
echo "  Initial a_x: $INITIAL_A_X"
echo "  Initial a_y: $INITIAL_A_Y"
echo "  Bin width: $BIN_WIDTH μs"
echo "  Skip segmentation: $SKIP_SEGMENTATION"

# Python environment activation (adjust if needed)
# source activate nhi_test  # Uncomment if you need to activate conda environment

# Script paths (in the same directory as this script)
SEGMENTATION_SCRIPT="$SCRIPT_DIR/simple_autocorr_analysis_segment_robust.py"
ALIGNMENT_SCRIPT="$SCRIPT_DIR/scanning_alignment_with_merge.py"

# Check if scripts exist
if [[ "$SKIP_SEGMENTATION" == false ]] && [[ ! -f "$SEGMENTATION_SCRIPT" ]]; then
    echo "Error: Segmentation script not found at $SEGMENTATION_SCRIPT"
    echo "Please adjust SEGMENTATION_SCRIPT path in this script or use --skip_segmentation"
    exit 1
fi

if [[ ! -f "$ALIGNMENT_SCRIPT" ]]; then
    echo "Error: Alignment script not found at $ALIGNMENT_SCRIPT"
    echo "Please adjust ALIGNMENT_SCRIPT path in this script"
    exit 1
fi

# # Scan compensation parameters
# INITIAL_A_X="0.541157"
# INITIAL_A_Y="-75.052391"
# BIN_WIDTH="50000"

# Function to process individual scan files
process_individual_scan() {
    local scan_file="$1"
    local scan_name="$2"
    
    echo ""
    echo "Processing individual scan: $scan_name"
    echo "File: $scan_file"
    
    # Check if this is a backward scan and reverse the signs of ax and ay
    local use_a_x="$INITIAL_A_X"
    local use_a_y="$INITIAL_A_Y"
    
    if [[ "$scan_name" == *"Backward"* ]]; then
        # Reverse signs for backward scan
        use_a_x=$(echo "$INITIAL_A_X * -1" | bc -l)
        use_a_y=$(echo "$INITIAL_A_Y * -1" | bc -l)
        echo "Backward scan detected - reversing parameter signs"
        echo "Using parameters: a_x=$use_a_x, a_y=$use_a_y, bin_width=$BIN_WIDTH"
    else
        echo "Using parameters: a_x=$use_a_x, a_y=$use_a_y, bin_width=$BIN_WIDTH"
    fi
    
    if [[ ! -f "$scan_file" ]]; then
        echo "✗ Scan file not found: $scan_file"
        return 1
    fi
    
    echo "Command: python \"$ALIGNMENT_SCRIPT\" \"$scan_file\" --bin_width $BIN_WIDTH --save_frames --save_enhanced_frames --direct_params --initial_a_x $use_a_x --initial_a_y $use_a_y"
    
    if python "$ALIGNMENT_SCRIPT" "$scan_file" \
        --bin_width "$BIN_WIDTH" \
        --save_frames \
        --save_enhanced_frames \
        --direct_params \
        --initial_a_x "$use_a_x" \
        --initial_a_y "$use_a_y"; then
        echo "✓ Individual scan processing completed for $scan_name"
        return 0
    else
        echo "✗ Individual scan processing failed for $scan_name"
        return 1
    fi
}

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
    
    # Step 1: Run segmentation analysis (unless skipped)
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
        echo "Step 1: Skipping segmentation analysis (--skip_segmentation specified)"
    fi
    
    # Step 2: Check if segments folder exists
    segments_folder="$(pwd)/${folder}/${base_name}_segments"
    if [[ ! -d "$segments_folder" ]]; then
        echo "✗ Segments folder not found: $segments_folder"
        if [[ "$SKIP_SEGMENTATION" == true ]]; then
            echo "   Note: Segmentation was skipped - make sure segments were created previously"
        fi
        return 1
    fi
    
    echo "✓ Found segments folder: $segments_folder"
    
    # Step 3: Process Scan 1 and Scan 2 individually
    echo ""
    if [[ "$SKIP_SEGMENTATION" == false ]]; then
        echo "Step 2: Processing individual scans (Scan 1 and Scan 2)..."
    else
        echo "Step 1: Processing individual scans (Scan 1 and Scan 2)..."
    fi
    
    # Look for Scan 1 and Scan 2 files
    scan1_file="$segments_folder/Scan_1_Forward_events.npz"
    scan2_file="$segments_folder/Scan_2_Backward_events.npz"
    
    local scan1_success=false
    local scan2_success=false
    
    # Process Scan 1
    if [[ -f "$scan1_file" ]]; then
        if process_individual_scan "$scan1_file" "Scan_1_Forward"; then
            scan1_success=true
        fi
    else
        echo "⚠ Scan 1 file not found: $scan1_file"
    fi
    
    # Process Scan 2
    if [[ -f "$scan2_file" ]]; then
        if process_individual_scan "$scan2_file" "Scan_2_Backward"; then
            scan2_success=true
        fi
    else
        echo "⚠ Scan 2 file not found: $scan2_file"
    fi
    
    # Check if at least one scan was processed successfully
    if [[ "$scan1_success" == true ]] || [[ "$scan2_success" == true ]]; then
        echo "✓ Individual scan processing completed for $folder"
        return 0
    else
        echo "✗ No scans were processed successfully for $folder"
        return 1
    fi
}

# Change to the plastics directory
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Plastics directory not found at $BASE_DIR"
    exit 1
fi

cd "$BASE_DIR"
echo "Changed to directory: $(pwd)"

# Find all folders and process them
total_folders=0
successful=0
failed=0
failed_folders=()

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
echo "Parameters used:"
echo "  Initial a_x: $INITIAL_A_X"
echo "  Initial a_y: $INITIAL_A_Y"
echo "  Bin width: $BIN_WIDTH μs"
echo "  Skip segmentation: $SKIP_SEGMENTATION"
echo ""
echo "Results:"
echo "Total folders processed: $total_folders"
echo "Successful: $successful"
echo "Failed: $failed"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "Failed folders:"
    for folder in "${failed_folders[@]}"; do
        echo "  - $folder"
    done
fi

echo ""
echo "Individual scan processing complete!"
echo "Note: Only Scan 1 (Forward) and Scan 2 (Backward) were processed individually"
if [[ "$SKIP_SEGMENTATION" == true ]]; then
    echo "Note: Segmentation step was skipped"
fi
echo "Parameters: a_x=$INITIAL_A_X, a_y=$INITIAL_A_Y, bin_width=$BIN_WIDTH μs"

# Exit with appropriate code
if [[ $failed -eq 0 ]]; then
    exit 0
else
    exit 1
fi