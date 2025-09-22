#!/bin/bash

# Script to process all synchronized recordings with segmentation and scan compensation
# Usage: ./process_all_recordings.sh

# Set the base directory to the plastics subdirectory
SCRIPT_DIR="$(pwd)"
BASE_DIR="$SCRIPT_DIR/plastics"
echo "Script directory: $SCRIPT_DIR"
echo "Processing recordings in: $BASE_DIR"

# Python environment activation (adjust if needed)
# source activate nhi_test  # Uncomment if you need to activate conda environment

# Script paths (in the same directory as this script)
SEGMENTATION_SCRIPT="$SCRIPT_DIR/simple_autocorr_analysis_segment_robust.py"
ALIGNMENT_SCRIPT="$SCRIPT_DIR/scanning_alignment_with_merge.py"

# Check if scripts exist
if [[ ! -f "$SEGMENTATION_SCRIPT" ]]; then
    echo "Error: Segmentation script not found at $SEGMENTATION_SCRIPT"
    echo "Please adjust SEGMENTATION_SCRIPT path in this script"
    exit 1
fi

if [[ ! -f "$ALIGNMENT_SCRIPT" ]]; then
    echo "Error: Alignment script not found at $ALIGNMENT_SCRIPT"
    echo "Please adjust ALIGNMENT_SCRIPT path in this script"
    exit 1
fi

# Scan compensation parameters
INITIAL_A_X="0.541157"
INITIAL_A_Y="-75.052391"
BIN_WIDTH="50000"

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
    
    # Step 1: Run segmentation analysis
    echo ""
    echo "Step 1: Running segmentation analysis..."
    echo "Command: python \"$SEGMENTATION_SCRIPT\" \"$raw_file_abs\" --segment_events"
    
    if python "$SEGMENTATION_SCRIPT" "$raw_file_abs" --segment_events; then
        echo "✓ Segmentation completed successfully"
    else
        echo "✗ Segmentation failed for $folder"
        return 1
    fi
    
    # Step 2: Check if segments folder was created
    segments_folder="$(pwd)/${folder}/${base_name}_segments"
    if [[ ! -d "$segments_folder" ]]; then
        echo "✗ Segments folder not found: $segments_folder"
        return 1
    fi
    
    echo "✓ Found segments folder: $segments_folder"
    
    # Step 3: Run scan compensation with merge
    echo ""
    echo "Step 2: Running scan compensation with merge..."
    echo "Command: python \"$ALIGNMENT_SCRIPT\" \"$segments_folder\" --merge --visualize --bin_width $BIN_WIDTH --save_frames --save_enhanced_frames --direct_params --initial_a_x $INITIAL_A_X --initial_a_y $INITIAL_A_Y"
    
    if python "$ALIGNMENT_SCRIPT" "$segments_folder" \
        --merge \
        --visualize \
        --bin_width "$BIN_WIDTH" \
        --save_frames \
        --save_enhanced_frames \
        --direct_params \
        --initial_a_x "$INITIAL_A_X" \
        --initial_a_y "$INITIAL_A_Y"; then
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

# Change to the plastics directory
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Plastics directory not found at $BASE_DIR"
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
