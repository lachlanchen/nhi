#!/bin/bash

# Segmentation-only script to process all synchronized recordings
# Usage: ./process_segmentation_only.sh [OPTIONS]
#
# OPTIONS:
#   -h, --help                      Show this help message
#   -d, --base-dir DIR              Set base directory (default: ./scan_speed_test)
#
# SEGMENTATION OPTIONS:
#   --time-bin-us VALUE             Set time bin size in microseconds for segmentation (default: 1000)
#   --max-events VALUE              Maximum events to load for segmentation
#
# Examples:
#   ./process_segmentation_only.sh
#   ./process_segmentation_only.sh --time-bin-us 500
#   ./process_segmentation_only.sh --max-events 1000000
#   ./process_segmentation_only.sh -d ./my_recordings --time-bin-us 2000

# Default values
SCRIPT_DIR="$(pwd)"
BASE_DIR_RELATIVE="scan_speed_test"

# Segmentation options
TIME_BIN_US="1000"
MAX_EVENTS=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Segmentation-only script to process all synchronized recordings"
    echo ""
    echo "GENERAL OPTIONS:"
    echo "  -h, --help                      Show this help message"
    echo "  -d, --base-dir DIR              Set base directory (default: $BASE_DIR_RELATIVE)"
    echo ""
    echo "SEGMENTATION OPTIONS:"
    echo "  --time-bin-us VALUE             Set time bin size in microseconds (default: $TIME_BIN_US)"
    echo "  --max-events VALUE              Maximum events to load for segmentation"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --time-bin-us 500"
    echo "  $0 --max-events 1000000"
    echo "  $0 -d ./my_recordings --time-bin-us 2000"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
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
echo ""
echo "Segmentation settings:"
echo "  Time bin (μs): $TIME_BIN_US"
echo "  Max events: ${MAX_EVENTS:-unlimited}"
echo ""

# Python environment activation (adjust if needed)
# source activate nhi_test  # Uncomment if you need to activate conda environment

# Script path (in the same directory as this script)
SEGMENTATION_SCRIPT="$SCRIPT_DIR/simple_autocorr_analysis_segment_robust.py"

# Check if script exists
if [[ ! -f "$SEGMENTATION_SCRIPT" ]]; then
    echo "Error: Segmentation script not found at $SEGMENTATION_SCRIPT"
    echo "Please adjust SEGMENTATION_SCRIPT path in this script"
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
    
    # Run segmentation analysis
    echo ""
    echo "Running segmentation analysis..."
    
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
    
    # Check if segments folder was created
    segments_folder="$(pwd)/${folder}/${base_name}_segments"
    if [[ ! -d "$segments_folder" ]]; then
        echo "✗ Segments folder not found: $segments_folder"
        return 1
    fi
    
    echo "✓ Found segments folder: $segments_folder"
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
for folder in sync_recording_*; do
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
echo "SEGMENTATION PROCESSING SUMMARY"
echo "=" $(printf '%.0s' {1..80})
echo "Total folders processed: $total_folders"
echo "Successful: $successful"
echo "Failed: $failed"
echo ""
echo "Configuration used:"
echo "  Time bin (μs): $TIME_BIN_US"
echo "  Max events: ${MAX_EVENTS:-unlimited}"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "Failed folders:"
    for folder in "${failed_folders[@]}"; do
        echo "  - $folder"
    done
fi

echo ""
echo "Segmentation processing complete!"

# Exit with appropriate code
if [[ $failed -eq 0 ]]; then
    exit 0
else
    exit 1
fi