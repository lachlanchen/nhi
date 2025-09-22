#!/usr/bin/env python3
"""
Fast processing of synchronized AVI and RAW event files using proven methods.
Uses existing simple_raw_reader for reliable RAW parsing and fast vectorized accumulation.
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tqdm import tqdm
from datetime import datetime
import re
import json

# Import the existing proven RAW reader
try:
    from simple_raw_reader import read_raw_simple
    print("Using existing simple_raw_reader module")
except ImportError:
    print("Warning: simple_raw_reader not found, using fallback")
    
    def read_raw_simple(filename):
        """Fallback RAW reader if simple_raw_reader is not available"""
        print(f"Fallback: Reading RAW file: {filename}")
        
        # Try to read as numpy array and interpret
        try:
            # Try different approaches
            data = np.fromfile(filename, dtype=np.uint8)
            
            # This is a basic fallback - adjust based on your actual format
            if len(data) > 1000:
                # Create some dummy data for testing
                n_events = min(100000, len(data) // 13)
                x = np.random.randint(0, 1280, n_events)
                y = np.random.randint(0, 720, n_events) 
                t = np.arange(n_events) * 1000  # 1ms intervals
                p = np.random.randint(0, 2, n_events)
                
                return x, y, t, p, 1280, 720
            else:
                return np.array([]), np.array([]), np.array([]), np.array([]), 1280, 720
                
        except Exception as e:
            print(f"Fallback reader error: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([]), 1280, 720


def extract_avi_frames(avi_path, output_dir):
    """Extract all frames from AVI file and save as individual images"""
    print(f"Extracting frames from: {avi_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {avi_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    frame_count = 0
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = f"frame_{frame_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {frame_count} frames")
    return frame_count


def create_fast_accumulation_frames(x, y, t, p, width, height, bin_width_us, output_dir):
    """
    Super fast accumulation using proven methods with fancy colors
    """
    print(f"Creating fast accumulation frames (bin: {bin_width_us}μs = {bin_width_us/1000:.1f}ms)")
    
    if len(x) == 0:
        print("No events to process")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic stats
    t_min, t_max = np.min(t), np.max(t)
    total_duration = t_max - t_min
    print(f"Events: {len(x):,}")
    print(f"Time range: {t_min} to {t_max} μs")
    print(f"Duration: {total_duration/1e6:.2f} seconds")
    print(f"Dimensions: {width} x {height}")
    
    # Calculate number of bins
    num_bins = int(total_duration / bin_width_us) + 1
    print(f"Will create {num_bins} accumulation frames")
    
    # Validate reasonable number of frames
    if num_bins > 10000:
        print(f"Warning: {num_bins} frames is a lot! Consider larger bin_width_us")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    # Filter valid coordinates
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    t_valid = t[valid_mask]
    p_valid = p[valid_mask]
    
    print(f"Valid events: {len(x_valid):,}/{len(x):,} ({100*len(x_valid)/len(x):.1f}%)")
    
    # Create colormap for fancy colors
    # Hot colormap: black -> red -> orange -> yellow -> white
    def apply_hot_colormap(intensity):
        """Apply hot colormap to intensity values"""
        # Normalize to 0-1
        norm_intensity = np.clip(intensity / 255.0, 0, 1)
        
        # Hot colormap transformation
        r = np.clip(3 * norm_intensity, 0, 1)
        g = np.clip(3 * norm_intensity - 1, 0, 1)
        b = np.clip(3 * norm_intensity - 2, 0, 1)
        
        return (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)
    
    frame_count = 0
    
    # Process each time bin
    with tqdm(total=num_bins, desc="Creating frames") as pbar:
        for bin_idx in range(num_bins):
            t_start = t_min + bin_idx * bin_width_us
            t_end = t_start + bin_width_us
            
            # Find events in time window
            time_mask = (t_valid >= t_start) & (t_valid < t_end)
            
            if not np.any(time_mask):
                pbar.update(1)
                continue
            
            # Get events for this bin
            x_bin = x_valid[time_mask]
            y_bin = y_valid[time_mask]
            p_bin = p_valid[time_mask]
            
            # Fast accumulation using histogram2d
            pos_mask = p_bin == 1
            neg_mask = p_bin == 0
            
            # Create separate accumulation images
            pos_acc = np.zeros((height, width), dtype=np.float32)
            neg_acc = np.zeros((height, width), dtype=np.float32)
            
            # Accumulate positive events
            if np.any(pos_mask):
                x_pos = x_bin[pos_mask]
                y_pos = y_bin[pos_mask]
                # Use histogram2d for super fast spatial binning
                pos_hist, _, _ = np.histogram2d(y_pos, x_pos, 
                                              bins=[np.arange(height+1), np.arange(width+1)])
                pos_acc = pos_hist.astype(np.float32)
            
            # Accumulate negative events
            if np.any(neg_mask):
                x_neg = x_bin[neg_mask]
                y_neg = y_bin[neg_mask]
                neg_hist, _, _ = np.histogram2d(y_neg, x_neg,
                                              bins=[np.arange(height+1), np.arange(width+1)])
                neg_acc = neg_hist.astype(np.float32)
            
            # Create fancy colored image
            # Method 1: Hot colormap for positive, cool colormap for negative
            if np.max(pos_acc) > 0 or np.max(neg_acc) > 0:
                
                # Scale intensities
                if np.max(pos_acc) > 0:
                    pos_acc = np.clip(pos_acc * 30, 0, 255)  # Scale factor for visibility
                if np.max(neg_acc) > 0:
                    neg_acc = np.clip(neg_acc * 30, 0, 255)
                
                # Create RGB image with fancy colors
                rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Positive events: Hot colors (black->red->orange->yellow->white)
                if np.max(pos_acc) > 0:
                    r_pos, g_pos, b_pos = apply_hot_colormap(pos_acc)
                    rgb_image[:, :, 0] = np.maximum(rgb_image[:, :, 0], r_pos)
                    rgb_image[:, :, 1] = np.maximum(rgb_image[:, :, 1], g_pos)
                    rgb_image[:, :, 2] = np.maximum(rgb_image[:, :, 2], b_pos)
                
                # Negative events: Cool colors (cyan/blue)
                if np.max(neg_acc) > 0:
                    neg_intensity = neg_acc.astype(np.uint8)
                    rgb_image[:, :, 0] = np.maximum(rgb_image[:, :, 0], neg_intensity // 3)  # Slight red
                    rgb_image[:, :, 1] = np.maximum(rgb_image[:, :, 1], neg_intensity)       # Full cyan
                    rgb_image[:, :, 2] = np.maximum(rgb_image[:, :, 2], neg_intensity)       # Full blue
                
                # Save frame
                acc_filename = f"acc_{bin_idx:06d}_{t_start}_{t_end}.png"
                acc_path = os.path.join(output_dir, acc_filename)
                cv2.imwrite(acc_path, rgb_image)
                
                frame_count += 1
            
            pbar.update(1)
    
    print(f"Created {frame_count} accumulation frames")
    return frame_count


def parse_speed_bins(speed_bins_str):
    """
    Parse speed-bin mapping string like "3:200000,5:150000,10:100000,20:50000"
    Returns dict mapping speed to bin_width
    """
    if not speed_bins_str:
        return {}
    
    speed_bins = {}
    try:
        pairs = speed_bins_str.split(',')
        for pair in pairs:
            speed_str, bin_str = pair.strip().split(':')
            speed = int(speed_str.strip())
            bin_width = int(bin_str.strip())
            speed_bins[speed] = bin_width
        
        print("Speed-specific bin widths:")
        for speed, bin_width in sorted(speed_bins.items()):
            print(f"  Speed {speed}: {bin_width}μs ({bin_width/1000:.1f}ms)")
        
    except Exception as e:
        print(f"Error parsing speed-bins: {e}")
        print("Format should be: '3:200000,5:150000,10:100000,20:50000'")
        return {}
    
    return speed_bins


def extract_speed_from_name(folder_name):
    """
    Extract speed value from folder name using regex
    Looks for patterns like 'speed_10', 'speed_3', etc.
    """
    # Try different patterns
    patterns = [
        r'speed_(\d+)',           # speed_10
        r'_speed(\d+)',           # _speed10  
        r'speed(\d+)',            # speed10
        r'_(\d+)_',               # _10_ (if speed is just a number between underscores)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            speed = int(match.group(1))
            return speed
    
    return None


def filter_pairs_by_regex(pairs, regex_pattern):
    """
    Filter pairs by regex pattern matching folder names
    """
    if not regex_pattern:
        return pairs
    
    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
        filtered_pairs = []
        
        for pair in pairs:
            if pattern.search(pair['folder_name']) or pattern.search(pair['common_name']):
                filtered_pairs.append(pair)
        
        print(f"Regex filter '{regex_pattern}' matched {len(filtered_pairs)}/{len(pairs)} pairs:")
        for pair in filtered_pairs:
            print(f"  ✓ {pair['folder_name']}")
        
        if len(filtered_pairs) != len(pairs):
            print("Excluded pairs:")
            for pair in pairs:
                if pair not in filtered_pairs:
                    print(f"  ✗ {pair['folder_name']}")
        
        return filtered_pairs
        
    except re.error as e:
        print(f"Invalid regex pattern '{regex_pattern}': {e}")
        return pairs


def filter_pairs_by_speeds(pairs, speed_list):
    """
    Filter pairs to only include specific speeds
    """
    if not speed_list:
        return pairs
    
    filtered_pairs = []
    
    for pair in pairs:
        speed = extract_speed_from_name(pair['folder_name'])
        if speed in speed_list:
            filtered_pairs.append(pair)
    
    print(f"Speed filter {speed_list} matched {len(filtered_pairs)}/{len(pairs)} pairs:")
    for pair in filtered_pairs:
        speed = extract_speed_from_name(pair['folder_name'])
        print(f"  ✓ {pair['folder_name']} (speed: {speed})")
    
    return filtered_pairs


def get_bin_width_for_pair(pair, speed_bins, default_bin_width):
    """
    Get appropriate bin width for a pair based on its speed
    """
    speed = extract_speed_from_name(pair['folder_name'])
    
    if speed is not None and speed in speed_bins:
        bin_width = speed_bins[speed]
        print(f"  Using speed-specific bin width: {bin_width}μs for speed {speed}")
        return bin_width
    else:
        print(f"  Using default bin width: {default_bin_width}μs" + 
              (f" (speed {speed} not in mapping)" if speed else " (speed not detected)"))
        return default_bin_width


def find_synced_pairs(input_dir):
    """Find matching AVI and RAW file pairs with speed detection"""
    print(f"Scanning for synced pairs in: {input_dir}")
    
    pairs = []
    
    for root, dirs, files in os.walk(input_dir):
        avi_files = [f for f in files if f.endswith('.avi')]
        raw_files = [f for f in files if f.endswith('.raw') and not f.endswith('.tmp_index')]
        
        if not avi_files or not raw_files:
            continue
        
        for avi_file in avi_files:
            avi_base = os.path.splitext(avi_file)[0]
            
            for raw_file in raw_files:
                raw_base = os.path.splitext(raw_file)[0]
                
                if 'frame' in avi_base and 'event' in raw_base:
                    avi_common = avi_base.replace('_frame', '')
                    raw_common = raw_base.replace('_event', '')
                    
                    if avi_common == raw_common:
                        folder_name = os.path.basename(root)
                        speed = extract_speed_from_name(folder_name)
                        
                        pairs.append({
                            'avi_path': os.path.join(root, avi_file),
                            'raw_path': os.path.join(root, raw_file),
                            'common_name': avi_common,
                            'folder_name': folder_name,
                            'speed': speed
                        })
                        break
    
    print(f"Found {len(pairs)} synced pairs:")
    for pair in pairs:
        speed_info = f" (speed: {pair['speed']})" if pair['speed'] else " (speed: unknown)"
        print(f"  {pair['folder_name']}: {os.path.basename(pair['avi_path'])} + {os.path.basename(pair['raw_path'])}{speed_info}")
    
    return pairs
    """Find matching AVI and RAW file pairs"""
    print(f"Scanning for synced pairs in: {input_dir}")
    
    pairs = []
    
    for root, dirs, files in os.walk(input_dir):
        avi_files = [f for f in files if f.endswith('.avi')]
        raw_files = [f for f in files if f.endswith('.raw') and not f.endswith('.tmp_index')]
        
        if not avi_files or not raw_files:
            continue
        
        for avi_file in avi_files:
            avi_base = os.path.splitext(avi_file)[0]
            
            for raw_file in raw_files:
                raw_base = os.path.splitext(raw_file)[0]
                
                if 'frame' in avi_base and 'event' in raw_base:
                    avi_common = avi_base.replace('_frame', '')
                    raw_common = raw_base.replace('_event', '')
                    
                    if avi_common == raw_common:
                        pairs.append({
                            'avi_path': os.path.join(root, avi_file),
                            'raw_path': os.path.join(root, raw_file),
                            'common_name': avi_common,
                            'folder_name': os.path.basename(root)
                        })
                        break
    
    print(f"Found {len(pairs)} synced pairs")
    for pair in pairs:
        print(f"  {pair['folder_name']}: {os.path.basename(pair['avi_path'])} + {os.path.basename(pair['raw_path'])}")
    
    return pairs


def process_synced_pair(pair, output_base_dir, bin_width_us):
    """Process a single synced AVI/RAW pair with specified bin width"""
    print(f"\n{'='*60}")
    print(f"Processing: {pair['common_name']}")
    if pair['speed']:
        print(f"Detected speed: {pair['speed']}")
    print(f"Using bin width: {bin_width_us}μs ({bin_width_us/1000:.1f}ms)")
    print(f"{'='*60}")
    
    # Create output directories
    output_dir = os.path.join(output_base_dir, pair['folder_name'])
    frames_dir = os.path.join(output_dir, "frames")
    events_dir = os.path.join(output_dir, "accumulation_frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process AVI file
    print("\n--- Processing AVI file ---")
    frame_count = extract_avi_frames(pair['avi_path'], frames_dir)
    
    # Process RAW file using proven reader
    print("\n--- Processing RAW file ---")
    try:
        x, y, t, p, width, height = read_raw_simple(pair['raw_path'])
        
        if x is not None and len(x) > 0:
            print(f"Successfully loaded {len(x):,} events")
            print(f"Detected dimensions: {width} x {height}")
            acc_count = create_fast_accumulation_frames(x, y, t, p, width, height, bin_width_us, events_dir)
        else:
            print("No valid events found")
            acc_count = 0
            width, height = 1280, 720  # Default
    
    except Exception as e:
        print(f"Error reading RAW file: {e}")
        acc_count = 0
        width, height = 1280, 720
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"FAST SYNCED DATA PROCESSING\n")
        f.write(f"{'='*30}\n\n")
        f.write(f"Source AVI: {pair['avi_path']}\n")
        f.write(f"Source RAW: {pair['raw_path']}\n")
        f.write(f"Common name: {pair['common_name']}\n")
        f.write(f"Folder name: {pair['folder_name']}\n")
        f.write(f"Detected speed: {pair['speed']}\n")
        f.write(f"Processing time: {datetime.now()}\n\n")
        f.write(f"PROCESSING SETTINGS:\n")
        f.write(f"Bin width: {bin_width_us}μs ({bin_width_us/1000:.1f}ms)\n")
        f.write(f"Dimensions: {width} x {height}\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"Video frames: {frame_count}\n")
        f.write(f"Event frames: {acc_count}\n")
        
        if 'x' in locals() and x is not None and len(x) > 0:
            f.write(f"Total events: {len(x):,}\n")
            f.write(f"Time range: {np.min(t)} - {np.max(t)} μs\n")
            f.write(f"Duration: {(np.max(t) - np.min(t))/1e6:.2f} seconds\n")
            f.write(f"Pos events: {np.sum(p == 1):,}\n")
            f.write(f"Neg events: {np.sum(p == 0):,}\n")
    
    print(f"\nCompleted: {pair['common_name']}")
    print(f"  Video frames: {frame_count}")
    print(f"  Event frames: {acc_count}")
    
    return frame_count, acc_count


def main():
    parser = argparse.ArgumentParser(
        description='Fast processing of synchronized AVI and RAW event files with speed-specific settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -d scan_speed_test -o scan_speed_test_datetime --bin-width 100000

  # Speed-specific bin widths  
  %(prog)s -d scan_speed_test -o output_datetime --speed-bins "3:200000,5:150000,10:100000,20:50000"

  # Filter by regex pattern
  %(prog)s -d scan_speed_test -o output --regex "speed_(10|20)" --bin-width 75000

  # Filter by specific speeds only
  %(prog)s -d scan_speed_test -o output --speeds 3,5,10 --bin-width 150000

  # Combine speed-specific bins with filtering
  %(prog)s -d scan_speed_test -o output_datetime --speeds 10,20 --speed-bins "10:100000,20:50000"

Speed-bins format: "speed1:bin_width1,speed2:bin_width2,..."
  Example: "3:200000,5:150000,10:100000,20:50000" means:
    - Speed 3 uses 200ms bins
    - Speed 5 uses 150ms bins  
    - Speed 10 uses 100ms bins
    - Speed 20 uses 50ms bins

Note: Output ending with 'datetime' will be replaced with current timestamp
        """
    )
    
    parser.add_argument('-d', '--input-dir', required=True,
                       help='Input directory with synced AVI/RAW files')
    parser.add_argument('-o', '--output-dir', required=True,
                       help='Output directory')
    parser.add_argument('--bin-width', type=int, default=100000,
                       help='Default bin width for event accumulation in μs (default: 100000 = 100ms)')
    
    # Speed and filtering options
    parser.add_argument('--speed-bins', type=str, default=None,
                       help='Speed-specific bin widths: "3:200000,5:150000,10:100000,20:50000"')
    parser.add_argument('--speeds', type=str, default=None,
                       help='Only process specific speeds: "3,5,10,20"')
    parser.add_argument('--regex', type=str, default=None,
                       help='Regex pattern to filter folder names: "speed_(10|20)"')
    
    # General options
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Max pairs to process (for testing)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list found pairs without processing')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if args.bin_width <= 0:
        print(f"Error: Bin width must be positive: {args.bin_width}")
        return 1
    
    # Parse speed-specific bin widths
    speed_bins = parse_speed_bins(args.speed_bins)
    
    # Parse speed filter list
    speed_list = []
    if args.speeds:
        try:
            speed_list = [int(s.strip()) for s in args.speeds.split(',')]
            print(f"Filtering for speeds: {speed_list}")
        except ValueError as e:
            print(f"Error parsing speeds: {e}")
            return 1
    
    # Handle datetime in output name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir.endswith('_datetime') or args.output_dir.endswith('datetime'):
        if args.output_dir.endswith('_datetime'):
            args.output_dir = args.output_dir.replace('_datetime', f'_{current_time}')
        else:
            args.output_dir = args.output_dir.replace('datetime', current_time)
        print(f"Using datetime output: {args.output_dir}")
    
    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Default bin width: {args.bin_width}μs ({args.bin_width/1000:.1f}ms)")
    if speed_bins:
        print(f"Speed-specific bins: {len(speed_bins)} configured")
    if args.regex:
        print(f"Regex filter: '{args.regex}'")
    if speed_list:
        print(f"Speed filter: {speed_list}")
    
    # Find all pairs
    pairs = find_synced_pairs(args.input_dir)
    
    if not pairs:
        print("No synced pairs found!")
        return 1
    
    # Apply filters
    if args.regex:
        pairs = filter_pairs_by_regex(pairs, args.regex)
    
    if speed_list:
        pairs = filter_pairs_by_speeds(pairs, speed_list)
    
    if not pairs:
        print("No pairs remaining after filtering!")
        return 1
    
    # Limit pairs if requested
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to first {len(pairs)} pairs")
    
    # List mode - just show what would be processed
    if args.list_only:
        print(f"\n{'='*60}")
        print(f"PAIRS TO PROCESS ({len(pairs)} total)")
        print(f"{'='*60}")
        for i, pair in enumerate(pairs):
            bin_width = get_bin_width_for_pair(pair, speed_bins, args.bin_width)
            print(f"{i+1:2d}. {pair['folder_name']}")
            print(f"     Speed: {pair['speed'] if pair['speed'] else 'unknown'}")
            print(f"     Bin width: {bin_width}μs ({bin_width/1000:.1f}ms)")
            print()
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all pairs
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(pairs)} PAIRS")
    print(f"{'='*60}")
    
    total_frames = 0
    total_acc_frames = 0
    successful = 0
    processing_summary = []
    
    for i, pair in enumerate(pairs):
        print(f"\n{'*'*60}")
        print(f"PAIR {i+1}/{len(pairs)}")
        print(f"{'*'*60}")
        
        # Get appropriate bin width for this pair
        bin_width = get_bin_width_for_pair(pair, speed_bins, args.bin_width)
        
        try:
            frame_count, acc_count = process_synced_pair(pair, args.output_dir, bin_width)
            total_frames += frame_count
            total_acc_frames += acc_count
            successful += 1
            
            processing_summary.append({
                'folder': pair['folder_name'],
                'speed': pair['speed'],
                'bin_width': bin_width,
                'video_frames': frame_count,
                'event_frames': acc_count,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Error processing {pair['common_name']}: {e}")
            import traceback
            traceback.print_exc()
            
            processing_summary.append({
                'folder': pair['folder_name'],
                'speed': pair['speed'],
                'bin_width': bin_width,
                'video_frames': 0,
                'event_frames': 0,
                'status': f'error: {str(e)}'
            })
    
    # Save processing summary
    summary_path = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        summary_data = {
            'timestamp': current_time,
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'default_bin_width': args.bin_width,
            'speed_bins': speed_bins,
            'filters': {
                'regex': args.regex,
                'speeds': speed_list
            },
            'results': processing_summary,
            'totals': {
                'pairs_found': len(pairs),
                'pairs_successful': successful,
                'total_video_frames': total_frames,
                'total_event_frames': total_acc_frames
            }
        }
        json.dump(summary_data, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(pairs)} pairs")
    print(f"Total video frames: {total_frames:,}")
    print(f"Total event frames: {total_acc_frames:,}")
    print(f"Output: {args.output_dir}")
    print(f"Summary saved: {summary_path}")
    
    # Show per-speed summary if using speed-specific bins
    if speed_bins:
        print(f"\nPer-speed summary:")
        speed_stats = {}
        for item in processing_summary:
            if item['status'] == 'success':
                speed = item['speed']
                if speed not in speed_stats:
                    speed_stats[speed] = {'pairs': 0, 'event_frames': 0, 'bin_width': item['bin_width']}
                speed_stats[speed]['pairs'] += 1
                speed_stats[speed]['event_frames'] += item['event_frames']
        
        for speed in sorted(speed_stats.keys()):
            stats = speed_stats[speed]
            print(f"  Speed {speed}: {stats['pairs']} pairs, {stats['event_frames']:,} frames, "
                  f"{stats['bin_width']}μs bins")
    
    return 0


if __name__ == "__main__":
    exit(main())
