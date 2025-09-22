#!/usr/bin/env python3
"""
Robust autocorrelation analysis to find scanning period using proven method
Enhanced with event segmentation into forward/backward scans
Improved with adaptive period estimation and iterative refinement
Added configurable activity fraction and simplified peak finding
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from simple_raw_reader import read_raw_simple


def events_to_activity_signal(t, time_bin_us=1000):
    """
    Convert event timestamps to 1D activity signal
    """
    print(f"Converting events to 1D activity signal with {time_bin_us}μs bins...")
    
    # Get time range
    t_min, t_max = t.min(), t.max()
    total_duration = t_max - t_min
    n_bins = int(total_duration / time_bin_us) + 1
    
    print(f"Time range: {t_min} - {t_max} μs ({total_duration/1e6:.2f} seconds)")
    print(f"Creating {n_bins} time bins")
    
    # Create 1D activity signal (event counts per bin)
    activity = np.zeros(n_bins)
    
    # Bin events
    bin_indices = ((t - t_min) / time_bin_us).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Count events in each bin
    np.add.at(activity, bin_indices, 1)
    
    print(f"Activity signal shape: {activity.shape}")
    print(f"Non-zero bins: {np.count_nonzero(activity)}")
    print(f"Max events per bin: {activity.max()}")
    print(f"Mean events per bin: {activity.mean():.1f}")
    
    return activity, t_min, time_bin_us


def find_smallest_window_with_target_events(activity, target_fraction=0.80):
    """
    Find the smallest possible window containing target_fraction of all events
    Returns start_idx, end_idx, and estimated period
    """
    print(f"Finding smallest window containing {target_fraction*100}% of events...")
    
    total_events = np.sum(activity)
    target_events = target_fraction * total_events
    
    print(f"Total events: {total_events:,}")
    print(f"Target events ({target_fraction*100}%): {target_events:,.0f}")
    
    # Use sliding window approach to find all possible windows
    # that contain at least target_events, then pick the smallest
    smallest_window_size = len(activity)
    best_start, best_end = 0, len(activity)
    
    # Calculate cumulative sum for efficient window sum calculation
    cumsum = np.cumsum(activity)
    
    # Try all possible windows starting from smallest reasonable size
    min_window = max(50, len(activity) // 50)  # At least 50 bins or 2% of total
    
    print(f"Trying window sizes from {min_window} to {len(activity)}...")
    
    for window_size in range(min_window, len(activity)):
        for start_idx in range(len(activity) - window_size + 1):
            end_idx = start_idx + window_size
            
            # Calculate window sum using cumulative sum
            if start_idx == 0:
                window_events = cumsum[end_idx - 1]
            else:
                window_events = cumsum[end_idx - 1] - cumsum[start_idx - 1]
            
            if window_events >= target_events:
                if window_size < smallest_window_size:
                    smallest_window_size = window_size
                    best_start = start_idx
                    best_end = end_idx
                    print(f"New smallest window found: size {window_size}, "
                          f"start {start_idx}, end {end_idx}, events {window_events:,.0f}")
                # Break this start position since we found a valid window
                break
        
        # If we found a window of this size, we can stop since we want the smallest
        if smallest_window_size <= window_size:
            break
    
    actual_events = np.sum(activity[best_start:best_end])
    actual_fraction = actual_events / total_events
    
    print(f"Smallest window: bins {best_start} to {best_end} (size: {smallest_window_size})")
    print(f"Events in window: {actual_events:,} ({actual_fraction*100:.1f}%)")
    
    # Estimate initial round period as window_size / 3 (3 round trips expected)
    estimated_round_period = smallest_window_size // 3
    
    print(f"Estimated initial round period: {estimated_round_period} bins")
    
    return best_start, best_end, estimated_round_period


def calculate_autocorrelation(signal):
    """
    Calculate autocorrelation using numpy correlate
    """
    print("Calculating autocorrelation...")
    
    # Normalize signal
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    # Calculate full autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    
    # Normalize by zero-lag value
    zero_lag_idx = len(autocorr) // 2
    if autocorr[zero_lag_idx] > 0:
        autocorr = autocorr / autocorr[zero_lag_idx]
    
    return autocorr


def calculate_reverse_correlation(signal):
    """
    Calculate correlation between original and reversed signal
    """
    print("Calculating reverse correlation...")
    
    # Normalize signal
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    # Reverse the signal
    reversed_signal = signal[::-1]
    
    # Calculate cross-correlation
    reverse_corr = np.correlate(signal, reversed_signal, mode='full')
    
    # Normalize by maximum absolute value
    max_abs = np.max(np.abs(reverse_corr))
    if max_abs > 0:
        reverse_corr = reverse_corr / max_abs
    
    return reverse_corr


def find_peak_iteratively(correlation, start_idx, end_idx, peak_name=""):
    """
    Iteratively find the best peak by checking if second half of remaining range has larger peaks
    """
    current_peak_idx = start_idx + np.argmax(correlation[start_idx:end_idx])
    current_peak_value = correlation[current_peak_idx]
    iteration = 0
    max_iterations = 10
    
    print(f"Initial {peak_name} peak at index {current_peak_idx}, value {current_peak_value:.4f}")
    
    while iteration < max_iterations:
        # Calculate remaining range from current peak to end
        remaining_start = current_peak_idx
        remaining_end = end_idx
        remaining_length = remaining_end - remaining_start
        
        if remaining_length <= 2:  # Not enough space to split
            break
            
        # Check second half of remaining range
        second_half_start = remaining_start + remaining_length // 2
        second_half_end = remaining_end
        
        if second_half_start >= second_half_end:
            break
            
        # Find largest peak in second half
        second_half_peak_idx = second_half_start + np.argmax(correlation[second_half_start:second_half_end])
        second_half_peak_value = correlation[second_half_peak_idx]
        
        print(f"  Iteration {iteration}: Checking second half [{second_half_start}:{second_half_end}]")
        print(f"  Second half peak at index {second_half_peak_idx}, value {second_half_peak_value:.4f}")
        
        # If second half peak is larger, update current peak
        if second_half_peak_value > current_peak_value:
            print(f"  Found larger peak! Updating from {current_peak_value:.4f} to {second_half_peak_value:.4f}")
            current_peak_idx = second_half_peak_idx
            current_peak_value = second_half_peak_value
        else:
            print(f"  No larger peak found, converged")
            break
            
        iteration += 1
    
    print(f"Final {peak_name} peak at index {current_peak_idx}, value {current_peak_value:.4f}")
    return current_peak_idx, current_peak_value


def find_three_largest_autocorr_peaks_adaptive(autocorr, estimated_period):
    """
    Find the three largest autocorrelation peaks with adaptive minimum distance
    Uses estimated period to set appropriate minimum distance and search range
    Enhanced with iterative peak refinement
    """
    print("Finding three largest autocorrelation peaks with adaptive algorithm and iterative refinement...")
    
    center_idx = len(autocorr) // 2
    center_peak = center_idx
    
    # Set minimum distance based on estimated period
    # Use 50% of estimated period as minimum distance, but ensure reasonable bounds
    min_distance = max(200, int(0.6 * estimated_period))
    
    print(f"Estimated period: {estimated_period} bins")
    print(f"Adaptive minimum distance from center: {min_distance} bins")
    
    # Define search range based on estimated period
    # Look for peaks within reasonable range around estimated period
    search_range = int(2 * estimated_period)
    search_range = min(search_range, center_idx)  # Don't exceed signal bounds
    
    print(f"Search range: ±{search_range} bins")
    
    # Split into left and right halves (excluding center region)
    left_start = max(0, center_idx - search_range)
    left_end = center_idx - min_distance
    right_start = center_idx + min_distance
    right_end = min(len(autocorr), center_idx + search_range)
    
    print(f"Left search region: {left_start} to {left_end} ({left_end - left_start} bins)")
    print(f"Right search region: {right_start} to {right_end} ({right_end - right_start} bins)")
    
    # Find highest peak on left side with iterative refinement
    if left_end > left_start:
        left_peak_idx, left_value = find_peak_iteratively(autocorr, left_start, left_end, "left")
        left_distance = center_idx - left_peak_idx
        print(f"Left peak at lag: -{left_distance}, value: {left_value:.4f}")
    else:
        print("No valid left search region!")
        return [], 0, center_idx
    
    # Find highest peak on right side with iterative refinement
    if right_end > right_start:
        right_peak_idx, right_value = find_peak_iteratively(autocorr, right_start, right_end, "right")
        right_distance = right_peak_idx - center_idx
        print(f"Right peak at lag: +{right_distance}, value: {right_value:.4f}")
    else:
        print("No valid right search region!")
        return [], 0, center_idx
    
    # Alternative strategy: find the largest peak beyond 25% of estimated period
    print("\nAlternative peak search beyond 25% of estimated period...")
    alt_min_distance = max(min_distance, int(0.25 * estimated_period))
    
    # Left side alternative with iterative refinement
    left_alt_end = center_idx - alt_min_distance
    if left_alt_end > 0:
        left_alt_peak_idx, left_alt_value = find_peak_iteratively(autocorr, 0, left_alt_end, "left alternative")
        left_alt_distance = center_idx - left_alt_peak_idx
        print(f"Left alternative peak at lag: -{left_alt_distance}, value: {left_alt_value:.4f}")
        
        # Use alternative if it's significantly better
        if left_alt_value > left_value * 1.1:  # 10% better
            left_peak_idx = left_alt_peak_idx
            left_distance = left_alt_distance
            left_value = left_alt_value
            print(f"Using left alternative peak")
    
    # Right side alternative with iterative refinement
    right_alt_start = center_idx + alt_min_distance
    if right_alt_start < len(autocorr):
        right_alt_peak_idx, right_alt_value = find_peak_iteratively(autocorr, right_alt_start, len(autocorr), "right alternative")
        right_alt_distance = right_alt_peak_idx - center_idx
        print(f"Right alternative peak at lag: +{right_alt_distance}, value: {right_alt_value:.4f}")
        
        # Use alternative if it's significantly better
        if right_alt_value > right_value * 1.1:  # 10% better
            right_peak_idx = right_alt_peak_idx
            right_distance = right_alt_distance
            right_value = right_alt_value
            print(f"Using right alternative peak")
    
    # Calculate round-trip period = max(left_distance, right_distance)
    round_trip_period = max(left_distance, right_distance)
    one_way_period = round_trip_period // 2
    
    print(f"\nFinal peaks found:")
    print(f"Left peak at lag: -{left_distance}")
    print(f"Right peak at lag: +{right_distance}")
    print(f"Peak values: ({left_value:.4f}, {right_value:.4f})")
    print(f"Round-trip period: {round_trip_period} bins (max of distances)")
    print(f"One-way period: {one_way_period} bins")
    
    # Cross-validation
    if left_distance > 0 and right_distance > 0:
        distance_ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
        value_ratio = min(left_value, right_value) / max(left_value, right_value)
        
        print(f"Distance symmetry ratio: {distance_ratio:.3f}")
        print(f"Value symmetry ratio: {value_ratio:.3f}")
        
        if distance_ratio < 0.7:
            print("Warning: Poor distance symmetry")
        if value_ratio < 0.3:
            print("Warning: Large value difference")
    
    peaks = [left_peak_idx, center_peak, right_peak_idx]
    
    return peaks, round_trip_period, center_idx


def find_global_reverse_correlation_peak(reverse_corr):
    """
    Find the global maximum peak in reverse correlation (anti-correlation)
    Simply find the overall top peak without any conditions
    """
    print("Finding global maximum peak in reverse correlation...")
    
    center_idx = len(reverse_corr) // 2
    
    # Find global maximum (by absolute value to handle negative correlations)
    abs_corr = np.abs(reverse_corr)
    global_peak_idx = np.argmax(abs_corr)
    global_peak_value = reverse_corr[global_peak_idx]
    global_peak_lag = global_peak_idx - center_idx
    
    print(f"Global peak at index {global_peak_idx}, lag {global_peak_lag}, value {global_peak_value:.4f}")
    
    return global_peak_lag, global_peak_value


def calculate_prelude_aftermath(full_length, round_trip_period, reverse_peak_lag):
    """
    Calculate prelude and aftermath using the equations:
    prelude - aftermath = reverse_peak_lag
    prelude + aftermath + 3*round_trip_period = full_length
    
    Solving:
    prelude = (full_length - 3*round_trip_period + reverse_peak_lag) / 2
    aftermath = (full_length - 3*round_trip_period - reverse_peak_lag) / 2
    """
    print("Calculating prelude and aftermath...")
    
    # Expected main scanning length (3 round trips = 6 one-way scans)
    main_scanning_length = 3 * round_trip_period
    
    # Solve the system of equations
    prelude = (full_length - main_scanning_length + reverse_peak_lag) / 2
    aftermath = (full_length - main_scanning_length - reverse_peak_lag) / 2
    
    # Ensure non-negative values
    prelude = max(0, int(prelude))
    aftermath = max(0, int(aftermath))
    
    # Adjust if needed to maintain total length
    adjusted_main = full_length - prelude - aftermath
    
    print(f"Full length: {full_length} bins")
    print(f"Expected main scanning: {main_scanning_length} bins")
    print(f"Reverse peak lag: {reverse_peak_lag}")
    print(f"Calculated prelude: {prelude} bins")
    print(f"Calculated aftermath: {aftermath} bins")
    print(f"Adjusted main scanning: {adjusted_main} bins")
    
    return prelude, aftermath, adjusted_main


def analyze_scanning_pattern_single(activity, activity_fraction=0.80, initial_estimates=None):
    """
    Single iteration of scanning analysis
    """
    print(f"\nAnalyzing scanning pattern (activity fraction: {activity_fraction*100}%)...")
    
    if initial_estimates is None:
        # Step 1: Find smallest window containing target fraction of events
        active_start, active_end, estimated_period = find_smallest_window_with_target_events(
            activity, activity_fraction
        )
    else:
        # Use provided estimates
        active_start, active_end, estimated_period = initial_estimates
        print(f"Using provided estimates: start={active_start}, end={active_end}, period={estimated_period}")
    
    # Calculate autocorrelation
    autocorr = calculate_autocorrelation(activity)
    
    # Find three largest peaks in autocorrelation using adaptive method with iterative refinement
    autocorr_peaks, round_trip_period, center_idx = find_three_largest_autocorr_peaks_adaptive(
        autocorr, estimated_period
    )
    
    if round_trip_period <= 0:
        print("Could not determine round-trip period!")
        return None
    
    # Calculate reverse correlation  
    reverse_corr = calculate_reverse_correlation(activity)
    
    # Find global maximum peak in reverse correlation
    reverse_peak_lag, reverse_peak_value = find_global_reverse_correlation_peak(reverse_corr)
    
    # Calculate prelude and aftermath
    full_length = len(activity)
    prelude, aftermath, main_length = calculate_prelude_aftermath(
        full_length, round_trip_period, reverse_peak_lag
    )
    
    # One-way period is half of round-trip
    one_way_period = round_trip_period // 2
    n_cycles = 6  # 6 one-way scans = 3 round trips
    
    # Calculate segment boundaries
    scan_start = prelude
    scan_end = prelude + main_length
    
    # Create refined estimates for next iteration
    refined_estimates = (scan_start, scan_end, round_trip_period)
    
    return {
        'autocorr': autocorr,
        'reverse_corr': reverse_corr,
        'autocorr_peaks': autocorr_peaks,
        'center_idx': center_idx,
        'activity_fraction': activity_fraction,
        'estimated_period': estimated_period,
        'round_trip_period': round_trip_period,
        'one_way_period': one_way_period,
        'reverse_peak_lag': reverse_peak_lag,
        'reverse_peak_value': reverse_peak_value,
        'prelude': prelude,
        'aftermath': aftermath,
        'main_length': main_length,
        'scan_start': scan_start,
        'scan_end': scan_end,
        'n_cycles': n_cycles,
        'full_length': full_length,
        'active_start': active_start,
        'active_end': active_end,
        'refined_estimates': refined_estimates
    }


def analyze_scanning_pattern(activity, activity_fraction=0.80, max_iterations=2):
    """
    Complete scanning analysis with iterative refinement
    Uses results from first iteration as initial values for second iteration
    """
    print("\n" + "="*60)
    print("SCANNING PATTERN ANALYSIS WITH ITERATIVE REFINEMENT")
    print("="*60)
    
    results = None
    
    for iteration in range(max_iterations):
        print(f"\n" + "-"*40)
        print(f"ITERATION {iteration + 1}")
        print("-"*40)
        
        if iteration == 0:
            # First iteration: use initial window detection
            results = analyze_scanning_pattern_single(activity, activity_fraction, None)
        else:
            # Subsequent iterations: use refined estimates from previous iteration
            if results is not None and 'refined_estimates' in results:
                refined_estimates = results['refined_estimates']
                print(f"Using refined estimates from iteration {iteration}: {refined_estimates}")
                results = analyze_scanning_pattern_single(activity, activity_fraction, refined_estimates)
            else:
                print("No valid results from previous iteration, stopping")
                break
        
        if results is None:
            print(f"Iteration {iteration + 1} failed!")
            break
        
        # Print iteration results
        print(f"\nIteration {iteration + 1} Results:")
        print(f"Round-trip period: {results['round_trip_period']} bins")
        print(f"One-way period: {results['one_way_period']} bins")
        print(f"Reverse peak lag: {results['reverse_peak_lag']} bins")
        print(f"Scan boundaries: {results['scan_start']} to {results['scan_end']}")
        print(f"Prelude: {results['prelude']}, Main: {results['main_length']}, Aftermath: {results['aftermath']}")
    
    if results is not None:
        print(f"\nFinal Results (after {max_iterations} iterations):")
        print(f"Activity fraction used: {results['activity_fraction']*100}%")
        print(f"Round-trip period: {results['round_trip_period']} bins")
        print(f"One-way period: {results['one_way_period']} bins")
        print(f"Reverse peak lag: {results['reverse_peak_lag']} bins")
        print(f"Prelude: {results['prelude']} bins")
        print(f"Main scanning: {results['main_length']} bins")
        print(f"Aftermath: {results['aftermath']} bins")
        print(f"Expected cycles: {results['n_cycles']}")
        print(f"Scan boundaries: {results['scan_start']} to {results['scan_end']}")
    
    return results


def segment_events_into_scans(x, y, t, p, results, time_bin_us, t_min):
    """
    Segment events into 6 scanning periods (forward/backward)
    """
    print("\n" + "="*60)
    print("SEGMENTING EVENTS INTO FORWARD/BACKWARD SCANS")
    print("="*60)
    
    if results is None:
        print("No scanning results available for segmentation!")
        return None
    
    # Calculate absolute time boundaries for each scan
    scan_segments = []
    scan_labels = []
    
    # Convert bin indices to absolute timestamps
    scan_start_time = t_min + results['scan_start'] * time_bin_us
    one_way_period_us = results['one_way_period'] * time_bin_us
    
    print(f"Scan start time: {scan_start_time} μs")
    print(f"One-way period: {one_way_period_us} μs ({one_way_period_us/1000:.1f} ms)")
    
    # Define the 6 scan periods
    for i in range(6):
        # Calculate time boundaries
        start_time = scan_start_time + i * one_way_period_us
        end_time = scan_start_time + (i + 1) * one_way_period_us
        
        # Make sure we don't exceed the scanning end boundary
        scan_end_time = t_min + results['scan_end'] * time_bin_us
        end_time = min(end_time, scan_end_time)
        
        # Determine scan direction (even indices = forward, odd = backward)
        direction = "Forward" if i % 2 == 0 else "Backward"
        
        # Find events in this time window
        mask = (t >= start_time) & (t < end_time)
        event_count = np.sum(mask)
        
        # Extract events for this scan
        scan_events = {
            'x': x[mask],
            'y': y[mask], 
            't': t[mask],
            'p': p[mask],
            'start_time': start_time,
            'end_time': end_time,
            'duration_us': end_time - start_time,
            'event_count': event_count,
            'direction': direction,
            'scan_id': i + 1
        }
        
        scan_segments.append(scan_events)
        scan_labels.append(f"Scan_{i+1}_{direction}")
        
        print(f"Scan {i+1} ({direction}): {start_time} - {end_time} μs, "
              f"{event_count:,} events, {(end_time-start_time)/1000:.1f} ms")
    
    return scan_segments, scan_labels


def save_segmented_events(scan_segments, scan_labels, output_dir, base_name):
    """
    Save segmented events to individual files
    """
    print("\n" + "="*60)
    print("SAVING SEGMENTED EVENTS")
    print("="*60)
    
    # Create subdirectory for segmented files
    segments_dir = os.path.join(output_dir, f"{base_name}_segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    segment_files = []
    
    for i, (segment, label) in enumerate(zip(scan_segments, scan_labels)):
        # Save as numpy arrays
        filename = f"{label}_events.npz"
        filepath = os.path.join(segments_dir, filename)
        
        np.savez_compressed(filepath,
                          x=segment['x'],
                          y=segment['y'],
                          t=segment['t'],
                          p=segment['p'],
                          start_time=segment['start_time'],
                          end_time=segment['end_time'],
                          duration_us=segment['duration_us'],
                          event_count=segment['event_count'],
                          direction=segment['direction'],
                          scan_id=segment['scan_id'])
        
        segment_files.append(filepath)
        print(f"Saved {label}: {segment['event_count']:,} events to {filename}")
    
    # Save summary
    summary_path = os.path.join(segments_dir, "segmentation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EVENT SEGMENTATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        total_events = sum(seg['event_count'] for seg in scan_segments)
        f.write(f"Total segmented events: {total_events:,}\n")
        f.write(f"Number of segments: {len(scan_segments)}\n\n")
        
        f.write("SEGMENT DETAILS:\n")
        f.write("-" * 50 + "\n")
        
        for i, segment in enumerate(scan_segments):
            f.write(f"Scan {segment['scan_id']} ({segment['direction']}):\n")
            f.write(f"  Time range: {segment['start_time']} - {segment['end_time']} μs\n")
            f.write(f"  Duration: {segment['duration_us']/1000:.1f} ms\n")
            f.write(f"  Event count: {segment['event_count']:,}\n")
            f.write(f"  Events/ms: {segment['event_count']/(segment['duration_us']/1000):.0f}\n")
            f.write(f"  File: {scan_labels[i]}_events.npz\n\n")
    
    print(f"Segmentation summary saved to: {summary_path}")
    return segment_files, segments_dir


def plot_segmented_scans(scan_segments, scan_labels, output_dir, base_name):
    """
    Plot the segmented scans showing spatial patterns
    """
    print("Creating segmented scan visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Define colors for forward/backward
    colors = {'Forward': 'blue', 'Backward': 'red'}
    
    for i, (segment, label) in enumerate(zip(scan_segments, scan_labels)):
        ax = axes[i]
        
        if segment['event_count'] > 0:
            # Subsample if too many events for plotting
            max_plot_events = 10000
            if segment['event_count'] > max_plot_events:
                indices = np.random.choice(segment['event_count'], max_plot_events, replace=False)
                x_plot = segment['x'][indices]
                y_plot = segment['y'][indices]
                p_plot = segment['p'][indices]
            else:
                x_plot = segment['x']
                y_plot = segment['y']
                p_plot = segment['p']
            
            # Plot events colored by polarity
            pos_mask = p_plot == 1
            neg_mask = p_plot == 0
            
            if np.any(pos_mask):
                ax.scatter(x_plot[pos_mask], y_plot[pos_mask], 
                          c='red', s=0.1, alpha=0.6, label='Positive')
            if np.any(neg_mask):
                ax.scatter(x_plot[neg_mask], y_plot[neg_mask], 
                          c='blue', s=0.1, alpha=0.6, label='Negative')
        
        direction_color = colors.get(segment['direction'], 'black')
        ax.set_title(f"{label}\n{segment['event_count']:,} events, "
                    f"{segment['duration_us']/1000:.1f} ms", 
                    color=direction_color, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        
        if i == 0:  # Add legend to first subplot
            ax.legend(loc='upper right', markerscale=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_segmented_scans.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Segmented scans plot saved to: {plot_path}")


def plot_results(activity, results, output_dir, base_name, time_bin_us, t_min):
    """
    Plot comprehensive results including estimated period analysis
    """
    if results is None:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    
    # Convert to time
    time_axis = np.arange(len(activity)) * time_bin_us / 1000 + t_min / 1000
    
    # Plot 1: Activity with boundaries and most active region
    axes[0].plot(time_axis, activity, 'b-', alpha=0.8, linewidth=0.8)
    
    # Mark boundaries
    scan_start_time = (results['scan_start'] * time_bin_us + t_min) / 1000
    scan_end_time = (results['scan_end'] * time_bin_us + t_min) / 1000
    
    axes[0].axvline(x=scan_start_time, color='red', linestyle='--', linewidth=2, label='Scan boundaries')
    axes[0].axvline(x=scan_end_time, color='red', linestyle='--', linewidth=2)
    
    # Mark smallest active window
    active_start_time = (results['active_start'] * time_bin_us + t_min) / 1000
    active_end_time = (results['active_end'] * time_bin_us + t_min) / 1000
    
    axes[0].axvspan(active_start_time, active_end_time, alpha=0.1, color='green', 
                   label=f'{results["activity_fraction"]*100:.0f}% smallest window')
    
    # Shade regions
    prelude_end_time = scan_start_time
    aftermath_start_time = scan_end_time
    
    axes[0].axvspan(time_axis[0], prelude_end_time, alpha=0.2, color='orange', label='Prelude')
    axes[0].axvspan(aftermath_start_time, time_axis[-1], alpha=0.2, color='gray', label='Aftermath')
    axes[0].axvspan(prelude_end_time, aftermath_start_time, alpha=0.2, color='lightblue', label='Main scanning')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Events per bin')
    axes[0].set_title(f'Event Activity with Scanning Boundaries (Est. period: {results["estimated_period"]} bins)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Main scanning region with cycle divisions
    if results['scan_start'] < results['scan_end'] and results['one_way_period'] > 0:
        main_time = time_axis[results['scan_start']:results['scan_end']]
        main_activity = activity[results['scan_start']:results['scan_end']]
        
        axes[1].plot(main_time, main_activity, 'b-', alpha=0.8)
        
        # Mark one-way scan boundaries and label them
        scan_colors = ['blue', 'red']  # Forward, Backward
        scan_labels = ['Forward', 'Backward']
        
        for i in range(6):  # 6 one-way scans
            cycle_pos = results['scan_start'] + i * results['one_way_period']
            if cycle_pos < results['scan_end']:
                cycle_time = (cycle_pos * time_bin_us + t_min) / 1000
                color_idx = i % 2
                color = scan_colors[color_idx]
                direction = scan_labels[color_idx]
                
                axes[1].axvline(x=cycle_time, color=color, linestyle=':', alpha=0.8, linewidth=2)
                
                # Add scan number and direction labels
                if i < 5:  # Don't label the last boundary
                    next_cycle_pos = results['scan_start'] + (i + 1) * results['one_way_period']
                    next_cycle_time = (min(next_cycle_pos, results['scan_end']) * time_bin_us + t_min) / 1000
                    mid_time = (cycle_time + next_cycle_time) / 2
                    max_val = np.max(main_activity)
                    
                    axes[1].text(mid_time, max_val * 0.9, f'Scan {i+1}\n{direction}', 
                               ha='center', va='top', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Events per bin')
        axes[1].set_title(f'Main Scanning Region (6 one-way scans, period = {results["one_way_period"]} bins)')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Smallest active window analysis
    axes[2].plot(time_axis, activity, 'b-', alpha=0.5, linewidth=0.8, label='Full signal')
    axes[2].plot(time_axis[results['active_start']:results['active_end']], 
                activity[results['active_start']:results['active_end']], 
                'g-', linewidth=2, label=f'{results["activity_fraction"]*100:.0f}% smallest window')
    
    axes[2].axvline(x=active_start_time, color='green', linestyle='--', linewidth=2)
    axes[2].axvline(x=active_end_time, color='green', linestyle='--', linewidth=2)
    
    window_size = results['active_end'] - results['active_start']
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Events per bin')
    axes[2].set_title(f'Smallest Window Detection (Size: {window_size} bins, Est. period: {results["estimated_period"]} bins)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Autocorrelation with three largest peaks
    n = len(activity)
    autocorr_lags = np.arange(-n + 1, n) * time_bin_us / 1000
    axes[3].plot(autocorr_lags, results['autocorr'], 'g-', alpha=0.8)
    
    # Mark the three largest peaks
    if len(results['autocorr_peaks']) >= 3:
        peak_labels = ['Left Peak', 'Center Peak', 'Right Peak']
        colors = ['blue', 'red', 'blue']
        for i, peak_idx in enumerate(results['autocorr_peaks']):
            if peak_idx < len(autocorr_lags):
                peak_time = autocorr_lags[peak_idx]
                peak_value = results['autocorr'][peak_idx]
                axes[3].scatter([peak_time], [peak_value], color=colors[i], s=100, zorder=5)
                axes[3].annotate(peak_labels[i], (peak_time, peak_value), 
                               xytext=(10, 10), textcoords='offset points')
    
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_xlabel('Lag (ms)')
    axes[3].set_ylabel('Autocorrelation')
    axes[3].set_title(f'Autocorrelation with Iterative Peak Finding (Round-trip period: {results["round_trip_period"]} bins)')
    axes[3].grid(True, alpha=0.3)
    
    # Limit view
    max_lag_ms = results['round_trip_period'] * 2 * time_bin_us / 1000
    axes[3].set_xlim(-max_lag_ms, max_lag_ms)
    
    # Plot 5: Reverse correlation with largest peak
    reverse_lags = np.arange(-n + 1, n) * time_bin_us / 1000
    axes[4].plot(reverse_lags, results['reverse_corr'], 'purple', alpha=0.8)
    
    # Mark the largest peak
    largest_peak_idx = results['center_idx'] + results['reverse_peak_lag']
    if 0 <= largest_peak_idx < len(reverse_lags):
        peak_time = reverse_lags[largest_peak_idx]
        peak_value = results['reverse_peak_value']
        axes[4].scatter([peak_time], [peak_value], color='red', s=100, zorder=5, label='Global Peak')
        axes[4].annotate(f'Lag: {results["reverse_peak_lag"]}', (peak_time, peak_value), 
                        xytext=(10, 10), textcoords='offset points')
    
    axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[4].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[4].set_xlabel('Lag (ms)')
    axes[4].set_ylabel('Reverse correlation')
    axes[4].set_title('Reverse Correlation (Global Maximum Peak)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim(-max_lag_ms, max_lag_ms)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_scanning_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plot saved to {plot_path}")


def save_results(results, time_bin_us, t_min, output_dir, base_name):
    """
    Save detailed results including iterative refinement info
    """
    if results is None:
        print("No results to save")
        return
        
    results_path = os.path.join(output_dir, f"{base_name}_scanning_results.txt")
    
    with open(results_path, 'w') as f:
        f.write("ENHANCED SCANNING PATTERN ANALYSIS RESULTS\n")
        f.write("WITH ITERATIVE REFINEMENT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Time bin size: {time_bin_us} μs\n")
        f.write(f"Recording start: {t_min} μs\n\n")
        
        f.write("ENHANCED ANALYSIS:\n")
        window_size = results['active_end'] - results['active_start']
        f.write(f"Activity fraction: {results['activity_fraction']*100:.0f}%\n")
        f.write(f"Smallest window: bins {results['active_start']} to {results['active_end']} (size: {window_size})\n")
        f.write(f"Estimated initial period: {results['estimated_period']} bins ({results['estimated_period']*time_bin_us/1000:.1f} ms)\n\n")
        
        f.write("PERIODS:\n")
        f.write(f"Round-trip period: {results['round_trip_period']} bins ({results['round_trip_period']*time_bin_us/1000:.1f} ms)\n")
        f.write(f"One-way period: {results['one_way_period']} bins ({results['one_way_period']*time_bin_us/1000:.1f} ms)\n\n")
        
        f.write("CORRELATION ANALYSIS:\n")
        f.write(f"Reverse peak lag (global max): {results['reverse_peak_lag']} bins\n")
        f.write(f"Reverse peak value: {results['reverse_peak_value']:.4f}\n\n")
        
        f.write("SEGMENTS (bins):\n")
        f.write(f"Prelude: {results['prelude']} bins\n")
        f.write(f"Main scanning: {results['main_length']} bins\n")
        f.write(f"Aftermath: {results['aftermath']} bins\n")
        f.write(f"Expected cycles: {results['n_cycles']}\n\n")
        
        f.write("SEGMENTS (time):\n")
        prelude_ms = results['prelude'] * time_bin_us / 1000
        main_ms = results['main_length'] * time_bin_us / 1000
        aftermath_ms = results['aftermath'] * time_bin_us / 1000
        
        f.write(f"Prelude: {prelude_ms:.1f} ms ({prelude_ms/1000:.3f} s)\n")
        f.write(f"Main scanning: {main_ms:.1f} ms ({main_ms/1000:.3f} s)\n")
        f.write(f"Aftermath: {aftermath_ms:.1f} ms ({aftermath_ms/1000:.3f} s)\n\n")
        
        f.write("ABSOLUTE TIMESTAMPS:\n")
        start_abs = t_min + results['scan_start'] * time_bin_us
        end_abs = t_min + results['scan_end'] * time_bin_us
        f.write(f"Scan start: {start_abs} μs ({start_abs/1e6:.3f} s)\n")
        f.write(f"Scan end: {end_abs} μs ({end_abs/1e6:.3f} s)\n\n")
        
        f.write("CYCLE TIMING:\n")
        for i in range(results['n_cycles']):
            cycle_start = results['scan_start'] + i * results['one_way_period']
            cycle_end = results['scan_start'] + (i + 1) * results['one_way_period']
            if cycle_end <= results['scan_end']:
                start_time = t_min + cycle_start * time_bin_us
                end_time = t_min + cycle_end * time_bin_us
                direction = "Forward" if i % 2 == 0 else "Backward"
                f.write(f"Cycle {i+1} ({direction}): {start_time} - {end_time} μs\n")
        
        f.write("\nVERIFICATION:\n")
        f.write(f"prelude - aftermath = {results['prelude'] - results['aftermath']} (should equal reverse_peak_lag = {results['reverse_peak_lag']})\n")
        f.write(f"prelude + aftermath + 3*period = {results['prelude'] + results['aftermath'] + 3*results['round_trip_period']} (should equal full_length = {results['full_length']})\n")
        
        f.write(f"\nPERIOD COMPARISON:\n")
        f.write(f"Estimated period: {results['estimated_period']} bins\n")
        f.write(f"Final round-trip period: {results['round_trip_period']} bins\n")
        f.write(f"Ratio (final/estimated): {results['round_trip_period']/results['estimated_period']:.3f}\n")
        
        f.write(f"\nENHANCEMENTS USED:\n")
        f.write(f"- Smallest window detection for {results['activity_fraction']*100:.0f}% of events\n")
        f.write(f"- Iterative peak refinement with second-half checking\n")
        f.write(f"- Global maximum reverse correlation peak finding\n")
        f.write(f"- Iterative refinement (2 iterations)\n")
        f.write(f"- Window size: {window_size} bins (vs full signal: {results['full_length']} bins)\n")
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Activity fraction: {results['activity_fraction']*100:.0f}%")
    print(f"Smallest window size: {window_size} bins")
    print(f"Estimated period: {results['estimated_period']*time_bin_us/1000:.1f} ms")
    print(f"Final round-trip period: {results['round_trip_period']*time_bin_us/1000:.1f} ms")
    print(f"One-way period: {results['one_way_period']*time_bin_us/1000:.1f} ms") 
    print(f"Total scanning time: {main_ms:.1f} ms")
    print(f"Number of one-way scans: {results['n_cycles']}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced scanning analysis with iterative refinement')
    parser.add_argument('raw_file', help='Path to RAW event file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--time_bin_us', type=int, default=1000, help='Time bin size in microseconds')
    parser.add_argument('--max_events', type=int, default=None, help='Maximum events to load')
    parser.add_argument('--segment_events', action='store_true', help='Segment events into forward/backward scans')
    parser.add_argument('--activity_fraction', type=float, default=0.80, 
                       help='Fraction of events to include in active region (default: 0.80)')
    parser.add_argument('--max_iterations', type=int, default=2,
                       help='Maximum number of refinement iterations (default: 2)')
    
    args = parser.parse_args()
    
    # Validate activity_fraction
    if not 0.1 <= args.activity_fraction <= 1.0:
        print(f"Error: activity_fraction must be between 0.1 and 1.0, got {args.activity_fraction}")
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.raw_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.raw_file))[0]
    
    print(f"Analyzing: {args.raw_file}")
    print(f"Activity fraction: {args.activity_fraction*100:.0f}%")
    print(f"Max iterations: {args.max_iterations}")
    
    # Read raw data
    x, y, t, p, width, height = read_raw_simple(args.raw_file)
    
    if x is None:
        print("Failed to read data!")
        return
    
    # Subsample if needed
    if args.max_events and len(x) > args.max_events:
        print(f"Subsampling to {args.max_events} events...")
        indices = np.random.choice(len(x), args.max_events, replace=False)
        x, y, t, p = x[indices], y[indices], t[indices], p[indices]
    
    # Convert to activity signal
    activity, t_min, time_bin_us = events_to_activity_signal(t, args.time_bin_us)
    
    # Analyze pattern with iterative refinement
    results = analyze_scanning_pattern(activity, args.activity_fraction, args.max_iterations)
    
    if results is not None:
        # Plot and save results
        plot_results(activity, results, args.output_dir, base_name, time_bin_us, t_min)
        save_results(results, time_bin_us, t_min, args.output_dir, base_name)
        
        # Segment events if requested
        if args.segment_events:
            scan_segments, scan_labels = segment_events_into_scans(x, y, t, p, results, time_bin_us, t_min)
            
            if scan_segments:
                # Save segmented events
                segment_files, segments_dir = save_segmented_events(scan_segments, scan_labels, 
                                                                  args.output_dir, base_name)
                
                # Plot segmented scans
                plot_segmented_scans(scan_segments, scan_labels, args.output_dir, base_name)
                
                print(f"\nEvent segmentation complete!")
                print(f"Segmented files saved in: {segments_dir}")
                
        print("\nAnalysis complete!")
    else:
        print("Analysis failed!")


if __name__ == "__main__":
    main()