#!/usr/bin/env python3
"""
Robust autocorrelation analysis to find scanning period using proven method
Enhanced with event segmentation into forward/backward scans
Improved with adaptive period estimation and dynamic peak finding
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


def find_most_active_region(activity, target_fraction=0.90):
    """
    Find the region containing target_fraction of all events using sliding window
    Returns start_idx, end_idx, and estimated period
    """
    print(f"Finding most active region containing {target_fraction*100}% of events...")
    
    total_events = np.sum(activity)
    target_events = target_fraction * total_events
    
    print(f"Total events: {total_events:,}")
    print(f"Target events ({target_fraction*100}%): {target_events:,.0f}")
    
    # Try different window sizes to find the shortest one containing target events
    min_window_events = target_events
    best_start, best_end = 0, len(activity)
    best_window_size = len(activity)
    
    # Start with a reasonable minimum window size
    min_window = max(100, len(activity) // 20)  # At least 100 bins or 5% of total
    max_window = len(activity)
    
    for window_size in range(min_window, max_window):
        for start_idx in range(len(activity) - window_size + 1):
            end_idx = start_idx + window_size
            window_events = np.sum(activity[start_idx:end_idx])
            
            if window_events >= target_events and window_size < best_window_size:
                best_start = start_idx
                best_end = end_idx
                best_window_size = window_size
                break
        
        # If we found a good window, we can stop
        if best_window_size < len(activity):
            break
    
    actual_events = np.sum(activity[best_start:best_end])
    actual_fraction = actual_events / total_events
    
    print(f"Best window: bins {best_start} to {best_end} (size: {best_window_size})")
    print(f"Events in window: {actual_events:,} ({actual_fraction*100:.1f}%)")
    
    # Estimate initial round period as window_size / 3 (3 round trips expected)
    estimated_round_period = best_window_size // 3
    
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


def find_top_peaks_with_period(correlation, period, num_peaks=3):
    """
    Find peaks in segments based on known period
    """
    peaks = []
    if period > 0:
        half_period = period // 2
        
        for i in range(0, len(correlation), half_period):
            segment_start = i
            segment_end = min(i + half_period, len(correlation))
            segment = correlation[segment_start:segment_end]
            if len(segment) == 0:
                continue
            peak_index = segment_start + np.argmax(segment)
            peaks.append(peak_index)

    # Keep only top peaks by value
    if len(peaks) > num_peaks:
        peak_values = correlation[peaks]
        top_indices = np.argpartition(peak_values, -num_peaks)[-num_peaks:]
        peaks = [peaks[i] for i in top_indices]
    
    peaks = sorted(peaks)
    return peaks


def find_period(peaks):
    """
    Calculate period from peak differences
    """
    if len(peaks) < 2:
        return 0
    differences = np.diff(peaks)
    return int(np.mean(differences))


def find_top_peaks(correlation, initial_period=200, num_peaks=3, max_iterations=20):
    """
    Iteratively find peaks until period converges
    """
    print("Finding peaks with iterative period refinement...")
    
    period = initial_period
    previous_period = -1
    iteration = 0
    
    while period != previous_period and iteration < max_iterations:
        previous_period = period
        peaks = find_top_peaks_with_period(correlation, period, num_peaks)
        if len(peaks) >= 2:
            period = find_period(peaks)
            print(f"Iteration {iteration}: period = {period}")
        else:
            print(f"Iteration {iteration}: Not enough peaks found")
            break
        iteration += 1
    
    print(f"Converged to period: {period}")
    return peaks, period


def find_three_largest_autocorr_peaks_adaptive(autocorr, estimated_period):
    """
    Find the three largest autocorrelation peaks with adaptive minimum distance
    Uses estimated period to set appropriate minimum distance and search range
    """
    print("Finding three largest autocorrelation peaks with adaptive algorithm...")
    
    center_idx = len(autocorr) // 2
    center_peak = center_idx
    
    # Set minimum distance based on estimated period
    # Use 50% of estimated period as minimum distance, but ensure reasonable bounds
    min_distance = max(50, int(0.5 * estimated_period))
    min_distance = min(min_distance, len(autocorr) // 10)  # Don't exceed 10% of signal
    
    print(f"Estimated period: {estimated_period} bins")
    print(f"Adaptive minimum distance from center: {min_distance} bins")
    
    # Define search range based on estimated period
    # Look for peaks within reasonable range around estimated period
    search_range = int(1.5 * estimated_period)  # 150% of estimated period
    search_range = min(search_range, center_idx)  # Don't exceed signal bounds
    
    print(f"Search range: ±{search_range} bins")
    
    # Split into left and right halves (excluding center region)
    left_start = max(0, center_idx - search_range)
    left_end = center_idx - min_distance
    right_start = center_idx + min_distance
    right_end = min(len(autocorr), center_idx + search_range)
    
    print(f"Left search region: {left_start} to {left_end} ({left_end - left_start} bins)")
    print(f"Right search region: {right_start} to {right_end} ({right_end - right_start} bins)")
    
    # Find highest peak on left side
    if left_end > left_start:
        left_region = autocorr[left_start:left_end]
        left_peak_local_idx = np.argmax(left_region)
        left_peak_idx = left_start + left_peak_local_idx
        left_distance = center_idx - left_peak_idx
        left_value = autocorr[left_peak_idx]
        print(f"Left peak at lag: -{left_distance}, value: {left_value:.4f}")
    else:
        print("No valid left search region!")
        return [], 0, center_idx
    
    # Find highest peak on right side
    if right_end > right_start:
        right_region = autocorr[right_start:right_end]
        right_peak_local_idx = np.argmax(right_region)
        right_peak_idx = right_start + right_peak_local_idx
        right_distance = right_peak_idx - center_idx
        right_value = autocorr[right_peak_idx]
        print(f"Right peak at lag: +{right_distance}, value: {right_value:.4f}")
    else:
        print("No valid right search region!")
        return [], 0, center_idx
    
    # Alternative strategy: find the largest peak beyond 50% of estimated period
    print("\nAlternative peak search beyond 50% of estimated period...")
    alt_min_distance = max(min_distance, int(0.5 * estimated_period))
    
    # Left side alternative
    left_alt_end = center_idx - alt_min_distance
    if left_alt_end > 0:
        left_alt_region = autocorr[:left_alt_end]
        left_alt_peak_idx = np.argmax(left_alt_region)
        left_alt_distance = center_idx - left_alt_peak_idx
        left_alt_value = autocorr[left_alt_peak_idx]
        print(f"Left alternative peak at lag: -{left_alt_distance}, value: {left_alt_value:.4f}")
        
        # Use alternative if it's significantly better
        if left_alt_value > left_value * 1.1:  # 10% better
            left_peak_idx = left_alt_peak_idx
            left_distance = left_alt_distance
            left_value = left_alt_value
            print(f"Using left alternative peak")
    
    # Right side alternative
    right_alt_start = center_idx + alt_min_distance
    if right_alt_start < len(autocorr):
        right_alt_region = autocorr[right_alt_start:]
        right_alt_peak_local_idx = np.argmax(right_alt_region)
        right_alt_peak_idx = right_alt_start + right_alt_peak_local_idx
        right_alt_distance = right_alt_peak_idx - center_idx
        right_alt_value = autocorr[right_alt_peak_idx]
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


def find_largest_reverse_correlation_peak(reverse_corr, estimated_period):
    """
    Find the lag of the largest peak in reverse correlation
    Uses estimated period for better initial guess
    """
    print("Finding largest peak in reverse correlation...")
    
    # Use estimated period for better initial guess
    initial_period = max(estimated_period, 200)
    
    # Find all peaks
    peaks, _ = find_top_peaks(reverse_corr, initial_period=initial_period)
    
    if not peaks:
        print("No peaks found in reverse correlation")
        return 0, 0
    
    # Get the largest peak by value
    peak_values = [(idx, reverse_corr[idx]) for idx in peaks]
    peak_values.sort(key=lambda x: x[1], reverse=True)
    
    largest_peak_idx = peak_values[0][0]
    largest_peak_value = peak_values[0][1]
    
    # Convert to lag (relative to center)
    center_idx = len(reverse_corr) // 2
    lag = largest_peak_idx - center_idx
    
    print(f"Largest reverse correlation peak at lag: {lag}")
    print(f"Peak value: {largest_peak_value:.4f}")
    
    return lag, largest_peak_value


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


def analyze_scanning_pattern(activity):
    """
    Complete scanning analysis using proven method with adaptive peak finding
    """
    print("\n" + "="*60)
    print("SCANNING PATTERN ANALYSIS")
    print("="*60)
    
    # Step 1: Find most active region to estimate initial period
    active_start, active_end, estimated_period = find_most_active_region(activity)
    
    # Calculate autocorrelation
    autocorr = calculate_autocorrelation(activity)
    
    # Find three largest peaks in autocorrelation using adaptive method
    autocorr_peaks, round_trip_period, center_idx = find_three_largest_autocorr_peaks_adaptive(
        autocorr, estimated_period
    )
    
    if round_trip_period <= 0:
        print("Could not determine round-trip period!")
        return None
    
    # Calculate reverse correlation  
    reverse_corr = calculate_reverse_correlation(activity)
    
    # Find largest peak in reverse correlation
    reverse_peak_lag, reverse_peak_value = find_largest_reverse_correlation_peak(
        reverse_corr, estimated_period
    )
    
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
    
    print(f"\nFinal Results:")
    print(f"Estimated initial period: {estimated_period} bins")
    print(f"Round-trip period: {round_trip_period} bins")
    print(f"One-way period: {one_way_period} bins")
    print(f"Reverse peak lag: {reverse_peak_lag} bins")
    print(f"Prelude: {prelude} bins")
    print(f"Main scanning: {main_length} bins")
    print(f"Aftermath: {aftermath} bins")
    print(f"Expected cycles: {n_cycles}")
    print(f"Scan boundaries: {scan_start} to {scan_end}")
    
    return {
        'autocorr': autocorr,
        'reverse_corr': reverse_corr,
        'autocorr_peaks': autocorr_peaks,
        'center_idx': center_idx,
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
        'active_end': active_end
    }


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
    
    # Mark most active region
    active_start_time = (results['active_start'] * time_bin_us + t_min) / 1000
    active_end_time = (results['active_end'] * time_bin_us + t_min) / 1000
    
    axes[0].axvspan(active_start_time, active_end_time, alpha=0.1, color='green', label='90% active region')
    
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
    
    # Plot 3: Most active region analysis
    axes[2].plot(time_axis, activity, 'b-', alpha=0.5, linewidth=0.8, label='Full signal')
    axes[2].plot(time_axis[results['active_start']:results['active_end']], 
                activity[results['active_start']:results['active_end']], 
                'g-', linewidth=2, label='90% active region')
    
    axes[2].axvline(x=active_start_time, color='green', linestyle='--', linewidth=2)
    axes[2].axvline(x=active_end_time, color='green', linestyle='--', linewidth=2)
    
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Events per bin')
    axes[2].set_title(f'Most Active Region Detection (Est. period: {results["estimated_period"]} bins)')
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
    axes[3].set_title(f'Autocorrelation (Round-trip period: {results["round_trip_period"]} bins)')
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
        axes[4].scatter([peak_time], [peak_value], color='red', s=100, zorder=5, label='Largest Peak')
        axes[4].annotate(f'Lag: {results["reverse_peak_lag"]}', (peak_time, peak_value), 
                        xytext=(10, 10), textcoords='offset points')
    
    axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[4].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[4].set_xlabel('Lag (ms)')
    axes[4].set_ylabel('Reverse correlation')
    axes[4].set_title('Reverse Correlation (Original vs Reversed)')
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
    Save detailed results including adaptive analysis info
    """
    if results is None:
        print("No results to save")
        return
        
    results_path = os.path.join(output_dir, f"{base_name}_scanning_results.txt")
    
    with open(results_path, 'w') as f:
        f.write("ADAPTIVE SCANNING PATTERN ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Time bin size: {time_bin_us} μs\n")
        f.write(f"Recording start: {t_min} μs\n\n")
        
        f.write("ADAPTIVE ANALYSIS:\n")
        f.write(f"90% active region: bins {results['active_start']} to {results['active_end']}\n")
        f.write(f"Estimated initial period: {results['estimated_period']} bins ({results['estimated_period']*time_bin_us/1000:.1f} ms)\n\n")
        
        f.write("PERIODS:\n")
        f.write(f"Round-trip period: {results['round_trip_period']} bins ({results['round_trip_period']*time_bin_us/1000:.1f} ms)\n")
        f.write(f"One-way period: {results['one_way_period']} bins ({results['one_way_period']*time_bin_us/1000:.1f} ms)\n\n")
        
        f.write("CORRELATION ANALYSIS:\n")
        f.write(f"Reverse peak lag: {results['reverse_peak_lag']} bins\n")
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
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Estimated period: {results['estimated_period']*time_bin_us/1000:.1f} ms")
    print(f"Final round-trip period: {results['round_trip_period']*time_bin_us/1000:.1f} ms")
    print(f"One-way period: {results['one_way_period']*time_bin_us/1000:.1f} ms") 
    print(f"Total scanning time: {main_ms:.1f} ms")
    print(f"Number of one-way scans: {results['n_cycles']}")


def main():
    parser = argparse.ArgumentParser(description='Adaptive robust scanning analysis with event segmentation')
    parser.add_argument('raw_file', help='Path to RAW event file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--time_bin_us', type=int, default=1000, help='Time bin size in microseconds')
    parser.add_argument('--max_events', type=int, default=None, help='Maximum events to load')
    parser.add_argument('--segment_events', action='store_true', help='Segment events into forward/backward scans')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.raw_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.raw_file))[0]
    
    print(f"Analyzing: {args.raw_file}")
    
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
    
    # Analyze pattern
    results = analyze_scanning_pattern(activity)
    
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