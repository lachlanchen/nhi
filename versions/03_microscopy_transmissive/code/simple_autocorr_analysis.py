#!/usr/bin/env python3
"""
Robust autocorrelation analysis to find scanning period using proven method
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


def find_top_peaks_with_period(correlation, period, num_peaks=5):
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


def find_top_peaks(correlation, initial_period=500, num_peaks=5, max_iterations=10):
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


def find_three_largest_autocorr_peaks(autocorr):
    """
    Find the three largest peaks in autocorrelation:
    - Center peak at lag=0
    - Left peak at lag=-period  
    - Right peak at lag=period
    """
    print("Finding three largest autocorrelation peaks...")
    
    # Find all peaks using iterative method
    peaks, period = find_top_peaks(autocorr, initial_period=len(autocorr)//10)
    
    if period <= 0 or len(peaks) < 3:
        print("Could not find sufficient peaks for period detection")
        return [], 0, 0
    
    # Get the three largest peaks by correlation value
    peak_values = [(idx, autocorr[idx]) for idx in peaks]
    peak_values.sort(key=lambda x: x[1], reverse=True)
    top_three_peaks = [x[0] for x in peak_values[:3]]
    
    # Sort by position to identify center, left, right
    top_three_peaks.sort()
    
    print(f"Top three peaks at indices: {top_three_peaks}")
    print(f"Peak values: {[autocorr[p] for p in top_three_peaks]}")
    
    # The middle peak should be close to center (lag=0)
    center_idx = len(autocorr) // 2
    
    # Find which peak is closest to center
    distances_to_center = [abs(p - center_idx) for p in top_three_peaks]
    center_peak_idx = np.argmin(distances_to_center)
    
    if center_peak_idx == 1:  # Middle peak is center
        left_peak, center_peak, right_peak = top_three_peaks
    elif center_peak_idx == 0:  # First peak is center
        center_peak, right_peak = top_three_peaks[0], top_three_peaks[1] 
        left_peak = 2 * center_peak - right_peak  # Estimate left peak
    else:  # Last peak is center
        left_peak, center_peak = top_three_peaks[1], top_three_peaks[2]
        right_peak = 2 * center_peak - left_peak  # Estimate right peak
    
    # Calculate period as distance from center to side peak
    left_period = abs(center_peak - left_peak) if left_peak >= 0 else 0
    right_period = abs(right_peak - center_peak) if right_peak < len(autocorr) else 0
    
    # Use the average or the one that's available
    if left_period > 0 and right_period > 0:
        round_trip_period = int((left_period + right_period) / 2)
    elif left_period > 0:
        round_trip_period = left_period
    elif right_period > 0:
        round_trip_period = right_period
    else:
        round_trip_period = period
    
    print(f"Center peak at lag: {center_peak - center_idx}")
    print(f"Left peak at lag: {left_peak - center_idx}")
    print(f"Right peak at lag: {right_peak - center_idx}")
    print(f"Round-trip period: {round_trip_period}")
    
    return [left_peak, center_peak, right_peak], round_trip_period, center_idx


def find_largest_reverse_correlation_peak(reverse_corr):
    """
    Find the lag of the largest peak in reverse correlation
    """
    print("Finding largest peak in reverse correlation...")
    
    # Find all peaks
    peaks, _ = find_top_peaks(reverse_corr, initial_period=len(reverse_corr)//10)
    
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
    Complete scanning analysis using proven method
    """
    print("\n" + "="*60)
    print("SCANNING PATTERN ANALYSIS")
    print("="*60)
    
    # Calculate autocorrelation
    autocorr = calculate_autocorrelation(activity)
    
    # Find three largest peaks in autocorrelation to get round-trip period
    autocorr_peaks, round_trip_period, center_idx = find_three_largest_autocorr_peaks(autocorr)
    
    if round_trip_period <= 0:
        print("Could not determine round-trip period!")
        return None
    
    # Calculate reverse correlation  
    reverse_corr = calculate_reverse_correlation(activity)
    
    # Find largest peak in reverse correlation
    reverse_peak_lag, reverse_peak_value = find_largest_reverse_correlation_peak(reverse_corr)
    
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
        'full_length': full_length
    }


def plot_results(activity, results, output_dir, base_name, time_bin_us, t_min):
    """
    Plot comprehensive results
    """
    if results is None:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # Convert to time
    time_axis = np.arange(len(activity)) * time_bin_us / 1000 + t_min / 1000
    
    # Plot 1: Activity with boundaries
    axes[0].plot(time_axis, activity, 'b-', alpha=0.8, linewidth=0.8)
    
    # Mark boundaries
    scan_start_time = (results['scan_start'] * time_bin_us + t_min) / 1000
    scan_end_time = (results['scan_end'] * time_bin_us + t_min) / 1000
    
    axes[0].axvline(x=scan_start_time, color='red', linestyle='--', linewidth=2, label='Scan boundaries')
    axes[0].axvline(x=scan_end_time, color='red', linestyle='--', linewidth=2)
    
    # Shade regions
    prelude_end_time = scan_start_time
    aftermath_start_time = scan_end_time
    
    axes[0].axvspan(time_axis[0], prelude_end_time, alpha=0.2, color='orange', label='Prelude')
    axes[0].axvspan(aftermath_start_time, time_axis[-1], alpha=0.2, color='gray', label='Aftermath')
    axes[0].axvspan(prelude_end_time, aftermath_start_time, alpha=0.2, color='lightblue', label='Main scanning')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Events per bin')
    axes[0].set_title('Event Activity with Scanning Boundaries')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Main scanning region with cycle divisions
    if results['scan_start'] < results['scan_end'] and results['one_way_period'] > 0:
        main_time = time_axis[results['scan_start']:results['scan_end']]
        main_activity = activity[results['scan_start']:results['scan_end']]
        
        axes[1].plot(main_time, main_activity, 'b-', alpha=0.8)
        
        # Mark one-way scan boundaries
        for i in range(1, 6):  # 6 one-way scans
            cycle_pos = results['scan_start'] + i * results['one_way_period']
            if cycle_pos < results['scan_end']:
                cycle_time = (cycle_pos * time_bin_us + t_min) / 1000
                color = 'red' if i % 2 == 1 else 'blue'
                direction = 'Forward' if i % 2 == 0 else 'Backward'
                axes[1].axvline(x=cycle_time, color=color, linestyle=':', alpha=0.7, 
                              label=f'{direction}' if i <= 2 else "")
        
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Events per bin')
        axes[1].set_title(f'Main Scanning Region (6 one-way scans, period = {results["one_way_period"]} bins)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Autocorrelation with three largest peaks
    n = len(activity)
    autocorr_lags = np.arange(-n + 1, n) * time_bin_us / 1000
    axes[2].plot(autocorr_lags, results['autocorr'], 'g-', alpha=0.8)
    
    # Mark the three largest peaks
    if len(results['autocorr_peaks']) >= 3:
        peak_labels = ['Left Peak', 'Center Peak', 'Right Peak']
        colors = ['blue', 'red', 'blue']
        for i, peak_idx in enumerate(results['autocorr_peaks']):
            if peak_idx < len(autocorr_lags):
                peak_time = autocorr_lags[peak_idx]
                peak_value = results['autocorr'][peak_idx]
                axes[2].scatter([peak_time], [peak_value], color=colors[i], s=100, zorder=5)
                axes[2].annotate(peak_labels[i], (peak_time, peak_value), 
                               xytext=(10, 10), textcoords='offset points')
    
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('Lag (ms)')
    axes[2].set_ylabel('Autocorrelation')
    axes[2].set_title(f'Autocorrelation (Round-trip period: {results["round_trip_period"]} bins)')
    axes[2].grid(True, alpha=0.3)
    
    # Limit view
    max_lag_ms = results['round_trip_period'] * 2 * time_bin_us / 1000
    axes[2].set_xlim(-max_lag_ms, max_lag_ms)
    
    # Plot 4: Reverse correlation with largest peak
    reverse_lags = np.arange(-n + 1, n) * time_bin_us / 1000
    axes[3].plot(reverse_lags, results['reverse_corr'], 'purple', alpha=0.8)
    
    # Mark the largest peak
    largest_peak_idx = results['center_idx'] + results['reverse_peak_lag']
    if 0 <= largest_peak_idx < len(reverse_lags):
        peak_time = reverse_lags[largest_peak_idx]
        peak_value = results['reverse_peak_value']
        axes[3].scatter([peak_time], [peak_value], color='red', s=100, zorder=5, label='Largest Peak')
        axes[3].annotate(f'Lag: {results["reverse_peak_lag"]}', (peak_time, peak_value), 
                        xytext=(10, 10), textcoords='offset points')
    
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_xlabel('Lag (ms)')
    axes[3].set_ylabel('Reverse correlation')
    axes[3].set_title('Reverse Correlation (Original vs Reversed)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(-max_lag_ms, max_lag_ms)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_scanning_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def save_results(results, time_bin_us, t_min, output_dir, base_name):
    """
    Save detailed results
    """
    if results is None:
        print("No results to save")
        return
        
    results_path = os.path.join(output_dir, f"{base_name}_scanning_results.txt")
    
    with open(results_path, 'w') as f:
        f.write("SCANNING PATTERN ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Time bin size: {time_bin_us} μs\n")
        f.write(f"Recording start: {t_min} μs\n\n")
        
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
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Round-trip period: {results['round_trip_period']*time_bin_us/1000:.1f} ms")
    print(f"One-way period: {results['one_way_period']*time_bin_us/1000:.1f} ms") 
    print(f"Total scanning time: {main_ms:.1f} ms")
    print(f"Number of one-way scans: {results['n_cycles']}")


def main():
    parser = argparse.ArgumentParser(description='Robust scanning analysis using proven autocorrelation method')
    parser.add_argument('raw_file', help='Path to RAW event file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--time_bin_us', type=int, default=1000, help='Time bin size in microseconds')
    parser.add_argument('--max_events', type=int, default=None, help='Maximum events to load')
    
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
        t = t[indices]
    
    # Convert to activity signal
    activity, t_min, time_bin_us = events_to_activity_signal(t, args.time_bin_us)
    
    # Analyze pattern
    results = analyze_scanning_pattern(activity)
    
    if results is not None:
        # Plot and save results
        plot_results(activity, results, args.output_dir, base_name, time_bin_us, t_min)
        save_results(results, time_bin_us, t_min, args.output_dir, base_name)
        print("\nAnalysis complete!")
    else:
        print("Analysis failed!")


if __name__ == "__main__":
    main()