#!/usr/bin/env python3
"""
3D block-based autocorrelation analysis to find scanning period using TxHxW data
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from simple_raw_reader import read_raw_simple


def events_to_3d_blocks(x, y, t, p, width, height, time_bin_us=1000, downsample_factor=10):
    """
    Convert events to 3D blocks (T x H x W) using PyTorch index_put with accumulation
    """
    print(f"Converting events to 3D blocks...")
    print(f"Original sensor size: {width} x {height}")
    print(f"Time bin size: {time_bin_us} μs")
    print(f"Downsample factor: {downsample_factor}")
    
    # Calculate downsampled dimensions
    new_width = width // downsample_factor
    new_height = height // downsample_factor
    
    print(f"Downsampled size: {new_width} x {new_height}")
    
    # Get time range and calculate number of bins
    t_min, t_max = t.min(), t.max()
    total_duration = t_max - t_min
    n_time_bins = int(total_duration / time_bin_us) + 1
    
    print(f"Time range: {t_min} - {t_max} μs ({total_duration/1e6:.2f} seconds)")
    print(f"Number of time bins: {n_time_bins}")
    
    # Apply spatial downsampling
    x_down = x // downsample_factor
    y_down = y // downsample_factor
    
    # Clip coordinates to valid range
    x_down = np.clip(x_down, 0, new_width - 1)
    y_down = np.clip(y_down, 0, new_height - 1)
    
    # Calculate time bin indices
    t_rel = t - t_min
    t_bins = (t_rel / time_bin_us).astype(int)
    t_bins = np.clip(t_bins, 0, n_time_bins - 1)
    
    # Convert polarity to +1/-1
    p_signed = p * 2 - 1  # Convert 0,1 to -1,1
    
    # Convert to torch tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create empty tensor
    blocks = torch.zeros((n_time_bins, new_height, new_width), dtype=torch.float32, device=device)
    
    # Convert arrays to tensors
    t_indices = torch.tensor(t_bins, dtype=torch.int64, device=device)
    y_indices = torch.tensor(y_down, dtype=torch.int64, device=device)
    x_indices = torch.tensor(x_down, dtype=torch.int64, device=device)
    polarities = torch.tensor(p_signed, dtype=torch.float32, device=device)
    
    print("Accumulating events into 3D blocks...")
    
    # Use index_put with accumulate=True to sum events at same locations
    blocks.index_put_(
        (t_indices, y_indices, x_indices),
        polarities,
        accumulate=True
    )
    
    print(f"Created 3D blocks tensor with shape: {blocks.shape}")
    print(f"Value range: {blocks.min():.3f} to {blocks.max():.3f}")
    print(f"Non-zero elements: {torch.count_nonzero(blocks).item():,}")
    
    return blocks, t_min, time_bin_us


def calculate_3d_autocorrelation(blocks):
    """
    Calculate autocorrelation on 3D blocks using the same approach as 1D version
    """
    print("Calculating 3D autocorrelation...")
    
    T, H, W = blocks.shape
    
    # First compute spatial average to get 1D signal, then correlate
    # This is much faster than correlating each spatial location separately
    spatial_avg = torch.mean(blocks.view(T, -1), dim=1)  # Shape: (T,)
    
    # Normalize the averaged signal (same as 1D version)
    signal = spatial_avg.cpu().numpy()
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    print(f"Spatial average signal shape: {signal.shape}")
    
    # Calculate autocorrelation using numpy correlate (same as 1D version)
    autocorr = np.correlate(signal, signal, mode='full')
    
    # Normalize by zero-lag value
    zero_lag_idx = len(autocorr) // 2
    if autocorr[zero_lag_idx] > 0:
        autocorr = autocorr / autocorr[zero_lag_idx]
    
    print(f"Autocorrelation computed, shape: {autocorr.shape}")
    return autocorr


def calculate_3d_reverse_correlation(blocks):
    """
    Calculate reverse correlation on 3D blocks using the same approach as 1D version
    """
    print("Calculating 3D reverse correlation...")
    
    T, H, W = blocks.shape
    
    # First compute spatial average to get 1D signal, then correlate
    # This is much faster and should give similar results
    spatial_avg = torch.mean(blocks.view(T, -1), dim=1)  # Shape: (T,)
    
    # Normalize the averaged signal (same as 1D version)
    signal = spatial_avg.cpu().numpy()
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    # Reverse the signal
    reversed_signal = signal[::-1]
    
    print(f"Spatial average signal shape: {signal.shape}")
    
    # Calculate cross-correlation using numpy correlate (same as 1D version)
    reverse_corr = np.correlate(signal, reversed_signal, mode='full')
    
    # Normalize by maximum absolute value
    max_abs = np.max(np.abs(reverse_corr))
    if max_abs > 0:
        reverse_corr = reverse_corr / max_abs
    
    print(f"Reverse correlation computed, shape: {reverse_corr.shape}")
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
    Find the three largest peaks in autocorrelation
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
    
    # Sort by position
    top_three_peaks.sort()
    
    print(f"Top three peaks at indices: {top_three_peaks}")
    print(f"Peak values: {[autocorr[p] for p in top_three_peaks]}")
    
    # Find center peak
    center_idx = len(autocorr) // 2
    distances_to_center = [abs(p - center_idx) for p in top_three_peaks]
    center_peak_idx = np.argmin(distances_to_center)
    
    if center_peak_idx == 1:  # Middle peak is center
        left_peak, center_peak, right_peak = top_three_peaks
    elif center_peak_idx == 0:  # First peak is center
        center_peak, right_peak = top_three_peaks[0], top_three_peaks[1] 
        left_peak = 2 * center_peak - right_peak
    else:  # Last peak is center
        left_peak, center_peak = top_three_peaks[1], top_three_peaks[2]
        right_peak = 2 * center_peak - left_peak
    
    # Calculate period
    left_period = abs(center_peak - left_peak) if left_peak >= 0 else 0
    right_period = abs(right_peak - center_peak) if right_peak < len(autocorr) else 0
    
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


def find_largest_reverse_correlation_peak(reverse_corr, max_reasonable_lag=None):
    """
    Find the lag of the largest peak in reverse correlation with constraints
    """
    print("Finding largest peak in reverse correlation...")
    
    # Set reasonable lag limit (e.g., within 3 periods)
    if max_reasonable_lag is None:
        max_reasonable_lag = len(reverse_corr) // 4  # Within 1/4 of the total length
    
    # Find all peaks
    peaks, _ = find_top_peaks(reverse_corr, initial_period=len(reverse_corr)//10)
    
    if not peaks:
        print("No peaks found in reverse correlation")
        return 0, 0
    
    # Filter peaks by reasonable lag range
    center_idx = len(reverse_corr) // 2
    reasonable_peaks = []
    
    for peak_idx in peaks:
        lag = peak_idx - center_idx
        if abs(lag) <= max_reasonable_lag:
            reasonable_peaks.append((peak_idx, reverse_corr[peak_idx], lag))
    
    if not reasonable_peaks:
        print(f"No peaks found within reasonable lag range (±{max_reasonable_lag})")
        # Fallback: find the largest peak within a smaller range
        search_range = max_reasonable_lag
        start_idx = max(0, center_idx - search_range)
        end_idx = min(len(reverse_corr), center_idx + search_range)
        
        segment = reverse_corr[start_idx:end_idx]
        max_idx = start_idx + np.argmax(segment)
        lag = max_idx - center_idx
        peak_value = reverse_corr[max_idx]
        
        print(f"Fallback: found peak at lag {lag} with value {peak_value:.4f}")
        return lag, peak_value
    
    # Get the largest reasonable peak by value
    reasonable_peaks.sort(key=lambda x: x[1], reverse=True)
    largest_peak_idx, largest_peak_value, lag = reasonable_peaks[0]
    
    print(f"Largest reasonable reverse correlation peak at lag: {lag}")
    print(f"Peak value: {largest_peak_value:.4f}")
    print(f"Rejected {len(peaks) - len(reasonable_peaks)} peaks outside reasonable range")
    
    return lag, largest_peak_value


def calculate_prelude_aftermath(full_length, round_trip_period, reverse_peak_lag):
    """
    Calculate prelude and aftermath using the equations with validation
    """
    print("Calculating prelude and aftermath...")
    
    main_scanning_length = 3 * round_trip_period
    
    prelude = (full_length - main_scanning_length + reverse_peak_lag) / 2
    aftermath = (full_length - main_scanning_length - reverse_peak_lag) / 2
    
    prelude_raw = prelude
    aftermath_raw = aftermath
    
    # Ensure non-negative values
    prelude = max(0, int(prelude))
    aftermath = max(0, int(aftermath))
    
    # Validate results
    if prelude < 0 or aftermath < 0 or (prelude + aftermath + main_scanning_length) > full_length * 1.1:
        print("Warning: Calculated values seem unreasonable, applying fallback method...")
        
        # Fallback: assume symmetric boundaries with minimal prelude/aftermath
        if main_scanning_length <= full_length:
            remaining = full_length - main_scanning_length
            prelude = remaining // 2
            aftermath = remaining - prelude
        else:
            # If main scanning is longer than total, just use the full length
            prelude = 0
            aftermath = 0
            main_scanning_length = full_length
        
        print(f"Fallback method used - Prelude: {prelude}, Aftermath: {aftermath}")
    
    adjusted_main = full_length - prelude - aftermath
    
    print(f"Full length: {full_length} bins")
    print(f"Expected main scanning: {main_scanning_length} bins")
    print(f"Reverse peak lag: {reverse_peak_lag}")
    print(f"Raw calculation - Prelude: {prelude_raw:.1f}, Aftermath: {aftermath_raw:.1f}")
    print(f"Final values - Prelude: {prelude} bins, Aftermath: {aftermath} bins")
    print(f"Adjusted main scanning: {adjusted_main} bins")
    
    return prelude, aftermath, adjusted_main


def analyze_3d_scanning_pattern(blocks, temporal_subsample=5):
    """
    Complete 3D scanning analysis
    
    Args:
        blocks: 3D tensor (T, H, W)
        temporal_subsample: Factor to subsample time dimension for correlation analysis
    """
    print("\n" + "="*60)
    print("3D SCANNING PATTERN ANALYSIS")
    print("="*60)
    
    # Subsample temporally for faster correlation analysis
    if temporal_subsample > 1:
        print(f"Subsampling time dimension by factor {temporal_subsample} for correlation analysis...")
        blocks_sub = blocks[::temporal_subsample]
        print(f"Subsampled blocks shape: {blocks_sub.shape}")
    else:
        blocks_sub = blocks
    
    # Calculate 3D autocorrelation
    autocorr = calculate_3d_autocorrelation(blocks_sub)
    
    # Find three largest peaks in autocorrelation
    autocorr_peaks, round_trip_period, center_idx = find_three_largest_autocorr_peaks(autocorr)
    
    if round_trip_period <= 0:
        print("Could not determine round-trip period!")
        return None
    
    # Scale period back to original time resolution
    if temporal_subsample > 1:
        round_trip_period = round_trip_period * temporal_subsample
        print(f"Scaled round-trip period back to original resolution: {round_trip_period}")
    
    # Calculate 3D reverse correlation  
    reverse_corr = calculate_3d_reverse_correlation(blocks_sub)
    
    # Find largest peak in reverse correlation with reasonable constraints
    # Use 1 period as reasonable lag limit (in subsampled space)
    max_reasonable_lag_sub = round_trip_period if round_trip_period > 0 else len(reverse_corr) // 6
    print(f"Using max reasonable lag: {max_reasonable_lag_sub} (subsampled), period: {round_trip_period}")
    reverse_peak_lag, reverse_peak_value = find_largest_reverse_correlation_peak(reverse_corr, max_reasonable_lag_sub)
    
    # Scale lag back to original time resolution
    if temporal_subsample > 1:
        reverse_peak_lag = reverse_peak_lag * temporal_subsample
        print(f"Scaled reverse peak lag back to original resolution: {reverse_peak_lag}")
    
    # Calculate prelude and aftermath
    full_length = blocks.shape[0]
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
        'full_length': full_length,
        'temporal_subsample': temporal_subsample
    }


def extract_and_save_segments(blocks, results, save_dir, base_name, time_bin_us):
    """
    Extract and save different segments
    """
    print("Extracting and saving segments...")
    
    if results is None:
        print("No results available, saving full data")
        full_path = os.path.join(save_dir, f"{base_name}_full_3d_blocks.npy")
        np.save(full_path, blocks.cpu().numpy())
        print(f"Saved full blocks to: {full_path}")
        return
    
    # Extract values
    prelude = results['prelude']
    main_length = results['main_length']
    aftermath = results['aftermath']
    one_way_period = results['one_way_period']
    
    scan_start = prelude
    scan_end = prelude + main_length
    
    # Save prelude
    if prelude > 0:
        prelude_blocks = blocks[:prelude]
        prelude_path = os.path.join(save_dir, f"{base_name}_prelude_3d_blocks.npy")
        np.save(prelude_path, prelude_blocks.cpu().numpy())
        print(f"Saved prelude ({prelude} bins, {prelude*time_bin_us/1000:.1f}ms) to: {prelude_path}")
    
    # Save aftermath
    if aftermath > 0:
        aftermath_blocks = blocks[scan_end:]
        aftermath_path = os.path.join(save_dir, f"{base_name}_aftermath_3d_blocks.npy")
        np.save(aftermath_path, aftermath_blocks.cpu().numpy())
        print(f"Saved aftermath ({aftermath} bins, {aftermath*time_bin_us/1000:.1f}ms) to: {aftermath_path}")
    
    # Save main scanning region
    main_blocks = blocks[scan_start:scan_end]
    main_path = os.path.join(save_dir, f"{base_name}_main_scanning_3d_blocks.npy")
    np.save(main_path, main_blocks.cpu().numpy())
    print(f"Saved main scanning ({main_length} bins, {main_length*time_bin_us/1000:.1f}ms) to: {main_path}")
    
    # Extract individual scanning cycles
    print("Extracting individual scanning cycles...")
    
    forward_cycles = []
    backward_cycles = []
    
    for i in range(6):  # 6 one-way scans
        cycle_start = scan_start + i * one_way_period
        cycle_end = scan_start + (i + 1) * one_way_period
        
        if cycle_end <= scan_end:
            cycle_blocks = blocks[cycle_start:cycle_end]
            direction = "forward" if i % 2 == 0 else "backward"
            cycle_path = os.path.join(save_dir, f"{base_name}_cycle_{i+1}_{direction}_3d_blocks.npy")
            np.save(cycle_path, cycle_blocks.cpu().numpy())
            
            cycle_duration_ms = one_way_period * time_bin_us / 1000
            print(f"Saved cycle {i+1} ({direction}, {one_way_period} bins, {cycle_duration_ms:.1f}ms) to: {cycle_path}")
            
            if i % 2 == 0:  # Forward
                forward_cycles.append(cycle_blocks)
            else:  # Backward
                backward_cycles.append(cycle_blocks)
    
    # Save stacked cycles
    if forward_cycles:
        forward_stack = torch.stack(forward_cycles, dim=0)
        forward_path = os.path.join(save_dir, f"{base_name}_forward_cycles_3d_stack.npy")
        np.save(forward_path, forward_stack.cpu().numpy())
        print(f"Saved {len(forward_cycles)} forward cycles stack to: {forward_path}")
        
        forward_mean = torch.mean(forward_stack, dim=0)
        forward_mean_path = os.path.join(save_dir, f"{base_name}_forward_mean_3d_blocks.npy")
        np.save(forward_mean_path, forward_mean.cpu().numpy())
        print(f"Saved forward cycles mean to: {forward_mean_path}")
    
    if backward_cycles:
        # Time-reverse backward cycles for comparison
        backward_stack = torch.stack([torch.flip(cycle, dims=[0]) for cycle in backward_cycles], dim=0)
        backward_path = os.path.join(save_dir, f"{base_name}_backward_cycles_3d_stack.npy")
        np.save(backward_path, backward_stack.cpu().numpy())
        print(f"Saved {len(backward_cycles)} backward cycles stack (time-reversed) to: {backward_path}")
        
        backward_mean = torch.mean(backward_stack, dim=0)
        backward_mean_path = os.path.join(save_dir, f"{base_name}_backward_mean_3d_blocks.npy")
        np.save(backward_mean_path, backward_mean.cpu().numpy())
        print(f"Saved backward cycles mean to: {backward_mean_path}")
    
    # Combined mean
    if forward_cycles and backward_cycles:
        combined_mean = (forward_mean + backward_mean) / 2
        combined_path = os.path.join(save_dir, f"{base_name}_combined_mean_3d_blocks.npy")
        np.save(combined_path, combined_mean.cpu().numpy())
        print(f"Saved combined forward+backward mean to: {combined_path}")


def plot_3d_results(blocks, results, output_dir, base_name, time_bin_us):
    """
    Plot results from 3D analysis
    """
    if results is None:
        return
    
    # Create activity signal for plotting
    activity = torch.sum(torch.abs(blocks), dim=(1, 2)).cpu().numpy()
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # Time axis in milliseconds
    time_axis = np.arange(len(activity)) * time_bin_us / 1000
    
    # Plot 1: Activity with boundaries
    axes[0].plot(time_axis, activity, 'b-', alpha=0.8, linewidth=0.8)
    
    scan_start_time = results['scan_start'] * time_bin_us / 1000
    scan_end_time = results['scan_end'] * time_bin_us / 1000
    
    axes[0].axvline(x=scan_start_time, color='red', linestyle='--', linewidth=2, label='Scan boundaries')
    axes[0].axvline(x=scan_end_time, color='red', linestyle='--', linewidth=2)
    
    axes[0].axvspan(0, scan_start_time, alpha=0.2, color='orange', label='Prelude')
    axes[0].axvspan(scan_end_time, time_axis[-1], alpha=0.2, color='gray', label='Aftermath')
    axes[0].axvspan(scan_start_time, scan_end_time, alpha=0.2, color='lightblue', label='Main scanning')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Total Activity')
    axes[0].set_title('3D Block Analysis - Event Activity with Scanning Boundaries')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Main scanning with cycles
    if results['scan_start'] < results['scan_end']:
        main_time = time_axis[results['scan_start']:results['scan_end']]
        main_activity = activity[results['scan_start']:results['scan_end']]
        
        axes[1].plot(main_time, main_activity, 'b-', alpha=0.8)
        
        for i in range(1, 6):
            cycle_pos = results['scan_start'] + i * results['one_way_period']
            if cycle_pos < results['scan_end']:
                cycle_time = cycle_pos * time_bin_us / 1000
                color = 'red' if i % 2 == 1 else 'blue'
                direction = 'Forward' if i % 2 == 0 else 'Backward'
                axes[1].axvline(x=cycle_time, color=color, linestyle=':', alpha=0.7, 
                              label=f'{direction}' if i <= 2 else "")
        
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Total Activity')
        axes[1].set_title(f'Main Scanning Region - 3D Analysis (period = {results["one_way_period"]} bins)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: 3D Autocorrelation
    n_corr = len(results['autocorr'])
    temporal_subsample = results.get('temporal_subsample', 1)
    
    # Create lag array that matches exactly the correlation calculation
    # range(-T + 1, T) where T is the subsampled length
    T_sub = n_corr // 2 + 1  # This should match the original subsampled length
    autocorr_lags = np.arange(-T_sub + 1, T_sub) * time_bin_us * temporal_subsample / 1000
    
    # Ensure arrays have the same length
    if len(autocorr_lags) != len(results['autocorr']):
        # Fallback: create lag array with exact same length
        autocorr_lags = np.arange(-(n_corr//2), n_corr - n_corr//2) * time_bin_us * temporal_subsample / 1000
    
    print(f"Debug: autocorr_lags shape: {autocorr_lags.shape}, autocorr shape: {len(results['autocorr'])}")
    
    axes[2].plot(autocorr_lags, results['autocorr'], 'g-', alpha=0.8)
    
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
    axes[2].set_ylabel('3D Autocorrelation')
    axes[2].set_title(f'3D Autocorrelation (Round-trip period: {results["round_trip_period"]} bins)')
    axes[2].grid(True, alpha=0.3)
    
    max_lag_ms = results['round_trip_period'] * 2 * time_bin_us / 1000
    axes[2].set_xlim(-max_lag_ms, max_lag_ms)
    
    # Plot 4: 3D Reverse correlation
    reverse_lags = autocorr_lags  # Same lag structure
    axes[3].plot(reverse_lags, results['reverse_corr'], 'purple', alpha=0.8)
    
    # For the reverse correlation peak, we need to be more careful about the indexing
    center_lag_idx = len(reverse_lags) // 2
    try:
        largest_peak_idx = center_lag_idx + results['reverse_peak_lag'] // temporal_subsample
        if 0 <= largest_peak_idx < len(reverse_lags):
            peak_time = reverse_lags[largest_peak_idx]
            peak_value = results['reverse_peak_value']
            axes[3].scatter([peak_time], [peak_value], color='red', s=100, zorder=5, label='Largest Peak')
            axes[3].annotate(f'Lag: {results["reverse_peak_lag"]}', (peak_time, peak_value), 
                            xytext=(10, 10), textcoords='offset points')
    except:
        # If indexing fails, just mark the peak value somewhere
        max_idx = np.argmax(np.abs(results['reverse_corr']))
        peak_time = reverse_lags[max_idx]
        peak_value = results['reverse_corr'][max_idx]
        axes[3].scatter([peak_time], [peak_value], color='red', s=100, zorder=5, label='Largest Peak')
    
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_xlabel('Lag (ms)')
    axes[3].set_ylabel('3D Reverse Correlation')
    axes[3].set_title('3D Reverse Correlation (Original vs Time-Reversed)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(-max_lag_ms, max_lag_ms)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{base_name}_3d_scanning_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D analysis plot saved to: {plot_path}")


def save_3d_results(results, blocks, time_bin_us, output_dir, base_name, downsample_factor):
    """
    Save 3D analysis results
    """
    if results is None:
        return
        
    results_path = os.path.join(output_dir, f"{base_name}_3d_scanning_results.txt")
    
    with open(results_path, 'w') as f:
        f.write("3D SCANNING PATTERN ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("TENSOR PROPERTIES:\n")
        f.write(f"3D Blocks shape: {tuple(blocks.shape)}\n")
        f.write(f"Time bin size: {time_bin_us} μs\n")
        f.write(f"Downsample factor: {downsample_factor}\n")
        f.write(f"Spatial resolution: {blocks.shape[2]} x {blocks.shape[1]}\n")
        f.write(f"Temporal subsample for correlation: {results.get('temporal_subsample', 1)}\n\n")
        
        f.write("PERIODS:\n")
        f.write(f"Round-trip period: {results['round_trip_period']} bins ({results['round_trip_period']*time_bin_us/1000:.1f} ms)\n")
        f.write(f"One-way period: {results['one_way_period']} bins ({results['one_way_period']*time_bin_us/1000:.1f} ms)\n\n")
        
        f.write("3D CORRELATION ANALYSIS:\n")
        f.write(f"3D Reverse peak lag: {results['reverse_peak_lag']} bins\n")
        f.write(f"3D Reverse peak value: {results['reverse_peak_value']:.4f}\n\n")
        
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
        
        f.write("VERIFICATION:\n")
        f.write(f"prelude - aftermath = {results['prelude'] - results['aftermath']} (should equal reverse_peak_lag = {results['reverse_peak_lag']})\n")
        f.write(f"prelude + aftermath + 3*period = {results['prelude'] + results['aftermath'] + 3*results['round_trip_period']} (should equal full_length = {results['full_length']})\n")
    
    print(f"3D analysis results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='3D block-based scanning analysis')
    parser.add_argument('raw_file', help='Path to RAW event file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--time_bin_us', type=int, default=1000, help='Time bin size in microseconds')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Spatial downsampling factor')
    parser.add_argument('--temporal_subsample', type=int, default=5, help='Temporal subsampling for correlation analysis (higher = faster)')
    parser.add_argument('--max_events', type=int, default=None, help='Maximum events to load (for testing)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.raw_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.raw_file))[0]
    
    print(f"Processing: {args.raw_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time bin: {args.time_bin_us}μs, Downsample: {args.downsample_factor}x, Temporal subsample: {args.temporal_subsample}x")
    
    # Read raw data
    print("\nReading raw data...")
    x, y, t, p, width, height = read_raw_simple(args.raw_file)
    
    if x is None:
        print("Failed to read data!")
        return
    
    # Subsample if needed for testing
    if args.max_events and len(x) > args.max_events:
        print(f"Subsampling to {args.max_events} events for testing...")
        indices = np.random.choice(len(x), args.max_events, replace=False)
        indices.sort()
        x, y, t, p = x[indices], y[indices], t[indices], p[indices]
    
    # Convert to 3D blocks
    print(f"\nConverting to 3D blocks...")
    blocks, t_min, time_bin_us = events_to_3d_blocks(
        x, y, t, p, width, height, 
        time_bin_us=args.time_bin_us, 
        downsample_factor=args.downsample_factor
    )
    
    # Analyze 3D scanning pattern
    print(f"\nAnalyzing 3D scanning pattern...")
    results = analyze_3d_scanning_pattern(blocks, temporal_subsample=args.temporal_subsample)
    
    if results is not None:
        # Extract and save segments
        print(f"\nExtracting and saving 3D segments...")
        extract_and_save_segments(blocks, results, args.output_dir, base_name, time_bin_us)
        
        # Plot results
        plot_3d_results(blocks, results, args.output_dir, base_name, time_bin_us)
        
        # Save results
        save_3d_results(results, blocks, time_bin_us, args.output_dir, base_name, args.downsample_factor)
        
        print(f"\n3D Analysis complete!")
        print(f"Round-trip period: {results['round_trip_period']*time_bin_us/1000:.1f} ms")
        print(f"One-way period: {results['one_way_period']*time_bin_us/1000:.1f} ms")
        print(f"Main scanning time: {results['main_length']*time_bin_us/1000:.1f} ms")
    else:
        print("3D Analysis failed!")


if __name__ == "__main__":
    main()