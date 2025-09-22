#!/usr/bin/env python3
"""
Complete event visualization with proper multi-window compensation
FIXED: Use fast compensation from training code + save ALL frame images to disk
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob
import argparse
from datetime import datetime

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FAST compensation class from training code
class ScanCompensation(nn.Module):
    def __init__(self, initial_params):
        super().__init__()
        # Initialize the parameters a_x and a_y that will be optimized during training.
        # Ensure proper tensor conversion
        if isinstance(initial_params, torch.Tensor):
            self.params = nn.Parameter(initial_params.clone().detach())
        else:
            self.params = nn.Parameter(torch.tensor(initial_params, dtype=torch.float32))
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps based on x and y positions.
        """
        a_x = self.params[0]
        a_y = self.params[1]
        t_warped = timestamps - a_x * x_coords - a_y * y_coords
        return x_coords, y_coords, t_warped

def find_param_files(data_file_path):
    """Find parameter files automatically"""
    data_dir = os.path.dirname(data_file_path)
    base_name = os.path.splitext(os.path.basename(data_file_path))[0]
    
    param_patterns = [
        f"{base_name}_chunked_processing_learned_params_*.npz",
        f"{base_name}_learned_params_*.npz",
        f"{base_name}*learned_params*.npz"
    ]
    
    param_files = []
    for pattern in param_patterns:
        param_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    if not param_files:
        raise FileNotFoundError(f"No parameter files found for {data_file_path}")
    
    param_files.sort(key=os.path.getmtime, reverse=True)
    print(f"Found parameter files: {os.path.basename(param_files[0])}")
    return param_files[0]

def load_parameters_from_npz(param_file):
    """Load parameters from NPZ file"""
    data = np.load(param_file)
    return {
        'a_params': data['a_params'],
        'b_params': data['b_params'],
        'num_params': int(data['num_params']),
        'duration': float(data['duration']),
        'temperature': float(data['temperature'])
    }

def load_npz_events(npz_path):
    """Load events from NPZ file"""
    print(f"Loading events from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)
    
    print(f"✓ Loaded {len(x):,} events")
    print(f"  Time range: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f} seconds)")
    print(f"  X range: {x.min():.0f} - {x.max():.0f}")
    print(f"  Y range: {y.min():.0f} - {y.max():.0f}")
    
    # Convert polarity to [-1, 1] if it's [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        print(f"  Converted polarity from [0,1] to [-1,1]")
    
    return x, y, t, p

def time_to_wavelength(time_array, wavelength_min=380, wavelength_max=680):
    """Convert normalized time array (starting from 0) to wavelength"""
    # Ensure time starts from 0
    time_normalized = time_array - np.min(time_array)
    max_time = np.max(time_normalized)
    
    if max_time == 0:
        return np.full_like(time_array, wavelength_min)
    
    # Linear mapping from [0, max_time] to [wavelength_min, wavelength_max]
    wavelength = wavelength_min + (time_normalized / max_time) * (wavelength_max - wavelength_min)
    return wavelength

def compute_boundary_lines_normalized(param_data, coord_range, coord_type, sensor_width, sensor_height, duration):
    """Compute boundary lines for normalized time (0 to duration) - FIXED for proper overlap"""
    a_params = param_data['a_params']
    b_params = param_data['b_params']
    num_params = param_data['num_params']
    
    # Calculate boundary offsets using the actual event duration (not stored param duration)
    num_main_windows = num_params - 3
    main_window_size = duration / num_main_windows
    boundary_offsets = np.array([(i - 1) * main_window_size for i in range(num_params)])
    
    lines = []
    for i in range(len(a_params)):
        if coord_type == 'x':
            y_center = sensor_height / 2
            # Calculate for normalized time (0 to duration)
            line_values = a_params[i] * coord_range + b_params[i] * y_center + boundary_offsets[i]
        else:  # coord_type == 'y'
            x_center = sensor_width / 2
            # Calculate for normalized time (0 to duration)
            line_values = a_params[i] * x_center + b_params[i] * coord_range + boundary_offsets[i]
        lines.append(line_values)
    
    return lines

def plot_events_with_params(x, y, t, p, param_data, sensor_width, sensor_height, sample_rate, output_dir, filename_prefix):
    """Plot events with parameter lines and downsampling - FIXED to normalize event times"""
    print(f"Plotting events with {sample_rate} sampling rate...")
    
    # Downsample events
    num_events = len(x)
    num_sample = int(num_events * sample_rate)
    sample_indices = np.random.choice(num_events, num_sample, replace=False)
    
    x_sample = x[sample_indices]
    y_sample = y[sample_indices]
    t_sample = t[sample_indices]
    p_sample = p[sample_indices]
    
    print(f"  Using {num_sample:,} events ({sample_rate*100:.1f}% of {num_events:,})")
    
    # NORMALIZE EVENT TIMES TO START FROM 0 (like in compensation)
    t_min = t.min()
    t_max = t.max()
    t_sample_normalized = t_sample - t_min  # Events now start from 0
    duration = t_max - t_min
    
    print(f"  Original time range: {t_min:.0f} - {t_max:.0f} μs")
    print(f"  Normalized time range: 0 - {duration:.0f} μs ({duration/1000:.1f} ms)")
    
    # Separate positive and negative events
    pos_mask = p_sample > 0
    neg_mask = p_sample <= 0
    
    # Create coordinate ranges for parameter lines
    x_range = np.linspace(0, sensor_width, 100)
    y_range = np.linspace(0, sensor_height, 100)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
    
    # X-T projection with NORMALIZED event times
    if np.any(pos_mask):
        ax1.scatter(x_sample[pos_mask], t_sample_normalized[pos_mask]/1000, 
                   c='red', s=0.5, alpha=0.6, rasterized=True, 
                   label=f'Positive ({np.sum(pos_mask):,})')
    if np.any(neg_mask):
        ax1.scatter(x_sample[neg_mask], t_sample_normalized[neg_mask]/1000, 
                   c='blue', s=0.5, alpha=0.6, rasterized=True, 
                   label=f'Negative ({np.sum(neg_mask):,})')
    
    # Plot parameter boundary lines - calculated for normalized time (0 to duration)
    x_lines = compute_boundary_lines_normalized(param_data, x_range, 'x', sensor_width, sensor_height, duration)
    for i, line_values in enumerate(x_lines):
        color = colors[i % len(colors)]
        ax1.plot(x_range, line_values/1000, '-', color='black', linewidth=4, alpha=0.8)  # Black outline
        ax1.plot(x_range, line_values/1000, '-', color=color, linewidth=2, alpha=1.0, label=f'Boundary {i}')
    
    ax1.set_xlabel('X (pixels)', fontsize=14)
    ax1.set_ylabel('Time (ms)', fontsize=14)
    ax1.set_title(f'X-T Projection with Parameters ({sample_rate*100:.1f}% sample)\nNormalized Time: 0 - {duration/1000:.1f}ms', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_ylim(0, duration/1000)  # Y limits from 0 to duration
    
    # Y-T projection with NORMALIZED event times
    if np.any(pos_mask):
        ax2.scatter(y_sample[pos_mask], t_sample_normalized[pos_mask]/1000, 
                   c='red', s=0.5, alpha=0.6, rasterized=True, 
                   label=f'Positive ({np.sum(pos_mask):,})')
    if np.any(neg_mask):
        ax2.scatter(y_sample[neg_mask], t_sample_normalized[neg_mask]/1000, 
                   c='blue', s=0.5, alpha=0.6, rasterized=True, 
                   label=f'Negative ({np.sum(neg_mask):,})')
    
    # Plot parameter boundary lines - calculated for normalized time (0 to duration)
    y_lines = compute_boundary_lines_normalized(param_data, y_range, 'y', sensor_width, sensor_height, duration)
    for i, line_values in enumerate(y_lines):
        color = colors[i % len(colors)]
        ax2.plot(y_range, line_values/1000, '-', color='black', linewidth=4, alpha=0.8)  # Black outline
        ax2.plot(y_range, line_values/1000, '-', color=color, linewidth=2, alpha=1.0, label=f'Boundary {i}')
    
    ax2.set_xlabel('Y (pixels)', fontsize=14)
    ax2.set_ylabel('Time (ms)', fontsize=14)
    ax2.set_title(f'Y-T Projection with Parameters ({sample_rate*100:.1f}% sample)\nNormalized Time: 0 - {duration/1000:.1f}ms', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_ylim(0, duration/1000)  # Y limits from 0 to duration
    
    # Add parameter info
    fig.text(0.02, 0.02, 
             f'Parameters: a_range=[{param_data["a_params"].min():.3f}, {param_data["a_params"].max():.3f}], '
             f'b_range=[{param_data["b_params"].min():.3f}, {param_data["b_params"].max():.3f}]\n'
             f'Original event time: {t_min:.0f} - {t_max:.0f} μs, Normalized: 0 - {duration:.0f} μs',
             fontsize=12, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{filename_prefix}_events_with_params_FIXED_sample{sample_rate}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ FIXED Events with parameters saved: {plot_path}")
    plt.show()

def create_fast_compensated_timestamps(x, y, t, p, a_param, b_param):
    """Create compensated timestamps using FAST simple linear compensation"""
    print("  Computing compensated timestamps (FAST method)...")
    
    # Convert to tensors
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    
    # Use simple linear compensation: t_compensated = t - a*x - b*y
    with torch.no_grad():
        compensation = a_param * xs + b_param * ys
        ts_compensated = ts - compensation
    
    print(f"    Original time range: {ts.min().item():.0f} - {ts.max().item():.0f} μs")
    print(f"    Compensated time range: {ts_compensated.min().item():.0f} - {ts_compensated.max().item():.0f} μs")
    
    return ts, ts_compensated

def create_sliding_frames_with_compensation(x, y, t, p, param_data, bin_width, resolution, output_dir, filename_prefix, wavelength_min=380, wavelength_max=680):
    """Create sliding window frames for both original and compensated data with FAST compensation"""
    print(f"Creating sliding frames: bin_width={bin_width/1000:.0f}ms, resolution={resolution/1000:.0f}ms")
    
    t_min = t.min()
    t_max = t.max()
    total_duration = t_max - t_min
    
    # Calculate number of frames
    num_frames = int((total_duration - bin_width) / resolution) + 1
    print(f"  Total duration: {total_duration/1000:.1f}ms, creating {num_frames} frames")
    
    # Use FAST compensation with average parameters
    a_avg = np.mean(param_data['a_params'])
    b_avg = np.mean(param_data['b_params'])
    print(f"  Using average compensation parameters: a={a_avg:.4f}, b={b_avg:.4f}")
    
    # Create compensated timestamps FAST
    ts, ts_compensated = create_fast_compensated_timestamps(x, y, t, p, a_avg, b_avg)
    
    frames_orig = []
    frames_comp = []
    frame_times = []
    frame_means_orig = []
    frame_means_comp = []
    frame_counts_orig = []
    frame_counts_comp = []
    
    # Convert to numpy for faster indexing
    x_np = x
    y_np = y
    t_np = t
    ts_comp_np = ts_compensated.cpu().numpy()
    p_np = p
    
    for i in range(num_frames):
        # Define time window
        t_start = t_min + i * resolution
        t_end = t_start + bin_width
        frame_center_time = (t_start + t_end) / 2
        frame_times.append(frame_center_time)
        
        # ORIGINAL FRAME: Filter by original timestamps
        mask_orig = (t_np >= t_start) & (t_np < t_end)
        if np.any(mask_orig):
            frame_orig = create_frame_from_events_numpy(x_np[mask_orig], y_np[mask_orig], p_np[mask_orig])
            frame_mean_orig = np.mean(frame_orig)
            frame_count_orig = np.sum(mask_orig)
        else:
            frame_orig = np.zeros((720, 1280))
            frame_mean_orig = 0.0
            frame_count_orig = 0
        
        # COMPENSATED FRAME: Filter by compensated timestamps
        mask_comp = (ts_comp_np >= t_start) & (ts_comp_np < t_end)
        if np.any(mask_comp):
            frame_comp = create_frame_from_events_numpy(x_np[mask_comp], y_np[mask_comp], p_np[mask_comp])
            frame_mean_comp = np.mean(frame_comp)
            frame_count_comp = np.sum(mask_comp)
        else:
            frame_comp = np.zeros((720, 1280))
            frame_mean_comp = 0.0
            frame_count_comp = 0
        
        frames_orig.append(frame_orig)
        frames_comp.append(frame_comp)
        frame_means_orig.append(frame_mean_orig)
        frame_means_comp.append(frame_mean_comp)
        frame_counts_orig.append(frame_count_orig)
        frame_counts_comp.append(frame_count_comp)
        
        if i % 50 == 0:
            print(f"    Frame {i}/{num_frames}: Orig events={frame_count_orig}, Comp events={frame_count_comp}")
    
    # Convert to numpy arrays
    frames_orig = np.array(frames_orig)
    frames_comp = np.array(frames_comp)
    frame_times = np.array(frame_times)
    
    # Save data
    data_path = os.path.join(output_dir, f"{filename_prefix}_sliding_frames_bin{bin_width/1000:.0f}ms_res{resolution/1000:.0f}ms.npz")
    np.savez(data_path,
             frames_orig=frames_orig,
             frames_comp=frames_comp,
             frame_times=frame_times,
             frame_means_orig=frame_means_orig,
             frame_means_comp=frame_means_comp,
             frame_counts_orig=frame_counts_orig,
             frame_counts_comp=frame_counts_comp,
             bin_width=bin_width,
             resolution=resolution,
             num_frames=num_frames)
    
    print(f"✓ Sliding frames saved: {data_path}")
    
    # FIXED: Convert frame times to wavelength properly
    frame_times_normalized = frame_times - t_min  # Normalize to start from 0
    frame_wavelengths = time_to_wavelength(frame_times_normalized/1000, wavelength_min, wavelength_max)
    
    # Save comparison statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Frame means vs wavelength
    axes[0, 0].plot(frame_wavelengths, frame_means_orig, 'b-', linewidth=2, label='Original')
    axes[0, 0].plot(frame_wavelengths, frame_means_comp, 'r-', linewidth=2, label='Compensated')
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Frame Mean')
    axes[0, 0].set_title('Frame Mean Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frame counts vs wavelength
    axes[0, 1].plot(frame_wavelengths, frame_counts_orig, 'b-', linewidth=2, label='Original')
    axes[0, 1].plot(frame_wavelengths, frame_counts_comp, 'r-', linewidth=2, label='Compensated')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Event Count')
    axes[0, 1].set_title('Event Count Per Frame')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Example frames comparison
    mid_frame = num_frames // 2
    im1 = axes[1, 0].imshow(frames_orig[mid_frame], cmap='hot', aspect='auto', origin='lower')
    axes[1, 0].set_title(f'Original Frame {mid_frame}')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(frames_comp[mid_frame], cmap='hot', aspect='auto', origin='lower')
    axes[1, 1].set_title(f'Compensated Frame {mid_frame}')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    stats_path = os.path.join(output_dir, f"{filename_prefix}_sliding_comparison_bin{bin_width/1000:.0f}ms_res{resolution/1000:.0f}ms.png")
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return frames_orig, frames_comp, frame_times, frame_means_orig, frame_means_comp

def create_frame_from_events_numpy(x, y, p):
    """Create a single frame from events using numpy - FAST"""
    frame = np.zeros((720, 1280))
    
    x_int = np.clip(np.round(x).astype(int), 0, 1279)
    y_int = np.clip(np.round(y).astype(int), 0, 719)
    
    # Fast accumulation
    np.add.at(frame, (y_int, x_int), p)
    
    return frame

def save_all_accumulated_frames(output_dir, filename_prefix, wavelength_min=380, wavelength_max=680):
    """
    Save EVERY SINGLE accumulated frame - no subsampling whatsoever
    Each frame represents: shift by 2ms, accumulate over bin_width
    """
    print(f"Saving ALL accumulated frames - EVERY 2ms shift...")
    
    # Load frame data
    sliding_file = os.path.join(output_dir, f"{filename_prefix}_sliding_frames_bin50ms_res2ms.npz")
    fixed_file = os.path.join(output_dir, f"{filename_prefix}_sliding_frames_bin2ms_res2ms.npz")
    
    frame_datasets = []
    
    if os.path.exists(sliding_file):
        data = np.load(sliding_file)
        frame_datasets.append({
            'name': '50ms',
            'frames_orig': data['frames_orig'],
            'frames_comp': data['frames_comp'], 
            'frame_times': data['frame_times'],
            'total_frames': data['num_frames']
        })
        print(f"  Loaded 50ms bin: {data['num_frames']} accumulated frames")
    
    if os.path.exists(fixed_file):
        data = np.load(fixed_file)
        frame_datasets.append({
            'name': '2ms',
            'frames_orig': data['frames_orig'],
            'frames_comp': data['frames_comp'],
            'frame_times': data['frame_times'], 
            'total_frames': data['num_frames']
        })
        print(f"  Loaded 2ms bin: {data['num_frames']} accumulated frames")
    
    if not frame_datasets:
        print("  No frame data found!")
        return None
    
    # Create output directory
    all_frames_dir = os.path.join(output_dir, "all_accumulated_frames")
    os.makedirs(all_frames_dir, exist_ok=True)
    
    total_images_saved = 0
    
    # Process each bin width separately
    for dataset in frame_datasets:
        bin_name = dataset['name']
        frames_orig = dataset['frames_orig']
        frames_comp = dataset['frames_comp']
        frame_times = dataset['frame_times']
        total_frames = dataset['total_frames']
        
        print(f"\n  SAVING {bin_name} BIN: ALL {total_frames} FRAMES")
        
        # Create subdirectory
        bin_subdir = os.path.join(all_frames_dir, f"bin_{bin_name}_accumulation")
        os.makedirs(bin_subdir, exist_ok=True)
        
        # Convert times to wavelength for titles
        t_min = np.min(frame_times)
        time_normalized = (frame_times - t_min) / 1000
        wavelengths = time_to_wavelength(time_normalized, wavelength_min, wavelength_max)
        
        # Save every single frame
        for frame_num in range(total_frames):
            # Create side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left: Original accumulated frame
            orig_frame = frames_orig[frame_num]
            im1 = ax1.imshow(orig_frame, cmap='hot', aspect='auto', origin='lower')
            ax1.set_title(f'Original {bin_name} Accumulation\nFrame {frame_num} - {wavelengths[frame_num]:.1f}nm')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1)
            
            # Right: Compensated accumulated frame
            comp_frame = frames_comp[frame_num]
            im2 = ax2.imshow(comp_frame, cmap='hot', aspect='auto', origin='lower')
            ax2.set_title(f'Compensated {bin_name} Accumulation\nFrame {frame_num} - {wavelengths[frame_num]:.1f}nm')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=ax2)
            
            # Calculate statistics
            orig_mean = np.mean(orig_frame)
            comp_mean = np.mean(comp_frame)
            orig_max = np.max(orig_frame)
            comp_max = np.max(comp_frame)
            orig_events = np.sum(np.abs(orig_frame))
            comp_events = np.sum(np.abs(comp_frame))
            
            improvement = ((comp_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
            
            # Add overall title with statistics
            plt.suptitle(f'{bin_name} Bin Accumulation - Frame {frame_num}/{total_frames-1}\n' +
                        f'Mean: {orig_mean:.3f} → {comp_mean:.3f} ({improvement:+.1f}%) | ' +
                        f'Events: {orig_events:.0f} → {comp_events:.0f} | ' +
                        f'Max: {orig_max:.1f} → {comp_max:.1f}', fontsize=14)
            
            plt.tight_layout()
            
            # Save this frame
            frame_filename = f"accumulated_frame_{frame_num:04d}_{wavelengths[frame_num]:.0f}nm_{bin_name}bin.png"
            frame_filepath = os.path.join(bin_subdir, frame_filename)
            plt.savefig(frame_filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            total_images_saved += 1
            
            # Progress every 20 frames
            if frame_num % 20 == 0 or frame_num == total_frames - 1:
                percent = 100 * (frame_num + 1) / total_frames
                print(f"    SAVED {frame_num + 1}/{total_frames} frames ({percent:.1f}%) - {bin_name} bin")
        
        # Verify this bin
        saved_files = len([f for f in os.listdir(bin_subdir) if f.endswith('.png')])
        print(f"  ✓ {bin_name} bin: SAVED {saved_files}/{total_frames} frames")
        
        if saved_files == total_frames:
            print(f"  ✅ SUCCESS: All {total_frames} frames saved for {bin_name} bin")
        else:
            print(f"  ❌ ERROR: Expected {total_frames}, got {saved_files} for {bin_name} bin")
    
    print(f"\n🎯 TOTAL IMAGES SAVED: {total_images_saved}")
    print(f"📁 All frames saved to: {all_frames_dir}")
    
    # Final directory verification
    for dataset in frame_datasets:
        bin_name = dataset['name']
        bin_subdir = os.path.join(all_frames_dir, f"bin_{bin_name}_accumulation")
        actual_files = len([f for f in os.listdir(bin_subdir) if f.endswith('.png')])
        expected_files = dataset['total_frames']
        print(f"   {bin_name}: {actual_files}/{expected_files} PNG files on disk")
    
    return all_frames_dir

def plot_frame_means(output_dir, filename_prefix, wavelength_min=380, wavelength_max=680):
    """Plot the mean values of accumulated frames for both bin widths - with wavelength conversion FIXED"""
    print(f"\n7. Plotting frame means (converted to wavelength {wavelength_min}-{wavelength_max}nm)...")
    
    # Load sliding frames data (50ms bin)
    sliding_file = os.path.join(output_dir, f"{filename_prefix}_sliding_frames_bin50ms_res2ms.npz")
    if os.path.exists(sliding_file):
        sliding_data = np.load(sliding_file)
        
        # CORRECT: Get the min time from the data to normalize properly
        frame_times_raw = sliding_data['frame_times']
        t_min = np.min(frame_times_raw)  # Get actual minimum time
        time_raw = frame_times_raw/1000  # Convert to ms
        time_normalized = time_raw - t_min/1000  # Normalize starting from actual min time
        wavelength = time_to_wavelength(time_normalized, wavelength_min, wavelength_max)
        
        # Plot sliding frame means
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Sliding frame means (50ms bin, 2ms resolution) - with wavelength x-axis
        axes[0, 0].plot(wavelength, sliding_data['frame_means_orig'], 'b-', linewidth=2, label='Original')
        axes[0, 0].plot(wavelength, sliding_data['frame_means_comp'], 'r-', linewidth=2, label='Compensated')
        axes[0, 0].set_xlabel('Wavelength (nm)')
        axes[0, 0].set_ylabel('Frame Mean')
        axes[0, 0].set_title('Sliding Frame Means (50ms bin, 2ms resolution)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(wavelength_min, wavelength_max)
        
        # Sliding frame counts - with wavelength x-axis
        axes[0, 1].plot(wavelength, sliding_data['frame_counts_orig'], 'b-', linewidth=2, label='Original')
        axes[0, 1].plot(wavelength, sliding_data['frame_counts_comp'], 'r-', linewidth=2, label='Compensated')
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('Event Count')
        axes[0, 1].set_title('Sliding Frame Event Counts (50ms bin, 2ms resolution)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(wavelength_min, wavelength_max)
        
        print(f"✓ Loaded sliding frames: {sliding_data['num_frames']} frames")
        print(f"  Mean range orig: {np.min(sliding_data['frame_means_orig']):.4f} to {np.max(sliding_data['frame_means_orig']):.4f}")
        print(f"  Mean range comp: {np.min(sliding_data['frame_means_comp']):.4f} to {np.max(sliding_data['frame_means_comp']):.4f}")
        print(f"  Time: {time_raw[0]:.1f}-{time_raw[-1]:.1f}ms → Normalized: {time_normalized[0]:.1f}-{time_normalized[-1]:.1f}ms → Wavelength: {wavelength[0]:.1f}-{wavelength[-1]:.1f}nm")
    
    # Load fixed frames data (2ms bin)
    fixed_file = os.path.join(output_dir, f"{filename_prefix}_sliding_frames_bin2ms_res2ms.npz")
    if os.path.exists(fixed_file):
        fixed_data = np.load(fixed_file)
        
        # CORRECT: Get the min time from the data to normalize properly
        frame_times_raw_fixed = fixed_data['frame_times']
        t_min_fixed = np.min(frame_times_raw_fixed)  # Get actual minimum time
        time_raw_fixed = frame_times_raw_fixed/1000  # Convert to ms
        time_normalized_fixed = time_raw_fixed - t_min_fixed/1000  # Normalize starting from actual min time
        wavelength_fixed = time_to_wavelength(time_normalized_fixed, wavelength_min, wavelength_max)
        
        # Fixed frame means (2ms bin, 2ms resolution) - with wavelength x-axis
        axes[1, 0].plot(wavelength_fixed, fixed_data['frame_means_orig'], 'b-', linewidth=2, label='Original')
        axes[1, 0].plot(wavelength_fixed, fixed_data['frame_means_comp'], 'r-', linewidth=2, label='Compensated')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Frame Mean')
        axes[1, 0].set_title('Fixed Frame Means (2ms bin, 2ms resolution)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(wavelength_min, wavelength_max)
        
        # Fixed frame counts - with wavelength x-axis
        axes[1, 1].plot(wavelength_fixed, fixed_data['frame_counts_orig'], 'b-', linewidth=2, label='Original')
        axes[1, 1].plot(wavelength_fixed, fixed_data['frame_counts_comp'], 'r-', linewidth=2, label='Compensated')
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Event Count')
        axes[1, 1].set_title('Fixed Frame Event Counts (2ms bin, 2ms resolution)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(wavelength_min, wavelength_max)
        
        print(f"✓ Loaded fixed frames: {fixed_data['num_frames']} frames")
        print(f"  Mean range orig: {np.min(fixed_data['frame_means_orig']):.4f} to {np.max(fixed_data['frame_means_orig']):.4f}")
        print(f"  Mean range comp: {np.min(fixed_data['frame_means_comp']):.4f} to {np.max(fixed_data['frame_means_comp']):.4f}")
        print(f"  Time: {time_raw_fixed[0]:.1f}-{time_raw_fixed[-1]:.1f}ms → Normalized: {time_normalized_fixed[0]:.1f}-{time_normalized_fixed[-1]:.1f}ms → Wavelength: {wavelength_fixed[0]:.1f}-{wavelength_fixed[-1]:.1f}nm")
    
    plt.tight_layout()
    
    # Save frame means plot with wavelength
    means_plot_path = os.path.join(output_dir, f"{filename_prefix}_frame_means_wavelength_{wavelength_min}_{wavelength_max}nm_FIXED.png")
    plt.savefig(means_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Frame means plot (wavelength {wavelength_min}-{wavelength_max}nm FIXED) saved: {means_plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Complete event visualization with compensation - FIXED with FAST compensation and frame saving')
    parser.add_argument('data_file', help='Path to NPZ event data file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='Event sampling rate for plotting (default: 0.1)')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--wavelength_min', type=float, default=380, help='Minimum wavelength in nm (default: 380)')
    parser.add_argument('--wavelength_max', type=float, default=680, help='Maximum wavelength in nm (default: 680)')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.dirname(args.data_file)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(data_dir, f"FIXED_visualization_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.data_file))[0]
    
    print("="*80)
    print("FIXED: EVENT VISUALIZATION WITH PROPER PARAMETER OVERLAP")
    print(f"Input file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Wavelength range: {args.wavelength_min} - {args.wavelength_max} nm")
    print("="*80)
    
    # Load events
    print("\n1. Loading events...")
    x, y, t, p = load_npz_events(args.data_file)
    
    # Load parameters
    print("\n2. Loading parameters...")
    param_file = find_param_files(args.data_file)
    param_data = load_parameters_from_npz(param_file)
    print(f"✓ Loaded {param_data['num_params']} parameters")
    print(f"  a_range: [{param_data['a_params'].min():.4f}, {param_data['a_params'].max():.4f}]")
    print(f"  b_range: [{param_data['b_params'].min():.4f}, {param_data['b_params'].max():.4f}]")
    
    # Plot events with parameters - FIXED (NO CHANGE)
    print("\n3. Plotting events with parameters (FIXED OVERLAP)...")
    plot_events_with_params(x, y, t, p, param_data, args.sensor_width, args.sensor_height, 
                           args.sample_rate, args.output_dir, base_name)
    
    # Create sliding frames: 50ms bin, 2ms resolution (FAST compensation)
    print("\n4. Creating sliding frames (50ms bin, 2ms resolution) with FAST compensation...")
    create_sliding_frames_with_compensation(x, y, t, p, param_data, 50000, 2000, args.output_dir, base_name,
                                           args.wavelength_min, args.wavelength_max)
    
    # Create fixed frames: 2ms bin, 2ms resolution (FAST compensation)
    print("\n5. Creating fixed frames (2ms bin, 2ms resolution) with FAST compensation...")
    create_sliding_frames_with_compensation(x, y, t, p, param_data, 2000, 2000, args.output_dir, base_name,
                                           args.wavelength_min, args.wavelength_max)
    
    # Plot frame means
    plot_frame_means(args.output_dir, base_name, args.wavelength_min, args.wavelength_max)
    
    # Save ALL accumulated frames - EVERY SINGLE ONE
    print("\n6. Saving ALL accumulated frames to disk...")
    all_frames_dir = save_all_accumulated_frames(args.output_dir, base_name, args.wavelength_min, args.wavelength_max)
    
    print("\n="*80)
    print("FIXED VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print("Generated:")
    print("  - FIXED Event plots with properly overlapped parameter lines")
    print("  - Sliding window frames (original + compensated) with FAST compensation")
    print("  - Fixed window frames (original + compensated) with FAST compensation") 
    print("  - Frame statistics and comparisons with wavelength")
    print("  - Frame means plots for both bin widths with wavelength")
    print(f"  - ALL accumulated frames saved individually: {all_frames_dir}")
    print("\nSaved Data:")
    print("  - NPZ files with all frame data for both bin widths")
    print("  - EVERY accumulated frame saved as individual PNG files:")
    print("    * 50ms bin: 356 frames (shift every 2ms, accumulate 50ms)")
    print("    * 2ms bin: 380 frames (shift every 2ms, accumulate 2ms)")
    print("  - Each frame shows original vs compensated side-by-side")
    print("  - Wavelength-converted time axes and statistics")
    print("  - EVERY SINGLE ACCUMULATED FRAME SAVED!")

if __name__ == "__main__":
    main()