#!/usr/bin/env python3
"""
Complete event visualization with proper multi-window compensation
FIXED: Event time normalization for proper parameter line overlap
MINIMAL CHANGES: Only fix sliding frames and frame means plots with wavelength
"""

import numpy as np
import torch
import torch.nn as nn
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

class Compensate(nn.Module):
    """Multi-window compensation class - FIXED to use actual event duration"""
    def __init__(self, a_params, b_params, duration, num_params=13, temperature=5000, device='cpu'):
        super(Compensate, self).__init__()
        
        self.device = device
        self.duration = float(duration)  # Use actual event duration
        self.temperature = float(temperature)
        self.num_params = num_params
        
        print(f"  Compensate initialized with duration: {self.duration/1000:.1f}ms")
        
        # Convert parameters to PyTorch tensors
        self.a_params = torch.tensor(a_params, dtype=torch.float32, device=device, requires_grad=False)
        self.b_params = torch.tensor(b_params, dtype=torch.float32, device=device, requires_grad=False)
        
        # Calculate window structure using ACTUAL duration
        self.num_boundaries = len(a_params)
        self.num_main_windows = self.num_boundaries - 3
        self.num_total_windows = self.num_main_windows + 2
        self.main_window_size = self.duration / self.num_main_windows
        
        print(f"  Main windows: {self.num_main_windows}, Window size: {self.main_window_size/1000:.1f}ms")
        
        # Fixed boundary offsets using ACTUAL duration
        boundary_offsets = torch.tensor([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.register_buffer('boundary_offsets', boundary_offsets)
        
        if len(a_params) > 0:
            print(f"  Boundary offsets: {boundary_offsets[0].item()/1000:.1f} to {boundary_offsets[-1].item()/1000:.1f}ms")
    
    def forward(self, x, y, t, chunk_size=500000):
        """Apply compensation to events"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        t_min = torch.min(t)
        t_shifted = t - t_min
        
        return self.compute_compensation(x, y, t_shifted, chunk_size=chunk_size)
    
    def compute_compensation(self, x, y, t_shifted, chunk_size=500000):
        """Compute compensation using multi-window approach"""
        num_events = len(x)
        compensations = []
        
        for start_idx in range(0, num_events, chunk_size):
            end_idx = min(start_idx + chunk_size, num_events)
            current_chunk_size = end_idx - start_idx
            
            x_chunk = x[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            t_chunk = t_shifted[start_idx:end_idx]
            
            # Expand parameters for broadcasting
            a_expanded = self.a_params[:, None].expand(-1, current_chunk_size)
            b_expanded = self.b_params[:, None].expand(-1, current_chunk_size)
            offset_expanded = self.boundary_offsets[:, None].expand(-1, current_chunk_size)
            
            x_expanded = x_chunk[None, :].expand(self.num_boundaries, -1)
            y_expanded = y_chunk[None, :].expand(self.num_boundaries, -1)
            
            # Compute boundary values
            boundary_values = a_expanded * x_expanded + b_expanded * y_expanded + offset_expanded
            
            # Compute window memberships
            t_expanded = t_chunk[None, :].expand(self.num_total_windows, -1)
            
            lower_bounds = boundary_values[:-1, :]
            upper_bounds = boundary_values[1:, :]
            
            lower_sigmoids = torch.sigmoid((t_expanded - lower_bounds) / self.temperature)
            upper_sigmoids = torch.sigmoid((upper_bounds - t_expanded) / self.temperature)
            memberships = lower_sigmoids * upper_sigmoids
            
            # Normalize memberships
            memberships_sum = torch.sum(memberships, dim=0, keepdim=True)
            memberships_sum = torch.clamp(memberships_sum, min=1e-8)
            memberships = memberships / memberships_sum
            
            # Compute interpolation weights
            window_widths = upper_bounds - lower_bounds
            window_widths = torch.clamp(window_widths, min=1e-8)
            alpha = (t_expanded - lower_bounds) / window_widths
            
            # Interpolate parameters
            a_lower = self.a_params[:-1, None].expand(-1, current_chunk_size)
            a_upper = self.a_params[1:, None].expand(-1, current_chunk_size)
            b_lower = self.b_params[:-1, None].expand(-1, current_chunk_size)
            b_upper = self.b_params[1:, None].expand(-1, current_chunk_size)
            
            slopes_a = (1 - alpha) * a_lower + alpha * a_upper
            slopes_b = (1 - alpha) * b_lower + alpha * b_upper
            
            # Compute weighted compensation
            x_contrib = memberships * slopes_a * x_chunk[None, :]
            y_contrib = memberships * slopes_b * y_chunk[None, :]
            
            compensation_x = torch.sum(x_contrib, dim=0)
            compensation_y = torch.sum(y_contrib, dim=0)
            compensation_chunk = compensation_x + compensation_y
            
            compensations.append(compensation_chunk)
            
            # Clear memory
            del boundary_values, t_expanded, lower_bounds, upper_bounds
            del lower_sigmoids, upper_sigmoids, memberships, window_widths, alpha
            del a_lower, a_upper, b_lower, b_upper, slopes_a, slopes_b
            del x_contrib, y_contrib, compensation_x, compensation_y
            torch.cuda.empty_cache()
        
        compensation = torch.cat(compensations, dim=0)
        return compensation

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

def create_sliding_frames_with_compensation(compensator, x, y, t, p, bin_width, resolution, output_dir, filename_prefix, wavelength_min=380, wavelength_max=680):
    """Create sliding window frames for both original and compensated data"""
    print(f"Creating sliding frames: bin_width={bin_width/1000:.0f}ms, resolution={resolution/1000:.0f}ms")
    
    t_min = t.min()
    t_max = t.max()
    total_duration = t_max - t_min
    
    # Calculate number of frames
    num_frames = int((total_duration - bin_width) / resolution) + 1
    print(f"  Total duration: {total_duration/1000:.1f}ms, creating {num_frames} frames")
    
    # Convert to tensors and pre-compute compensated timestamps
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    
    print("  Computing compensated timestamps...")
    with torch.no_grad():
        compensation = compensator(xs, ys, ts)
        ts_compensated = ts - compensation
    
    print(f"  Original time range: {ts.min().item():.0f} - {ts.max().item():.0f} μs")
    print(f"  Compensated time range: {ts_compensated.min().item():.0f} - {ts_compensated.max().item():.0f} μs")
    
    frames_orig = []
    frames_comp = []
    frame_times = []
    frame_means_orig = []
    frame_means_comp = []
    frame_counts_orig = []
    frame_counts_comp = []
    
    for i in range(num_frames):
        # Define time window
        t_start = t_min + i * resolution
        t_end = t_start + bin_width
        frame_center_time = (t_start + t_end) / 2
        frame_times.append(frame_center_time)
        
        # ORIGINAL FRAME: Filter by original timestamps
        mask_orig = (ts >= t_start) & (ts < t_end)
        if torch.any(mask_orig):
            frame_orig = create_frame_from_events(xs[mask_orig], ys[mask_orig], ps[mask_orig])
            frame_mean_orig = np.mean(frame_orig)
            frame_count_orig = torch.sum(mask_orig).item()
        else:
            frame_orig = np.zeros((720, 1280))
            frame_mean_orig = 0.0
            frame_count_orig = 0
        
        # COMPENSATED FRAME: Filter by compensated timestamps
        mask_comp = (ts_compensated >= t_start) & (ts_compensated < t_end)
        if torch.any(mask_comp):
            frame_comp = create_frame_from_events(xs[mask_comp], ys[mask_comp], ps[mask_comp])
            frame_mean_comp = np.mean(frame_comp)
            frame_count_comp = torch.sum(mask_comp).item()
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

def create_frame_from_events(x, y, p):
    """Create a single frame from events"""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    
    frame = np.zeros((720, 1280))
    
    x_int = np.clip(np.round(x_np).astype(int), 0, 1279)
    y_int = np.clip(np.round(y_np).astype(int), 0, 719)
    
    for i in range(len(x_int)):
        frame[y_int[i], x_int[i]] += p_np[i]
    
    return frame

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
    parser = argparse.ArgumentParser(description='Complete event visualization with compensation - FIXED')
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
    
    # Create compensator
    print("\n3. Creating compensator...")
    duration = t.max() - t.min()
    compensator = Compensate(
        param_data['a_params'], 
        param_data['b_params'], 
        duration,
        num_params=param_data['num_params'],
        temperature=param_data['temperature'],
        device=device
    )
    
    # Plot events with parameters - FIXED (NO CHANGE)
    print("\n4. Plotting events with parameters (FIXED OVERLAP)...")
    plot_events_with_params(x, y, t, p, param_data, args.sensor_width, args.sensor_height, 
                           args.sample_rate, args.output_dir, base_name)
    
    # Create sliding frames: 50ms bin, 2ms resolution
    print("\n5. Creating sliding frames (50ms bin, 2ms resolution)...")
    create_sliding_frames_with_compensation(compensator, x, y, t, p, 50000, 2000, args.output_dir, base_name,
                                           args.wavelength_min, args.wavelength_max)
    
    # Create fixed frames: 2ms bin, 2ms resolution
    print("\n6. Creating fixed frames (2ms bin, 2ms resolution)...")
    create_sliding_frames_with_compensation(compensator, x, y, t, p, 2000, 2000, args.output_dir, base_name,
                                           args.wavelength_min, args.wavelength_max)
    
    # Plot frame means
    plot_frame_means(args.output_dir, base_name, args.wavelength_min, args.wavelength_max)
    
    print("\n="*80)
    print("FIXED VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print("Generated:")
    print("  - FIXED Event plots with properly overlapped parameter lines")
    print("  - Sliding window frames (original + compensated) with wavelength")
    print("  - Fixed window frames (original + compensated) with wavelength")
    print("  - Frame statistics and comparisons with wavelength")
    print("  - Frame means plots for both bin widths with wavelength")

if __name__ == "__main__":
    main()