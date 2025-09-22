#!/usr/bin/env python3
"""
Enhanced scan compensation code for NPZ event files with Multi-Window Compensation
Replaces simple linear compensation with our advanced Compensate class
Added support for fixing a_params or b_params during training
Fixed: Skip training when no parameters are trainable
MEMORY FIX: Uses chunked processing throughout the entire pipeline to avoid GPU memory overflow
CRITICAL: Still processes all events for unified variance calculation and proper gradient flow

NEW FEATURES:
- Save learned parameters with parameter length in filename
- Load saved parameters and skip optimization process
- Reuse previously trained parameters
- Multi-format support: NPZ, JSON, and CSV files
- CSV files can be opened in Excel for easy parameter inspection
- Time-binned frame comparison plots showing temporal evolution
- Organized output with time_binned_frames/ subfolder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
import csv
from matplotlib.gridspec import GridSpec

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import our Compensate class (copy from the optimized 3D code)
class Compensate(nn.Module):
    def __init__(self, a_params, b_params, duration, num_params=13, temperature=5000, device='cpu', 
                 a_fixed=True, b_fixed=False, boundary_trainable=False, debug=False):
        """
        3D Multi-window compensation class with PyTorch and gradients
        Added support for fixing parameters during training
        MEMORY FIX: Uses chunked processing throughout pipeline to avoid GPU memory overflow while maintaining proper gradient flow
        """
        super(Compensate, self).__init__()
        
        self.device = device
        self.duration = float(duration)
        self.temperature = float(temperature)
        self.a_fixed = a_fixed
        self.b_fixed = b_fixed
        self.boundary_trainable = boundary_trainable
        self.debug = debug
        self.num_params = num_params
        
        # Validate input
        if len(a_params) != len(b_params):
            raise ValueError("a_params and b_params must have the same length")
        
        if len(a_params) != num_params:
            raise ValueError(f"Parameter arrays must have length {num_params}, got {len(a_params)}")
        
        # Convert parameters to PyTorch tensors with conditional gradient tracking
        if a_fixed:
            self.a_params = torch.tensor(a_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('a_params_buffer', self.a_params)
            if debug:
                print(f"  a_params: FIXED at {a_params[0]:.3f} (no gradients)")
        else:
            self.a_params = nn.Parameter(torch.tensor(a_params, dtype=torch.float32, device=device))
            if debug:
                print(f"  a_params: TRAINABLE (with gradients)")
        
        if b_fixed:
            self.b_params = torch.tensor(b_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('b_params_buffer', self.b_params)
            if debug:
                print(f"  b_params: FIXED at {b_params[0]:.3f} (no gradients)")
        else:
            self.b_params = nn.Parameter(torch.tensor(b_params, dtype=torch.float32, device=device))
            if debug:
                print(f"  b_params: TRAINABLE (with gradients)")
        
        # Calculate window structure
        self.num_boundaries = len(a_params)
        self.num_main_windows = self.num_boundaries - 3  # n_main = len(ax) - 3
        self.num_total_windows = self.num_main_windows + 2  # Add 2 edge windows
        
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")
        
        # Calculate main window size based on main windows only (in microseconds)
        self.main_window_size = self.duration / self.num_main_windows
        
        # Handle boundary offsets - either fixed or trainable
        if boundary_trainable:
            # Initialize raw boundary parameters (can be any value)
            # Default initialization to match fixed offsets when abs() and cumsum() applied
            initial_raw_boundaries = torch.tensor([self.main_window_size] * self.num_boundaries, 
                                                 dtype=torch.float32, device=device)
            initial_raw_boundaries[0] = -self.main_window_size  # First boundary at negative offset
            self.raw_boundary_params = nn.Parameter(initial_raw_boundaries)
            if debug:
                print(f"  boundary_offsets: TRAINABLE (monotonic via cumsum)")
        else:
            # Fixed boundary offsets as before
            boundary_offsets = torch.tensor([
                (i - 1) * self.main_window_size
                for i in range(self.num_boundaries)
            ], dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('fixed_boundary_offsets', boundary_offsets)
            if debug:
                print(f"  boundary_offsets: FIXED (no gradients)")
        
        # Count trainable parameters
        trainable_params = 0
        if not a_fixed:
            trainable_params += len(a_params)
        if not b_fixed:
            trainable_params += len(b_params)
        if boundary_trainable:
            trainable_params += len(a_params)
            
        if debug:
            device_str = "GPU" if self.device == 'cuda:0' else "CPU"
            print(f"  Compensate ready: {self.num_main_windows} main windows, {device_str}, temp={self.temperature:.0f}μs")
            print(f"  Trainable parameters: {trainable_params}/{len(a_params) * (3 if boundary_trainable else 2)}")
    
    @property
    def boundary_offsets(self):
        """
        Get boundary offsets - either fixed or computed from trainable parameters
        For trainable: ensures monotonic increasing via cumulative sum of absolute values
        """
        if self.boundary_trainable:
            # Convert raw parameters to monotonic increasing offsets
            # Use cumsum(abs()) to ensure monotonic increasing
            abs_params = torch.abs(self.raw_boundary_params)
            cumulative_offsets = torch.cumsum(abs_params, dim=0)
            # Adjust to start from first boundary position
            return cumulative_offsets + self.raw_boundary_params[0] - abs_params[0]
        else:
            return self.fixed_boundary_offsets
    
    def get_a_params(self):
        """Get a_params regardless of whether they're fixed or trainable"""
        if self.a_fixed:
            return self.a_params_buffer
        else:
            return self.a_params
    
    def get_b_params(self):
        """Get b_params regardless of whether they're fixed or trainable"""
        if self.b_fixed:
            return self.b_params_buffer
        else:
            return self.b_params
    
    def get_boundary_surfaces(self, x, y):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        """
        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Get current parameters
        a_params = self.get_a_params()
        b_params = self.get_b_params()
        
        if x.dim() == 1 and y.dim() == 1:
            # Create meshgrid
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Vectorized computation: [num_boundaries, len_x, len_y]
            boundary_values = (
                a_params[:, None, None] * X[None, :, :] + 
                b_params[:, None, None] * Y[None, :, :] + 
                self.boundary_offsets[:, None, None]
            )
        else:
            raise ValueError("x and y must be 1D tensors/arrays")
        
        return boundary_values
    
    def forward(self, x, y, t, chunk_size=500000, debug=False):
        """
        Compute compensation for given (x, y, t) coordinates
        CRITICAL: Process all events for proper variance calculation (chunked for memory efficiency)
        """
        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        # Shift time to start from 0
        t_min = torch.min(t)
        t_shifted = t - t_min
        
        # CRITICAL: Process ALL events (chunked for memory efficiency, but still returns complete result)
        return self.compute_event_compensation(x, y, t_shifted, chunk_size=chunk_size, debug=debug)
    
    def compute_event_compensation(self, x, y, t_shifted, chunk_size=500000, debug=False):
        """
        Compute compensation directly for event coordinates using chunked processing
        MEMORY EFFICIENT: Process events in smaller chunks to avoid GPU memory overflow
        CRITICAL: Still returns ALL compensations for proper variance calculation
        """
        # Get current parameters
        a_params = self.get_a_params()
        b_params = self.get_b_params()
        
        num_events = len(x)
        compensations = []
        
        if debug:
            print(f"Computing compensation in chunks of {chunk_size:,} events for memory efficiency...")
        
        for start_idx in range(0, num_events, chunk_size):
            end_idx = min(start_idx + chunk_size, num_events)
            current_chunk_size = end_idx - start_idx
            
            # Get chunk
            x_chunk = x[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            t_chunk = t_shifted[start_idx:end_idx]
            
            # Process this chunk
            # Expand parameters for broadcasting: [num_boundaries, chunk_size]
            a_expanded = a_params[:, None].expand(-1, current_chunk_size)
            b_expanded = b_params[:, None].expand(-1, current_chunk_size)
            offset_expanded = self.boundary_offsets[:, None].expand(-1, current_chunk_size)
            
            # Expand coordinates for broadcasting: [num_boundaries, chunk_size]
            x_expanded = x_chunk[None, :].expand(self.num_boundaries, -1)
            y_expanded = y_chunk[None, :].expand(self.num_boundaries, -1)
            
            # Compute boundary values: [num_boundaries, chunk_size]
            boundary_values = a_expanded * x_expanded + b_expanded * y_expanded + offset_expanded
            
            # Compute window memberships for each event in chunk
            t_expanded = t_chunk[None, :].expand(self.num_total_windows, -1)  # [windows, chunk_size]
            
            # Get boundaries for each window: [num_total_windows, chunk_size]
            lower_bounds = boundary_values[:-1, :]
            upper_bounds = boundary_values[1:, :]
            
            # Compute sigmoid memberships
            lower_sigmoids = torch.sigmoid((t_expanded - lower_bounds) / self.temperature)
            upper_sigmoids = torch.sigmoid((upper_bounds - t_expanded) / self.temperature)
            memberships = lower_sigmoids * upper_sigmoids
            
            # Normalize memberships
            memberships_sum = torch.sum(memberships, dim=0, keepdim=True)
            memberships_sum = torch.clamp(memberships_sum, min=1e-8)
            memberships = memberships / memberships_sum
            
            # Compute interpolation weights for each window
            window_widths = upper_bounds - lower_bounds
            window_widths = torch.clamp(window_widths, min=1e-8)
            alpha = (t_expanded - lower_bounds) / window_widths
            
            # Interpolate parameters for each window
            a_lower = a_params[:-1, None].expand(-1, current_chunk_size)
            a_upper = a_params[1:, None].expand(-1, current_chunk_size)
            b_lower = b_params[:-1, None].expand(-1, current_chunk_size)
            b_upper = b_params[1:, None].expand(-1, current_chunk_size)
            
            slopes_a = (1 - alpha) * a_lower + alpha * a_upper
            slopes_b = (1 - alpha) * b_lower + alpha * b_upper
            
            # Compute weighted compensation for each event in chunk
            x_contrib = memberships * slopes_a * x_chunk[None, :]
            y_contrib = memberships * slopes_b * y_chunk[None, :]
            
            # Sum over windows: [chunk_size]
            compensation_x = torch.sum(x_contrib, dim=0)
            compensation_y = torch.sum(y_contrib, dim=0)
            compensation_chunk = compensation_x + compensation_y
            
            # Store chunk result
            compensations.append(compensation_chunk)
            
            # Clear intermediate tensors to save memory
            del boundary_values, t_expanded, lower_bounds, upper_bounds
            del lower_sigmoids, upper_sigmoids, memberships, window_widths, alpha
            del a_lower, a_upper, b_lower, b_upper, slopes_a, slopes_b
            del x_contrib, y_contrib, compensation_x, compensation_y
            torch.cuda.empty_cache()
            
            # Progress indicator (only in debug mode)
            if debug and (start_idx // chunk_size) % 20 == 0:
                progress = (end_idx / num_events) * 100
                print(f"  Compensation: {end_idx:,}/{num_events:,} events ({progress:.1f}%)")
        
        # CRITICAL: Concatenate all chunks to return complete compensation array
        # This ensures variance calculation sees ALL events together
        compensation = torch.cat(compensations, dim=0)
        
        if debug:
            print(f"✓ Completed compensation calculation for all {num_events:,} events")
        return compensation
    
    def get_boundary_lines_for_plot(self, coord_range, coord_type):
        """
        Compute boundary lines for plotting (uses original grid-based method)
        """
        if coord_type == 'x':
            # X-T view: fix Y at center, vary X
            y_center = 360.0  # Approximate sensor center
            x_range = coord_range
            y_fixed = [y_center]
        else:  # coord_type == 'y'
            # Y-T view: fix X at center, vary Y
            x_center = 640.0  # Approximate sensor center
            y_range = coord_range
            x_fixed = [x_center]
        
        # Use small arrays for plotting
        if coord_type == 'x':
            boundary_surfaces = self.get_boundary_surfaces(x_range, y_fixed)
            # Extract the line at fixed Y: [num_boundaries, len_x]
            lines = boundary_surfaces[:, :, 0].cpu().numpy()
        else:
            boundary_surfaces = self.get_boundary_surfaces(x_fixed, coord_range)
            # Extract the line at fixed X: [num_boundaries, len_y]
            lines = boundary_surfaces[:, 0, :].cpu().numpy()
        
        return lines

def load_npz_events(npz_path, debug=False):
    """
    Load events from NPZ file
    Expected format: x, y, t, p arrays
    """
    if debug:
        print(f"Loading events from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Extract arrays
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)  # Should already be in microseconds
    p = data['p'].astype(np.float32)
    
    if debug:
        print(f"✓ Loaded {len(x):,} events")
        print(f"  Time: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f}s)")
        print(f"  Spatial: X=[{x.min():.0f}, {x.max():.0f}], Y=[{y.min():.0f}, {y.max():.0f}]")
    
    # Convert polarity to [-1, 1] if it's [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        if debug:
            print(f"  Converted polarity from [0,1] to [-1,1]")
    
    return x, y, t, p

def load_and_merge_segments(segments_folder, debug=False):
    """
    Load and merge all scan segments with proper time normalization and polarity handling
    """
    if debug:
        print("\n" + "="*60)
        print("MERGING SCAN SEGMENTS")
        print("="*60)
        print(f"Loading from: {segments_folder}")
    
    # Find all segment files
    segment_files = glob.glob(os.path.join(segments_folder, "Scan_*_events.npz"))
    segment_files.sort()  # Ensure consistent ordering
    
    if not segment_files:
        raise FileNotFoundError(f"No segment files found in {segments_folder}")
    
    if debug:
        print(f"Found {len(segment_files)} segment files")
    
    all_x = []
    all_y = []
    all_t = []
    all_p = []
    
    for segment_file in segment_files:
        filename = os.path.basename(segment_file)
        
        # Load segment data
        data = np.load(segment_file)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        t = data['t'].astype(np.float32)
        p = data['p'].astype(np.float32)
        
        # Get metadata
        start_time = float(data['start_time'])
        duration = float(data['duration_us'])
        direction = str(data['direction'])
        
        if debug:
            print(f"  {filename}: {len(x):,} events, {direction}, {duration/1000:.0f}ms")
        
        # Step 1: Normalize time by subtracting start time
        t_normalized = t - start_time
        
        # Step 2: Convert polarity to [-1, 1] if needed
        if p.min() >= 0 and p.max() <= 1:
            p = (p - 0.5) * 2
        
        # Step 3: Handle backward scans
        if 'Backward' in direction:
            # Reverse time: period - timestamps
            t_normalized = duration - t_normalized
            # Flip polarity
            p = -p
        
        # Add to lists
        all_x.append(x)
        all_y.append(y)
        all_t.append(t_normalized)
        all_p.append(p)
    
    # Concatenate all arrays
    if debug:
        print(f"Merging and sorting {len(segment_files)} segments...")
    merged_x = np.concatenate(all_x)
    merged_y = np.concatenate(all_y)
    merged_t = np.concatenate(all_t)
    merged_p = np.concatenate(all_p)
    
    # Sort by time for better processing
    sort_indices = np.argsort(merged_t)
    merged_x = merged_x[sort_indices]
    merged_y = merged_y[sort_indices]
    merged_t = merged_t[sort_indices]
    merged_p = merged_p[sort_indices]
    
    print(f"✓ Merged: {len(merged_x):,} events, {(merged_t.max() - merged_t.min())/1000:.1f} ms duration")
    if debug:
        print(f"  Spatial: X=[{merged_x.min():.0f}, {merged_x.max():.0f}], Y=[{merged_y.min():.0f}, {merged_y.max():.0f}]")
        print(f"  Temporal: [{merged_t.min():.0f}, {merged_t.max():.0f}] μs")
    
    return merged_x, merged_y, merged_t, merged_p

def save_learned_parameters(model, output_dir, filename_prefix, duration, debug=False):
    """
    Save learned parameters in a comprehensive format with parameter length in filename
    Saves in NPZ, JSON, and CSV formats
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameters
    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
    num_params = len(final_a_params)
    
    # Create filenames with parameter count
    param_file = os.path.join(output_dir, f"{filename_prefix}_learned_params_n{num_params}.npz")
    json_file = os.path.join(output_dir, f"{filename_prefix}_learned_params_n{num_params}.json")
    csv_file = os.path.join(output_dir, f"{filename_prefix}_learned_params_n{num_params}.csv")
    
    # Save parameters and metadata
    save_data = {
        'a_params': final_a_params,
        'b_params': final_b_params,
        'num_params': num_params,
        'duration': duration,
        'temperature': model.compensate.temperature,
        'a_fixed': model.compensate.a_fixed,
        'b_fixed': model.compensate.b_fixed,
        'boundary_trainable': model.compensate.boundary_trainable,
        'num_main_windows': model.compensate.num_main_windows,
        'num_total_windows': model.compensate.num_total_windows
    }
    
    # Add boundary offsets if they exist
    boundary_offsets = None
    if hasattr(model.compensate, 'boundary_offsets'):
        boundary_offsets = model.compensate.boundary_offsets.detach().cpu().numpy()
        save_data['boundary_offsets'] = boundary_offsets
    
    # Save NPZ file
    np.savez(param_file, **save_data)
    
    # Save JSON file
    json_data = {
        'a_params': final_a_params.tolist(),
        'b_params': final_b_params.tolist(),
        'num_params': int(num_params),
        'duration': float(duration),
        'temperature': float(model.compensate.temperature),
        'a_fixed': bool(model.compensate.a_fixed),
        'b_fixed': bool(model.compensate.b_fixed),
        'boundary_trainable': bool(model.compensate.boundary_trainable),
        'num_main_windows': int(model.compensate.num_main_windows),
        'num_total_windows': int(model.compensate.num_total_windows),
        'a_range': [float(final_a_params.min()), float(final_a_params.max())],
        'b_range': [float(final_b_params.min()), float(final_b_params.max())]
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with metadata
        writer.writerow(['# Multi-Window Scan Compensation Parameters'])
        writer.writerow([f'# Generated from: {filename_prefix}'])
        writer.writerow([f'# Number of parameters: {num_params}'])
        writer.writerow([f'# Duration: {duration:.0f} microseconds ({duration/1000:.1f} ms)'])
        writer.writerow([f'# Temperature: {model.compensate.temperature:.0f}'])
        writer.writerow([f'# Main windows: {model.compensate.num_main_windows}'])
        writer.writerow([f'# Total windows: {model.compensate.num_total_windows}'])
        writer.writerow([f'# a_fixed: {model.compensate.a_fixed}'])
        writer.writerow([f'# b_fixed: {model.compensate.b_fixed}'])
        writer.writerow([f'# boundary_trainable: {model.compensate.boundary_trainable}'])
        writer.writerow([f'# a_range: [{final_a_params.min():.6f}, {final_a_params.max():.6f}]'])
        writer.writerow([f'# b_range: [{final_b_params.min():.6f}, {final_b_params.max():.6f}]'])
        writer.writerow(['#'])
        
        # Write parameter table header
        if boundary_offsets is not None:
            writer.writerow(['param_index', 'a_param', 'b_param', 'boundary_offset'])
            # Write parameter data
            for i in range(num_params):
                writer.writerow([i, f'{final_a_params[i]:.8f}', f'{final_b_params[i]:.8f}', f'{boundary_offsets[i]:.8f}'])
        else:
            writer.writerow(['param_index', 'a_param', 'b_param'])
            # Write parameter data
            for i in range(num_params):
                writer.writerow([i, f'{final_a_params[i]:.8f}', f'{final_b_params[i]:.8f}'])
        
        # Write summary statistics
        writer.writerow(['#'])
        writer.writerow(['# Summary Statistics'])
        writer.writerow(['parameter_type', 'min_value', 'max_value', 'mean_value', 'std_value'])
        writer.writerow(['a_params', f'{final_a_params.min():.8f}', f'{final_a_params.max():.8f}', 
                        f'{final_a_params.mean():.8f}', f'{final_a_params.std():.8f}'])
        writer.writerow(['b_params', f'{final_b_params.min():.8f}', f'{final_b_params.max():.8f}', 
                        f'{final_b_params.mean():.8f}', f'{final_b_params.std():.8f}'])
        
        if boundary_offsets is not None:
            writer.writerow(['boundary_offsets', f'{boundary_offsets.min():.8f}', f'{boundary_offsets.max():.8f}', 
                            f'{boundary_offsets.mean():.8f}', f'{boundary_offsets.std():.8f}'])
    
    print(f"📁 Saved learned parameters:")
    print(f"   NPZ: {param_file}")
    print(f"   JSON: {json_file}")
    print(f"   CSV: {csv_file}")
    print(f"   Parameters: {num_params} each (a_params, b_params)")
    print(f"   a_range: [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
    print(f"   b_range: [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
    
    return param_file, json_file, csv_file

def load_learned_parameters(param_file, debug=False):
    """
    Load previously saved parameters from NPZ, JSON, or CSV files
    """
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    if debug:
        print(f"📂 Loading learned parameters from: {param_file}")
    
    file_ext = os.path.splitext(param_file)[1].lower()
    
    if file_ext == '.npz':
        return load_parameters_from_npz(param_file, debug)
    elif file_ext == '.json':
        return load_parameters_from_json(param_file, debug)
    elif file_ext == '.csv':
        return load_parameters_from_csv(param_file, debug)
    else:
        raise ValueError(f"Unsupported parameter file format: {file_ext}. Use .npz, .json, or .csv")

def load_parameters_from_npz(param_file, debug=False):
    """Load parameters from NPZ file"""
    data = np.load(param_file)
    
    # Extract parameters
    a_params = data['a_params']
    b_params = data['b_params']
    num_params = int(data['num_params'])
    duration = float(data['duration'])
    temperature = float(data['temperature'])
    a_fixed = bool(data['a_fixed'])
    b_fixed = bool(data['b_fixed'])
    boundary_trainable = bool(data['boundary_trainable'])
    
    if debug:
        print(f"   ✓ Loaded {num_params} parameters from NPZ")
        print(f"   ✓ Duration: {duration:.0f} μs")
        print(f"   ✓ Temperature: {temperature:.0f}")
        print(f"   ✓ a_fixed: {a_fixed}, b_fixed: {b_fixed}")
        print(f"   ✓ boundary_trainable: {boundary_trainable}")
        print(f"   ✓ a_range: [{a_params.min():.6f}, {a_params.max():.6f}]")
        print(f"   ✓ b_range: [{b_params.min():.6f}, {b_params.max():.6f}]")
    
    return {
        'a_params': a_params,
        'b_params': b_params,
        'num_params': num_params,
        'duration': duration,
        'temperature': temperature,
        'a_fixed': a_fixed,
        'b_fixed': b_fixed,
        'boundary_trainable': boundary_trainable
    }

def load_parameters_from_json(param_file, debug=False):
    """Load parameters from JSON file"""
    with open(param_file, 'r') as f:
        data = json.load(f)
    
    a_params = np.array(data['a_params'])
    b_params = np.array(data['b_params'])
    num_params = int(data['num_params'])
    duration = float(data['duration'])
    temperature = float(data['temperature'])
    a_fixed = bool(data['a_fixed'])
    b_fixed = bool(data['b_fixed'])
    boundary_trainable = bool(data['boundary_trainable'])
    
    if debug:
        print(f"   ✓ Loaded {num_params} parameters from JSON")
        print(f"   ✓ Duration: {duration:.0f} μs")
        print(f"   ✓ Temperature: {temperature:.0f}")
        print(f"   ✓ a_fixed: {a_fixed}, b_fixed: {b_fixed}")
        print(f"   ✓ boundary_trainable: {boundary_trainable}")
        print(f"   ✓ a_range: [{a_params.min():.6f}, {a_params.max():.6f}]")
        print(f"   ✓ b_range: [{b_params.min():.6f}, {b_params.max():.6f}]")
    
    return {
        'a_params': a_params,
        'b_params': b_params,
        'num_params': num_params,
        'duration': duration,
        'temperature': temperature,
        'a_fixed': a_fixed,
        'b_fixed': b_fixed,
        'boundary_trainable': boundary_trainable
    }

def load_parameters_from_csv(param_file, debug=False):
    """Load parameters from CSV file"""
    # Read metadata from comments
    metadata = {}
    a_params = []
    b_params = []
    
    with open(param_file, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if len(row) == 0:
                continue
                
            # Parse metadata from comments
            if row[0].startswith('#'):
                line = row[0]
                if 'Number of parameters:' in line:
                    metadata['num_params'] = int(line.split(':')[1].strip())
                elif 'Duration:' in line:
                    duration_str = line.split(':')[1].strip().split()[0]
                    metadata['duration'] = float(duration_str)
                elif 'Temperature:' in line:
                    metadata['temperature'] = float(line.split(':')[1].strip())
                elif 'a_fixed:' in line:
                    metadata['a_fixed'] = line.split(':')[1].strip().lower() == 'true'
                elif 'b_fixed:' in line:
                    metadata['b_fixed'] = line.split(':')[1].strip().lower() == 'true'
                elif 'boundary_trainable:' in line:
                    metadata['boundary_trainable'] = line.split(':')[1].strip().lower() == 'true'
                continue
            
            # Skip header row
            if row[0] == 'param_index' or row[0] == 'parameter_type':
                continue
                
            # Read parameter data
            if row[0].isdigit():
                param_index = int(row[0])
                a_param = float(row[1])
                b_param = float(row[2])
                a_params.append(a_param)
                b_params.append(b_param)
    
    # Convert to numpy arrays
    a_params = np.array(a_params)
    b_params = np.array(b_params)
    
    # Set defaults if not found in metadata
    num_params = metadata.get('num_params', len(a_params))
    duration = metadata.get('duration', 100000.0)  # Default duration
    temperature = metadata.get('temperature', 5000.0)  # Default temperature
    a_fixed = metadata.get('a_fixed', True)
    b_fixed = metadata.get('b_fixed', False)
    boundary_trainable = metadata.get('boundary_trainable', False)
    
    if debug:
        print(f"   ✓ Loaded {num_params} parameters from CSV")
        print(f"   ✓ Duration: {duration:.0f} μs")
        print(f"   ✓ Temperature: {temperature:.0f}")
        print(f"   ✓ a_fixed: {a_fixed}, b_fixed: {b_fixed}")
        print(f"   ✓ boundary_trainable: {boundary_trainable}")
        print(f"   ✓ a_range: [{a_params.min():.6f}, {a_params.max():.6f}]")
        print(f"   ✓ b_range: [{b_params.min():.6f}, {b_params.max():.6f}]")
    
    return {
        'a_params': a_params,
        'b_params': b_params,
        'num_params': num_params,
        'duration': duration,
        'temperature': temperature,
        'a_fixed': a_fixed,
        'b_fixed': b_fixed,
        'boundary_trainable': boundary_trainable
    }

class ScanCompensation(nn.Module):
    def __init__(self, duration, num_params=13, device='cuda', a_fixed=True, b_fixed=False, 
                 boundary_trainable=False, a_default=0.0, b_default=-76.0, temperature=5000, debug=False):
        super().__init__()
        
        # Initialize parameters based on defaults and fixed flags
        if debug:
            print(f"Initializing Multi-Window Compensation:")
            print(f"  Duration: {duration/1000:.1f} ms, Parameters: {num_params*3 if boundary_trainable else num_params*2} total")
            print(f"  a_fixed: {a_fixed}, b_fixed: {b_fixed}, boundary_trainable: {boundary_trainable}")
            print(f"  a_default: {a_default}, b_default: {b_default}, temperature: {temperature}")
        
        # Initialize with default values
        a_params = [a_default] * num_params
        b_params = [b_default] * num_params
        
        # Create the compensate object - it will create its own Parameters
        self.compensate = Compensate(a_params, b_params, duration, num_params=num_params,
                                   device=device, a_fixed=a_fixed, b_fixed=b_fixed, 
                                   boundary_trainable=boundary_trainable, temperature=temperature, debug=debug)
    
    def load_parameters(self, a_params, b_params, debug=False):
        """
        Load pre-trained parameters into the model
        """
        if debug:
            print(f"Loading pre-trained parameters into model...")
        
        # Update a_params
        if self.compensate.a_fixed:
            # Update the buffer
            self.compensate.a_params_buffer.data = torch.tensor(a_params, dtype=torch.float32, device=self.compensate.device)
        else:
            # Update the parameter
            self.compensate.a_params.data = torch.tensor(a_params, dtype=torch.float32, device=self.compensate.device)
        
        # Update b_params
        if self.compensate.b_fixed:
            # Update the buffer
            self.compensate.b_params_buffer.data = torch.tensor(b_params, dtype=torch.float32, device=self.compensate.device)
        else:
            # Update the parameter
            self.compensate.b_params.data = torch.tensor(b_params, dtype=torch.float32, device=self.compensate.device)
        
        if debug:
            print(f"   ✓ Updated a_params: [{a_params.min():.6f}, {a_params.max():.6f}]")
            print(f"   ✓ Updated b_params: [{b_params.min():.6f}, {b_params.max():.6f}]")
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps using multi-window compensation.
        """
        # Apply multi-window compensation
        compensation = self.compensate(x_coords, y_coords, timestamps)
        t_warped = timestamps - compensation
        
        return x_coords, y_coords, t_warped

    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, 
                original_t_start=None, original_t_end=None, chunk_size=500000, debug=False):
        """
        Process events through the model by warping them and then computing the loss.
        CRITICAL: Process in chunks for memory efficiency but sum variances for unified backpropagation
        """
        # Filter events to original time range if provided
        if original_t_start is not None and original_t_end is not None:
            valid_time_mask = (timestamps >= original_t_start) & (timestamps <= original_t_end)
            x_coords = x_coords[valid_time_mask]
            y_coords = y_coords[valid_time_mask]
            timestamps = timestamps[valid_time_mask]
            polarities = polarities[valid_time_mask]
            
            # Use original time range for binning
            t_start = original_t_start
            t_end = original_t_end
        else:
            # Use warped time range
            t_start = timestamps.min()
            t_end = timestamps.max()
        
        # Define time binning parameters
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1
        
        # Process events in chunks to avoid memory overflow
        num_events = len(x_coords)
        
        if debug:
            print(f"Processing {num_events:,} events in chunks of {chunk_size:,} for variance calculation...")
        
        # Create complete event tensor for all chunks
        complete_event_tensor = torch.zeros(num_bins, H, W, device=device, dtype=torch.float32)
        
        for start_idx in range(0, num_events, chunk_size):
            end_idx = min(start_idx + chunk_size, num_events)
            current_chunk_size = end_idx - start_idx
            
            # Get chunk
            x_chunk = x_coords[start_idx:end_idx]
            y_chunk = y_coords[start_idx:end_idx]
            t_chunk = timestamps[start_idx:end_idx]
            p_chunk = polarities[start_idx:end_idx]
            
            # Apply compensation to chunk
            compensation_chunk = self.compensate(x_chunk, y_chunk, t_chunk, chunk_size=chunk_size, debug=debug)
            t_warped_chunk = t_chunk - compensation_chunk
            
            # Normalize time to [0, num_bins) for chunk
            t_norm = (t_warped_chunk - t_start) / time_bin_width

            # Compute floor and ceil indices for time bins
            t0 = torch.floor(t_norm)
            t1 = t0 + 1

            # Compute weights for linear interpolation over time
            wt = (t_norm - t0).float()

            # Clamping indices to valid range
            t0_clamped = t0.clamp(0, num_bins - 1)
            t1_clamped = t1.clamp(0, num_bins - 1)

            # Cast x and y to long for indexing
            x_indices = x_chunk.long()
            y_indices = y_chunk.long()

            # Ensure spatial indices are within bounds
            valid_mask = (x_indices >= 0) & (x_indices < W) & \
                         (y_indices >= 0) & (y_indices < H)

            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            t0_clamped = t0_clamped[valid_mask]
            t1_clamped = t1_clamped[valid_mask]
            wt = wt[valid_mask]
            p_chunk = p_chunk[valid_mask]

            # Compute linear indices for the event tensor
            spatial_indices = y_indices * W + x_indices
            spatial_indices = spatial_indices.long()

            # For t0
            flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
            flat_indices_t0 = flat_indices_t0.long()
            weights_t0 = ((1 - wt) * p_chunk).float()

            # For t1
            flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
            flat_indices_t1 = flat_indices_t1.long()
            weights_t1 = (wt * p_chunk).float()

            # Combine indices and weights
            flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
            flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

            # Add explicit bounds checking to prevent CUDA errors
            num_elements = num_bins * H * W
            valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
            flat_indices = flat_indices[valid_flat_mask]
            flat_weights = flat_weights[valid_flat_mask]

            # Create the flattened event tensor for this chunk
            event_tensor_flat_chunk = torch.zeros(num_elements, device=device, dtype=torch.float32)

            # Accumulate events into the flattened tensor using scatter_add
            if len(flat_indices) > 0:
                event_tensor_flat_chunk = event_tensor_flat_chunk.scatter_add(0, flat_indices, flat_weights)

            # Reshape and add to complete tensor
            event_tensor_chunk = event_tensor_flat_chunk.view(num_bins, H, W)
            complete_event_tensor += event_tensor_chunk
            
            # Clear chunk tensors to free memory
            del event_tensor_chunk, event_tensor_flat_chunk
            del flat_indices, flat_weights, compensation_chunk, t_warped_chunk
            torch.cuda.empty_cache()
            
            # Progress indicator (only in debug mode)
            if debug and (start_idx // chunk_size) % 10 == 0:
                progress = (end_idx / num_events) * 100
                print(f"  Processed {end_idx:,}/{num_events:,} events ({progress:.1f}%)")

        # Compute the variance over x and y within each time bin on the COMPLETE tensor
        variances = torch.var(complete_event_tensor.view(num_bins, -1), dim=1)
        # Loss is the sum of variances from ALL chunks
        total_loss = torch.sum(variances)
        
        if debug:
            print(f"✓ Completed processing all {num_events:,} events, computed unified variance")

        return complete_event_tensor, total_loss

def train_scan_compensation(x, y, t, p, sensor_width=1280, sensor_height=720, 
                          bin_width=1e5, num_iterations=1000, learning_rate=1.0, debug=False,
                          smoothness_weight=0.001, a_fixed=True, b_fixed=False, boundary_trainable=False,
                          a_default=0.0, b_default=-76.0, num_params=13, temperature=5000, chunk_size=250000):
    """
    Train the multi-window scan compensation model with smoothness regularization
    CRITICAL FIX: Removed downsampling and batching for proper variance calculation
    Added support for fixing parameters during training
    Fixed: Skip training when no parameters are trainable
    """
    if debug:
        print(f"Training multi-window scan compensation...")
        print(f"  Sensor: {sensor_width} x {sensor_height}")
        print(f"  Bin width: {bin_width/1000:.1f} ms")
        print(f"  Iterations: {num_iterations}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Smoothness weight: {smoothness_weight}")
        print(f"  Parameter settings: a_fixed={a_fixed}, b_fixed={b_fixed}, boundary_trainable={boundary_trainable}")
        print(f"  Default values: a={a_default}, b={b_default}")
        print(f"  Num params: {num_params}, Temperature: {temperature}, Chunk size: {chunk_size:,}")
    
    # CRITICAL: Use ALL events - no more downsampling for proper variance calculation
    num_events = len(x)
    if debug:
        print(f"  Using ALL {num_events:,} events for training (no downsampling)")
    
    x_train = x
    y_train = y
    t_train = t
    p_train = p
    
    # Convert to tensors with explicit dtype
    xs = torch.tensor(x_train, device=device, dtype=torch.float32)
    ys = torch.tensor(y_train, device=device, dtype=torch.float32)
    ts = torch.tensor(t_train, device=device, dtype=torch.float32)
    ps = torch.tensor(p_train, device=device, dtype=torch.float32)

    # Store original time range (from original data)
    original_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
    duration = original_t_end - original_t_start
    
    if debug:
        print(f"  Time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs ({duration.item()/1000:.1f} ms)")

    # Initialize multi-window compensation model with parameter fixing
    model = ScanCompensation(duration.item(), num_params=num_params, device=device, 
                            a_fixed=a_fixed, b_fixed=b_fixed, boundary_trainable=boundary_trainable,
                            a_default=a_default, b_default=b_default, temperature=temperature, debug=debug)

    # Check if there are any trainable parameters
    trainable_params = list(model.parameters())
    has_trainable_params = len(trainable_params) > 0
    
    if not has_trainable_params:
        print(f"\n⚠️  NO TRAINABLE PARAMETERS - SKIPPING TRAINING")
        print(f"   All parameters are fixed. Model will use default values:")
        print(f"   a_params = {a_default}")
        print(f"   b_params = {b_default}")
        print(f"   Proceeding with evaluation and visualization...")
        
        # Create dummy loss histories for compatibility
        losses = [0.0] * num_iterations
        variance_losses = [0.0] * num_iterations
        smoothness_losses = [0.0] * num_iterations
        a_params_history = []
        b_params_history = []
        
        # Get fixed parameters for history
        final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
        final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
        
        # Fill history with same values (since they're fixed)
        for i in range(num_iterations):
            a_params_history.append(final_a_params.copy())
            b_params_history.append(final_b_params.copy())
        
        # Compute actual loss with fixed parameters for reporting
        with torch.no_grad():
            _, actual_loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                 original_t_start, original_t_end, chunk_size=chunk_size, debug=debug)
            actual_loss_value = actual_loss.item()
            
            # Update loss histories with actual computed loss
            losses = [actual_loss_value] * num_iterations
            variance_losses = [actual_loss_value] * num_iterations
        
        if debug:
            print(f"\nFixed Parameters Evaluation:")
            print(f"  a_params (FIXED): [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
            print(f"  b_params (FIXED): [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
            print(f"  Computed loss: {actual_loss_value:.6f}")
        
        return model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start, original_t_end

    # Define the optimizer - only optimizes non-fixed parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training Progress (ALL {num_events:,} events, chunked processing, unified variance):")
    print(f"{'Iter':<6} {'Total Loss':<12} {'Var Loss':<12} {'Smooth Loss':<12} {'a_range':<18} {'b_range':<18}")
    print("-" * 90)

    # Training loop
    losses = []
    variance_losses = []
    smoothness_losses = []
    a_params_history = []
    b_params_history = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute main variance loss - ALL events processed together
        event_tensor, variance_loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                           original_t_start, original_t_end, chunk_size=chunk_size, debug=debug)
        
        # Add smoothness regularization to encourage neighboring parameters to be similar
        # Only apply to non-fixed parameters
        smoothness_loss = torch.tensor(0.0, device=device)
        
        if not a_fixed:
            a_params = model.compensate.get_a_params()
            a_smooth_loss = torch.mean((a_params[1:] - a_params[:-1])**2)
            smoothness_loss += a_smooth_loss
        
        if not b_fixed:
            b_params = model.compensate.get_b_params()
            b_smooth_loss = torch.mean((b_params[1:] - b_params[:-1])**2)
            smoothness_loss += b_smooth_loss
        
        if boundary_trainable:
            # Add smoothness to boundary offsets
            boundary_offsets = model.compensate.boundary_offsets
            boundary_smooth_loss = torch.mean((boundary_offsets[1:] - boundary_offsets[:-1])**2)
            smoothness_loss += boundary_smooth_loss
        
        # Total loss: variance loss + smoothness regularization
        total_loss = variance_loss + smoothness_weight * smoothness_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        current_total_loss = total_loss.item()
        current_variance_loss = variance_loss.item()
        current_smoothness_loss = smoothness_loss.item()
        
        losses.append(current_total_loss)
        variance_losses.append(current_variance_loss)
        smoothness_losses.append(current_smoothness_loss)
        
        # Store parameter history
        current_a_params = model.compensate.get_a_params().detach().cpu().numpy().copy()
        current_b_params = model.compensate.get_b_params().detach().cpu().numpy().copy()
        a_params_history.append(current_a_params)
        b_params_history.append(current_b_params)
        
        # Clean progress updates
        if i % 100 == 0 or i == num_iterations - 1:  # Every 100 iterations + final
            a_range = f"[{current_a_params.min():.3f}, {current_a_params.max():.3f}]"
            b_range = f"[{current_b_params.min():.3f}, {current_b_params.max():.3f}]"
            print(f"{i:<6} {current_total_loss:<12.6f} {current_variance_loss:<12.6f} {current_smoothness_loss:<12.6f} {a_range:<18} {b_range:<18}")
        
        # Debug: More frequent updates
        elif debug and i % 50 == 0:
            a_range = f"[{current_a_params.min():.3f}, {current_a_params.max():.3f}]"
            b_range = f"[{current_b_params.min():.3f}, {current_b_params.max():.3f}]"
            print(f"{i:<6} {current_total_loss:<12.6f} {current_variance_loss:<12.6f} {current_smoothness_loss:<12.6f} {a_range:<18} {b_range:<18}")
        
        # Adjust the learning rate if needed
        if i == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            if debug:
                print(f"  → Learning rate reduced to {optimizer.param_groups[0]['lr']:.4f}")
        elif i == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            if debug:
                print(f"  → Learning rate reduced to {optimizer.param_groups[0]['lr']:.4f}")

    print("-" * 90)
    print(f"Training completed!")
    print(f"  Final total loss: {losses[-1]:.6f}")
    print(f"  Final variance loss: {variance_losses[-1]:.6f}")
    print(f"  Final smoothness loss: {smoothness_losses[-1]:.6f}")
    
    return model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start, original_t_end

def create_time_binned_frame_comparison(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, 
                                       output_dir=None, filename_prefix="", chunk_size=250000, debug=False):
    """
    Create and save individual time-binned frame comparison plots (before vs after compensation)
    Saves each time bin as separate frames showing temporal evolution
    """
    print("Creating time-binned frame comparison plots...")
    
    # Create event frames
    event_tensor_orig = create_event_frames(model, x, y, t, p, H, W, bin_width, 
                                           original_t_start, original_t_end, compensated=False, 
                                           chunk_size=chunk_size, debug=debug)
    event_tensor_comp = create_event_frames(model, x, y, t, p, H, W, bin_width, 
                                           original_t_start, original_t_end, compensated=True, 
                                           chunk_size=chunk_size, debug=debug)
    
    num_bins = event_tensor_orig.shape[0]
    duration_ms = (original_t_end - original_t_start).item() / 1000
    bin_width_ms = bin_width / 1000
    
    print(f"   📊 Processing {num_bins} time bins over {duration_ms:.1f}ms (bin width: {bin_width_ms:.1f}ms)")
    
    # Create output directory for time-binned frames
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subfolder for time-binned frame comparisons
        frames_dir = os.path.join(output_dir, "time_binned_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get parameter suffix for filename
        param_suffix = get_param_suffix(model)
        
        # Create overview comparison plot with multiple time bins
        fig, axes = plt.subplots(2, min(num_bins, 8), figsize=(min(num_bins, 8) * 3, 8))
        if num_bins == 1:
            axes = axes.reshape(2, 1)
        
        # Show up to 8 time bins in overview
        bins_to_show = min(num_bins, 8)
        bin_indices = np.linspace(0, num_bins-1, bins_to_show, dtype=int)
        
        for i, bin_idx in enumerate(bin_indices):
            # Original frame
            frame_orig = event_tensor_orig[bin_idx].detach().cpu().numpy()
            im1 = axes[0, i].imshow(frame_orig, cmap='hot', aspect='auto', origin='lower', vmin=0)
            axes[0, i].set_title(f'Original\nBin {bin_idx} ({bin_idx*bin_width_ms:.1f}ms)', fontsize=10)
            axes[0, i].set_xlabel('X (pixels)', fontsize=8)
            if i == 0:
                axes[0, i].set_ylabel('Y (pixels)', fontsize=8)
            
            # Compensated frame
            frame_comp = event_tensor_comp[bin_idx].detach().cpu().numpy()
            im2 = axes[1, i].imshow(frame_comp, cmap='hot', aspect='auto', origin='lower', vmin=0)
            axes[1, i].set_title(f'Compensated\nBin {bin_idx} ({bin_idx*bin_width_ms:.1f}ms)', fontsize=10)
            axes[1, i].set_xlabel('X (pixels)', fontsize=8)
            if i == 0:
                axes[1, i].set_ylabel('Y (pixels)', fontsize=8)
        
        # Add colorbar
        plt.colorbar(im2, ax=axes, shrink=0.6, label='Events')
        
        plt.suptitle(f'Time-Binned Frame Comparison Overview\n{len(x):,} events, {num_bins} bins, {duration_ms:.1f}ms total', fontsize=14)
        plt.tight_layout()
        
        # Save overview plot
        overview_path = os.path.join(frames_dir, f"{filename_prefix}_time_binned_overview{param_suffix}.png")
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📸 Time-binned overview saved to: {overview_path}")
        
        # Save individual frames for each time bin
        all_frame_data = {}
        statistics = []
        
        for bin_idx in range(num_bins):
            # Get frames for this time bin
            frame_orig = event_tensor_orig[bin_idx].detach().cpu().numpy()
            frame_comp = event_tensor_comp[bin_idx].detach().cpu().numpy()
            
            # Calculate statistics for this bin
            orig_std = np.std(frame_orig)
            comp_std = np.std(frame_comp)
            orig_sum = np.sum(frame_orig)
            comp_sum = np.sum(frame_comp)
            orig_max = np.max(frame_orig)
            comp_max = np.max(frame_comp)
            
            std_improvement = ((comp_std - orig_std) / orig_std * 100) if orig_std > 0 else 0
            max_improvement = ((comp_max - orig_max) / orig_max * 100) if orig_max > 0 else 0
            
            statistics.append({
                'bin_idx': bin_idx,
                'time_ms': bin_idx * bin_width_ms,
                'orig_std': orig_std,
                'comp_std': comp_std,
                'orig_sum': orig_sum,
                'comp_sum': comp_sum,
                'orig_max': orig_max,
                'comp_max': comp_max,
                'std_improvement': std_improvement,
                'max_improvement': max_improvement
            })
            
            # Store frame data
            all_frame_data[f'original_bin_{bin_idx}'] = frame_orig
            all_frame_data[f'compensated_bin_{bin_idx}'] = frame_comp
            
            # Create side-by-side comparison for this time bin
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original frame
            im1 = axes[0].imshow(frame_orig, cmap='hot', aspect='auto', origin='lower')
            axes[0].set_title(f'Original - Time Bin {bin_idx}\nTime: {bin_idx*bin_width_ms:.1f}ms', fontsize=12)
            axes[0].set_xlabel('X (pixels)')
            axes[0].set_ylabel('Y (pixels)')
            cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
            cbar1.set_label('Events')
            
            # Compensated frame
            im2 = axes[1].imshow(frame_comp, cmap='hot', aspect='auto', origin='lower')
            axes[1].set_title(f'Compensated - Time Bin {bin_idx}\nTime: {bin_idx*bin_width_ms:.1f}ms', fontsize=12)
            axes[1].set_xlabel('X (pixels)')
            axes[1].set_ylabel('Y (pixels)')
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
            cbar2.set_label('Events')
            
            # Add statistics text
            stats_text = (
                f'Bin {bin_idx} Statistics:\n'
                f'Original: std={orig_std:.2f}, max={orig_max:.0f}, sum={orig_sum:.0f}\n'
                f'Compensated: std={comp_std:.2f}, max={comp_max:.0f}, sum={comp_sum:.0f}\n'
                f'Std improvement: {std_improvement:+.1f}%\n'
                f'Max improvement: {max_improvement:+.1f}%'
            )
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.9))
            
            plt.suptitle(f'Time Bin {bin_idx} Comparison - Multi-Window Compensation\n'
                        f'{bin_width_ms:.1f}ms bin at {bin_idx*bin_width_ms:.1f}ms', fontsize=14)
            plt.tight_layout()
            
            # Save individual comparison
            bin_path = os.path.join(frames_dir, f"{filename_prefix}_bin_{bin_idx:02d}_comparison{param_suffix}.png")
            plt.savefig(bin_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if debug or bin_idx % 5 == 0:  # Print progress every 5 bins or in debug mode
                print(f"   📸 Saved bin {bin_idx}/{num_bins-1}: {bin_path}")
        
        # Save all frame data as NPZ
        data_path = os.path.join(frames_dir, f"{filename_prefix}_all_time_bins_data{param_suffix}.npz")
        
        # Add metadata to frame data
        all_frame_data.update({
            'num_bins': num_bins,
            'bin_width_us': bin_width,
            'bin_width_ms': bin_width_ms,
            'duration_ms': duration_ms,
            'total_events': len(x),
            'tensor_shape': event_tensor_orig.shape,
            'statistics': statistics
        })
        
        np.savez(data_path, **all_frame_data)
        
        # Save statistics as CSV
        stats_path = os.path.join(frames_dir, f"{filename_prefix}_time_bin_statistics{param_suffix}.csv")
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# Time-Binned Frame Statistics'])
            writer.writerow([f'# Total bins: {num_bins}'])
            writer.writerow([f'# Bin width: {bin_width_ms:.1f}ms'])
            writer.writerow([f'# Duration: {duration_ms:.1f}ms'])
            writer.writerow([f'# Total events: {len(x):,}'])
            writer.writerow(['#'])
            
            # Write header
            writer.writerow(['bin_idx', 'time_ms', 'orig_std', 'comp_std', 'orig_sum', 'comp_sum', 
                           'orig_max', 'comp_max', 'std_improvement_pct', 'max_improvement_pct'])
            
            # Write data
            for stat in statistics:
                writer.writerow([
                    stat['bin_idx'], f"{stat['time_ms']:.1f}", 
                    f"{stat['orig_std']:.4f}", f"{stat['comp_std']:.4f}",
                    f"{stat['orig_sum']:.0f}", f"{stat['comp_sum']:.0f}",
                    f"{stat['orig_max']:.0f}", f"{stat['comp_max']:.0f}",
                    f"{stat['std_improvement']:.2f}", f"{stat['max_improvement']:.2f}"
                ])
        
        print(f"📊 Time bin data saved to: {data_path}")
        print(f"📈 Statistics CSV saved to: {stats_path}")
        print(f"📸 Individual bin comparisons saved: {num_bins} files in {frames_dir}/")
        
        # Calculate overall statistics
        overall_orig_std = np.mean([s['orig_std'] for s in statistics])
        overall_comp_std = np.mean([s['comp_std'] for s in statistics])
        overall_improvement = ((overall_comp_std - overall_orig_std) / overall_orig_std * 100) if overall_orig_std > 0 else 0
        
        print(f"\n📈 Time-Binned Frame Analysis:")
        print(f"  Total time bins: {num_bins}")
        print(f"  Bin width: {bin_width_ms:.1f}ms")
        print(f"  Total duration: {duration_ms:.1f}ms")
        print(f"  Total events: {len(x):,}")
        print(f"  Average original std: {overall_orig_std:.3f}")
        print(f"  Average compensated std: {overall_comp_std:.3f}")
        print(f"  Average std improvement: {overall_improvement:+.1f}%")
        
        return event_tensor_orig, event_tensor_comp, statistics
    
    else:
        print("⚠️  No output directory specified - skipping frame saves")
        return event_tensor_orig, event_tensor_comp, []

def create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, 
                       compensated=True, chunk_size=250000, debug=False):
    """
    Create event frames with or without compensation
    """
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        if compensated:
            # Use current model parameters
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end, 
                                   chunk_size=chunk_size, debug=debug)
        else:
            # Temporarily set all parameters to their initialization values
            if not model.compensate.a_fixed:
                original_a_params = model.compensate.a_params.clone()
                model.compensate.a_params.data.zero_()
            if not model.compensate.b_fixed:
                original_b_params = model.compensate.b_params.clone()
                model.compensate.b_params.data.zero_()
                
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end,
                                   chunk_size=chunk_size, debug=debug)
            
            # Restore parameters
            if not model.compensate.a_fixed:
                model.compensate.a_params.data = original_a_params
            if not model.compensate.b_fixed:
                model.compensate.b_params.data = original_b_params
    
    return event_tensor

def plot_learned_parameters_with_data(model, x, y, t, sensor_width, sensor_height, output_dir=None, filename_prefix=""):
    """
    Plot learned parameters with actual event data projections in X-T and Y-T planes
    """
    print("Creating learned parameters visualization with data projections...")
    
    # Get final parameters
    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
    
    # Create coordinate ranges for visualization
    x_range = np.linspace(0, sensor_width, 100)
    y_range = np.linspace(0, sensor_height, 100)
    
    # Time range based on the compensation model's duration
    duration = model.compensate.duration
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
    
    # Helper function to compute boundary lines
    def compute_boundary_lines_simple(a_params, b_params, coord_range, coord_type):
        boundary_offsets = model.compensate.boundary_offsets.detach().cpu().numpy()
        lines = []
        
        for i in range(len(a_params)):
            if coord_type == 'x':
                # X-T view: fix Y at center, vary X
                y_center = sensor_height / 2
                line_values = a_params[i] * coord_range + b_params[i] * y_center + boundary_offsets[i]
            else:  # coord_type == 'y'
                # Y-T view: fix X at center, vary Y  
                x_center = sensor_width / 2
                line_values = a_params[i] * x_center + b_params[i] * coord_range + boundary_offsets[i]
            
            lines.append(line_values)
        
        return lines
    
    # Subsample data for visualization (to avoid overcrowding)
    # max_plot_events = 50000  # Limit events for clear visualization
    max_plot_events = 10000000  # Limit events for clear visualization
    if len(x) > max_plot_events:
        plot_indices = np.random.choice(len(x), max_plot_events, replace=False)
        x_plot = x[plot_indices]
        y_plot = y[plot_indices]
        t_plot = t[plot_indices]
    else:
        x_plot = x
        y_plot = y
        t_plot = t
    
    # Plot 1: X-T view with data and boundaries
    ax = axes[0, 0]
    
    # Plot event data as scatter (in background)
    ax.scatter(x_plot, t_plot/1000, c='lightblue', alpha=0.1, s=0.1, rasterized=True, label='Events')
    
    # Plot learned boundary lines (on top)
    final_x_lines = compute_boundary_lines_simple(final_a_params, final_b_params, x_range, 'x')
    for i, line_values in enumerate(final_x_lines):
        color = colors[i % len(colors)]
        ax.plot(x_range, line_values/1000, '--', alpha=0.8, linewidth=2.5, color=color, label=f'Boundary {i}')
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Learned Compensation - X-T View\n(Events + Boundary Lines)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, duration/1000)
    ax.set_xlim(0, sensor_width)
    
    # Plot 2: Y-T view with data and boundaries
    ax = axes[0, 1]
    
    # Plot event data as scatter (in background)
    ax.scatter(y_plot, t_plot/1000, c='lightgreen', alpha=0.1, s=0.1, rasterized=True, label='Events')
    
    # Plot learned boundary lines (on top)
    final_y_lines = compute_boundary_lines_simple(final_a_params, final_b_params, y_range, 'y')
    for i, line_values in enumerate(final_y_lines):
        color = colors[i % len(colors)]
        ax.plot(y_range, line_values/1000, '--', alpha=0.8, linewidth=2.5, color=color, label=f'Boundary {i}')
    
    ax.set_xlabel('Y (pixels)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Learned Compensation - Y-T View\n(Events + Boundary Lines)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, duration/1000)
    ax.set_xlim(0, sensor_height)
    
    # Plot 3: X-T view boundaries only (for clarity)
    ax = axes[1, 0]
    for i, line_values in enumerate(final_x_lines):
        color = colors[i % len(colors)]
        ax.plot(x_range, line_values/1000, '--', alpha=0.8, linewidth=3, color=color, label=f'Boundary {i}')
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Learned Boundaries Only - X-T View', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, duration/1000)
    ax.set_xlim(0, sensor_width)
    
    # Plot 4: Y-T view boundaries only (for clarity)
    ax = axes[1, 1]
    for i, line_values in enumerate(final_y_lines):
        color = colors[i % len(colors)]
        ax.plot(y_range, line_values/1000, '--', alpha=0.8, linewidth=3, color=color, label=f'Boundary {i}')
    
    ax.set_xlabel('Y (pixels)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Learned Boundaries Only - Y-T View', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_ylim(0, duration/1000)
    ax.set_xlim(0, sensor_height)
    
    # Add comprehensive parameter statistics
    a_status = "FIXED" if model.compensate.a_fixed else "TRAINABLE"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINABLE"
    
    fig.text(0.02, 0.02, 
             f'Multi-Window Compensation Summary:\n'
             f'• Total events: {len(x):,} (plotted: {len(x_plot):,})\n'
             f'• Duration: {duration/1000:.1f} ms\n'
             f'• Main windows: {model.compensate.num_main_windows}, Total windows: {model.compensate.num_total_windows}\n'
             f'• a_params ({a_status}): [{final_a_params.min():.4f}, {final_a_params.max():.4f}] (range: {final_a_params.max()-final_a_params.min():.4f})\n'
             f'• b_params ({b_status}): [{final_b_params.min():.4f}, {final_b_params.max():.4f}] (range: {final_b_params.max()-final_b_params.min():.4f})',
             fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.suptitle('Multi-Window Scan Compensation: Learned Boundaries with Event Data (CHUNKED PROCESSING)', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, f"{filename_prefix}_learned_parameters_with_data_chunked_processing.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Parameters with data plot saved to: {plot_path}")
    
    plt.show()
    
    # Print parameter summary
    print(f"\nLearned Parameters Summary:")
    print(f"  a_params ({a_status}): {final_a_params}")
    print(f"  b_params ({b_status}): {final_b_params}")
    print(f"  a_params range: {final_a_params.min():.6f} to {final_a_params.max():.6f}")
    print(f"  b_params range: {final_b_params.min():.6f} to {final_b_params.max():.6f}")

def get_param_string(model):
    """
    Get a formatted string of the model parameters for use in titles
    """
    a_params = model.compensate.get_a_params().detach().cpu().numpy()
    b_params = model.compensate.get_b_params().detach().cpu().numpy()
    a_status = "FIXED" if model.compensate.a_fixed else "TRAIN"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAIN"
    return f"a({a_status})=[{a_params.min():.4f},{a_params.max():.4f}], b({b_status})=[{b_params.min():.4f},{b_params.max():.4f}]"

def get_param_suffix(model):
    """
    Get a filename-safe suffix with the model parameters
    """
    a_params = model.compensate.get_a_params().detach().cpu().numpy()
    b_params = model.compensate.get_b_params().detach().cpu().numpy()
    a_status = "fixed" if model.compensate.a_fixed else "train"
    b_status = "fixed" if model.compensate.b_fixed else "train"
    return f"_multiwindow_chunked_processing_a{a_status}_{a_params.min():.4f}_{a_params.max():.4f}_b{b_status}_{b_params.min():.4f}_{b_params.max():.4f}"

def visualize_results(model, x, y, t, p, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, bin_width, 
                     sensor_width, sensor_height, original_t_start, original_t_end, 
                     output_dir=None, filename_prefix=""):
    """
    Visualize training results and compensated events with smoothness loss tracking
    """
    # Get parameter string for plots
    param_str = get_param_string(model)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plots
    axes[0, 0].plot(losses, label='Total Loss', alpha=0.8)
    axes[0, 0].plot(variance_losses, label='Variance Loss', alpha=0.8)
    axes[0, 0].plot(smoothness_losses, label='Smoothness Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses (Chunked Processing)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Only set log scale if there are positive values
    if any(l > 0 for l in losses):
        axes[0, 0].set_yscale('log')  # Log scale for better visualization
    
    # Parameters evolution - show range of a_params and b_params
    # Handle case where parameter history might be empty or malformed
    if len(a_params_history) > 0 and len(b_params_history) > 0:
        a_params_array = np.array(a_params_history)
        b_params_array = np.array(b_params_history)
        
        # Check if arrays have the right shape
        if a_params_array.ndim >= 2 and b_params_array.ndim >= 2:
            a_min = np.min(a_params_array, axis=1)
            a_max = np.max(a_params_array, axis=1)
            b_min = np.min(b_params_array, axis=1)
            b_max = np.max(b_params_array, axis=1)
            
            axes[0, 1].plot(a_min, label='a_params min', alpha=0.7)
            axes[0, 1].plot(a_max, label='a_params max', alpha=0.7)
            axes[0, 1].plot(b_min, label='b_params min', alpha=0.7)
            axes[0, 1].plot(b_max, label='b_params max', alpha=0.7)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Parameter Value')
            axes[0, 1].set_title('Parameter Evolution (Min/Max)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, 'Parameter history\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parameter Evolution (Min/Max)')
    else:
        axes[0, 1].text(0.5, 0.5, 'Parameter history\nnot available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Parameter Evolution (Min/Max)')
    
    # Generate event frames with original time range
    event_tensor_orig = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
                                           original_t_start, original_t_end, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
                                           original_t_start, original_t_end, compensated=True)
    
    # Get actual number of bins from tensor shape
    actual_num_bins = event_tensor_orig.shape[0]
    print(f"Visualization - Original tensor shape: {event_tensor_orig.shape}")
    print(f"Visualization - Compensated tensor shape: {event_tensor_comp.shape}")
    
    # Select a middle time bin to visualize
    bin_idx = actual_num_bins // 2
    
    # Original event frame
    frame_orig = event_tensor_orig[bin_idx].detach().cpu().numpy()
    im1 = axes[0, 2].imshow(frame_orig, cmap='inferno', aspect='auto')
    axes[0, 2].set_title(f'Original - Bin {bin_idx}')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Compensated event frame  
    frame_comp = event_tensor_comp[bin_idx].detach().cpu().numpy()
    im2 = axes[1, 2].imshow(frame_comp, cmap='inferno', aspect='auto')
    axes[1, 2].set_title(f'Multi-Window Compensated - Bin {bin_idx}')
    plt.colorbar(im2, ax=axes[1, 2])
    
    # Variance comparison
    with torch.no_grad():
        # Both tensors should now have the same shape due to time range filtering
        if event_tensor_orig.shape != event_tensor_comp.shape:
            print("Warning: Tensors still have different shapes after time range filtering!")
            min_bins = min(event_tensor_orig.shape[0], event_tensor_comp.shape[0])
            event_tensor_orig = event_tensor_orig[:min_bins]
            event_tensor_comp = event_tensor_comp[:min_bins]
            print(f"Trimmed to {min_bins} bins")
            actual_num_bins = min_bins
        
        # Calculate variance for each time bin
        current_num_bins, H, W = event_tensor_orig.shape
        var_orig_tensor = torch.var(event_tensor_orig.reshape(current_num_bins, H * W), dim=1)
        var_comp_tensor = torch.var(event_tensor_comp.reshape(current_num_bins, H * W), dim=1)
        
        # Convert to lists for plotting
        var_orig_list = var_orig_tensor.cpu().tolist()
        var_comp_list = var_comp_tensor.cpu().tolist()
        
        # Calculate mean values
        var_orig_mean = var_orig_tensor.mean().item()
        var_comp_mean = var_comp_tensor.mean().item()
    
    axes[1, 0].plot(var_orig_list, label='Original', alpha=0.7)
    axes[1, 0].plot(var_comp_list, label='Multi-Window Compensated (Chunked)', alpha=0.7)
    axes[1, 0].set_xlabel('Time Bin')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Variance Comparison (Chunked Compensation)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary statistics
    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
    improvement_pct = (var_comp_mean/var_orig_mean - 1) * 100
    
    # Parameter smoothness metrics
    a_smoothness = np.mean((final_a_params[1:] - final_a_params[:-1])**2)
    b_smoothness = np.mean((final_b_params[1:] - final_b_params[:-1])**2)
    
    # Status indicators
    a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
    
    axes[1, 1].text(0.1, 0.95, f'Original mean variance: {var_orig_mean:.2f}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.90, f'Compensated mean variance: {var_comp_mean:.2f}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.85, f'Improvement: {improvement_pct:.1f}%', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.80, f'CHUNKED PROCESSING - ALL {len(x):,} events', transform=axes[1, 1].transAxes, fontsize=10, weight='bold')
    axes[1, 1].text(0.1, 0.75, f'a_params ({a_status}): [{final_a_params.min():.2f}, {final_a_params.max():.2f}]', transform=axes[1, 1].transAxes, fontsize=9)
    axes[1, 1].text(0.1, 0.70, f'b_params ({b_status}): [{final_b_params.min():.2f}, {final_b_params.max():.2f}]', transform=axes[1, 1].transAxes, fontsize=9)
    axes[1, 1].text(0.1, 0.65, f'a_smoothness: {a_smoothness:.4f}', transform=axes[1, 1].transAxes, fontsize=9)
    axes[1, 1].text(0.1, 0.60, f'b_smoothness: {b_smoothness:.4f}', transform=axes[1, 1].transAxes, fontsize=9)
    axes[1, 1].text(0.1, 0.50, f'Final total loss: {losses[-1]:.6f}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.45, f'Final variance loss: {variance_losses[-1]:.6f}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.40, f'Final smoothness loss: {smoothness_losses[-1]:.6f}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.35, f'Total events: {len(x):,}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.30, f'Time bins: {actual_num_bins}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.25, f'Main windows: {model.compensate.num_main_windows}', transform=axes[1, 1].transAxes, fontsize=10)
    
    # Count trainable parameters
    trainable_params = 0
    if not model.compensate.a_fixed:
        trainable_params += len(final_a_params)
    if not model.compensate.b_fixed:
        trainable_params += len(final_b_params)
    total_params = len(final_a_params) + len(final_b_params)
    
    axes[1, 1].text(0.1, 0.20, f'Trainable params: {trainable_params}/{total_params}', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Multi-Window Summary (Chunked Compensation)')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Add overall title with parameters
    fig.suptitle(f'Multi-Window Scan Compensation Results (CHUNKED PROCESSING - UNIFIED VARIANCE)\n{param_str}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_multiwindow_compensation_results{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
    
    plt.show()

def save_results(model, losses, a_params_history, b_params_history, output_dir, filename_prefix):
    """
    Save training results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get parameter suffix for filenames
        param_suffix = get_param_suffix(model)
        
        # Save final parameters
        final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
        final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
        
        results_path = os.path.join(output_dir, f"{filename_prefix}_multiwindow_compensation_results{param_suffix}.txt")
        
        with open(results_path, 'w') as f:
            f.write("MULTI-WINDOW SCAN COMPENSATION RESULTS (CHUNKED PROCESSING - UNIFIED VARIANCE)\n")
            f.write("=" * 80 + "\n\n")
            
            a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
            b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
            
            f.write(f"Parameter Status:\n")
            f.write(f"  a_params: {a_status}\n")
            f.write(f"  b_params: {b_status}\n\n")
            
            f.write(f"Processing Method: CHUNKED PROCESSING - Memory efficient processing throughout\n")
            f.write(f"entire pipeline with unified variance calculation for proper gradient descent learning.\n\n")
            
            f.write(f"Final a_params: {final_a_params.tolist()}\n")
            f.write(f"Final b_params: {final_b_params.tolist()}\n")
            f.write(f"a_params range: [{final_a_params.min():.6f}, {final_a_params.max():.6f}]\n")
            f.write(f"b_params range: [{final_b_params.min():.6f}, {final_b_params.max():.6f}]\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Training iterations: {len(losses)}\n")
            f.write(f"Main windows: {model.compensate.num_main_windows}\n")
            f.write(f"Total windows: {model.compensate.num_total_windows}\n")
            
            # Count trainable parameters
            trainable_params = 0
            if not model.compensate.a_fixed:
                trainable_params += len(final_a_params)
            if not model.compensate.b_fixed:
                trainable_params += len(final_b_params)
            total_params = len(final_a_params) + len(final_b_params)
            
            f.write(f"Trainable parameters: {trainable_params}/{total_params}\n")
            f.write(f"Duration: {model.compensate.duration:.0f} μs\n")
        
        print(f"Results saved to: {results_path}")
        
        # Save parameters as numpy arrays
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_a_params{param_suffix}.npy"), final_a_params)
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_b_params{param_suffix}.npy"), final_b_params)
        np.save(os.path.join(output_dir, f"{filename_prefix}_loss_history{param_suffix}.npy"), np.array(losses))
        np.save(os.path.join(output_dir, f"{filename_prefix}_a_params_history{param_suffix}.npy"), np.array(a_params_history))
        np.save(os.path.join(output_dir, f"{filename_prefix}_b_params_history{param_suffix}.npy"), np.array(b_params_history))

def main():
    parser = argparse.ArgumentParser(description='Multi-Window Scan compensation for NPZ event files (CHUNKED PROCESSING VERSION with Time-Binned Frame Comparisons)')
    parser.add_argument('input_path', help='Path to NPZ event file OR segments folder (when using --merge)')
    parser.add_argument('--merge', action='store_true', help='Merge all scan segments from folder instead of processing single file')
    parser.add_argument('--output_dir', default=None, help='Output directory for results and time-binned frame comparisons')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--smoothness_weight', type=float, default=0.001, help='Weight for smoothness regularization (default: 0.001)')
    
    # Parameter configuration arguments
    parser.add_argument('--num_params', type=int, default=13, help='Number of parameters for a_params and b_params (default: 13)')
    parser.add_argument('--temperature', type=float, default=5000, help='Temperature for sigmoid functions in compensation (default: 5000)')
    parser.add_argument('--chunk_size', type=int, default=250000, help='Chunk size for memory-efficient processing (default: 250000)')
    
    # Parameter fixing arguments
    parser.add_argument('--a_fixed', action='store_true', default=True, help='Fix a_params during training (default: True)')
    parser.add_argument('--a_trainable', dest='a_fixed', action='store_false', help='Make a_params trainable during training')
    parser.add_argument('--b_fixed', action='store_true', help='Fix b_params during training (default: False)')
    parser.add_argument('--b_trainable', dest='b_fixed', action='store_false', default=True, help='Make b_params trainable during training (default)')
    parser.add_argument('--boundary_trainable', action='store_true', help='Make boundary offsets trainable (default: False)')
    parser.add_argument('--a_default', type=float, default=0.0, help='Default value for a_params (default: 0.0)')
    parser.add_argument('--b_default', type=float, default=-76.0, help='Default value for b_params (default: -76.0)')
    
    # NEW: Parameter loading/saving arguments
    parser.add_argument('--load_params', type=str, default=None, help='Path to saved parameters file (.npz, .json, or .csv) to load and skip training')
    parser.add_argument('--save_params', action='store_true', default=True, help='Save learned parameters in NPZ, JSON, and CSV formats (default: True)')
    parser.add_argument('--no_save_params', dest='save_params', action='store_false', help='Do not save learned parameters')
    
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--plot_params', action='store_true', help='Plot learned parameters in X-T and Y-T planes')
    parser.add_argument('--save_frames', action='store_true', default=True, help='Save time-binned frame comparison plots (default: True)')
    parser.add_argument('--no_save_frames', dest='save_frames', action='store_false', help='Do not save time-binned frame comparison plots')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    if not args.debug:
        print("="*80)
        print("MULTI-WINDOW SCAN COMPENSATION - CHUNKED PROCESSING VERSION") 
        print("CRITICAL: Chunked processing throughout pipeline + unified variance for proper learning")
        print("NEW: Save/Load learned parameters functionality (NPZ/JSON/CSV formats)")
        print("NEW: Time-binned frame comparison plots showing temporal evolution")
        print("="*80)
    
    # Handle input path and create base name
    if args.merge:
        # Input is a folder containing segments
        segments_folder = args.input_path
        if not os.path.isdir(segments_folder):
            raise ValueError(f"When using --merge, input_path must be a directory: {segments_folder}")
        
        # Set output directory
        if args.output_dir is None:
            args.output_dir = segments_folder
        
        # Create filename prefix for merged results
        folder_name = os.path.basename(segments_folder.rstrip('/'))
        base_name = f"{folder_name}_merged_chunked_processing"
        
        if not args.debug:
            print(f"Merging segments from: {segments_folder}")
        
        # Load and merge all segments
        x, y, t, p = load_and_merge_segments(segments_folder, debug=args.debug)
        
    else:
        # Input is a single NPZ file
        npz_file = args.input_path
        if not os.path.isfile(npz_file):
            raise ValueError(f"NPZ file not found: {npz_file}")
        
        # Set output directory
        if args.output_dir is None:
            args.output_dir = os.path.dirname(npz_file)
        
        # Create filename prefix
        base_name = f"{os.path.splitext(os.path.basename(npz_file))[0]}_chunked_processing"
        
        if not args.debug:
            print(f"Analyzing: {npz_file}")
        
        # Load events from single file
        x, y, t, p = load_npz_events(npz_file, debug=args.debug)
    
    if not args.debug:
        print(f"\n🔥 PROCESSING {len(x):,} EVENTS - CHUNKED PROCESSING, UNIFIED EVENT TENSOR & VARIANCE 🔥")
    
    # Calculate duration for model initialization
    original_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
    duration = original_t_end - original_t_start
    
    # Check if we should load saved parameters
    if args.load_params is not None:
        print(f"\n📂 LOADING SAVED PARAMETERS")
        print("="*50)
        
        # Load saved parameters
        param_data = load_learned_parameters(args.load_params, debug=args.debug)
        
        # Validate parameter count matches
        if param_data['num_params'] != args.num_params:
            print(f"⚠️  Warning: Loaded parameters have {param_data['num_params']} params, but --num_params={args.num_params}")
            print(f"   Using loaded parameter count: {param_data['num_params']}")
            args.num_params = param_data['num_params']
        
        # Validate duration compatibility (allow some tolerance)
        loaded_duration = param_data['duration']
        current_duration = duration.item()
        duration_diff = abs(loaded_duration - current_duration)
        if duration_diff > 1000:  # 1ms tolerance
            print(f"⚠️  Warning: Duration mismatch - Loaded: {loaded_duration:.0f}μs, Current: {current_duration:.0f}μs")
            print(f"   Difference: {duration_diff:.0f}μs")
        
        # Create model with loaded configuration
        model = ScanCompensation(
            duration.item(), 
            num_params=param_data['num_params'], 
            device=device,
            a_fixed=param_data['a_fixed'], 
            b_fixed=param_data['b_fixed'], 
            boundary_trainable=param_data['boundary_trainable'],
            a_default=args.a_default, 
            b_default=args.b_default, 
            temperature=param_data['temperature'], 
            debug=args.debug
        )
        
        # Load the saved parameters
        model.load_parameters(param_data['a_params'], param_data['b_params'], debug=args.debug)
        
        # Create dummy training history for compatibility
        losses = [0.0]
        variance_losses = [0.0] 
        smoothness_losses = [0.0]
        a_params_history = [param_data['a_params']]
        b_params_history = [param_data['b_params']]
        
        print(f"✅ Successfully loaded parameters from: {args.load_params}")
        print(f"   Skipping training - using pre-trained parameters")
        
    else:
        print(f"\n🎯 TRAINING NEW PARAMETERS")
        print("="*50)
        
        # Train multi-window model with all configurable parameters
        model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start, original_t_end = train_scan_compensation(
            x, y, t, p,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            bin_width=args.bin_width,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            debug=args.debug,
            smoothness_weight=args.smoothness_weight,
            a_fixed=args.a_fixed,
            b_fixed=args.b_fixed,
            boundary_trainable=args.boundary_trainable,
            a_default=args.a_default,
            b_default=args.b_default,
            num_params=args.num_params,
            temperature=args.temperature,
            chunk_size=args.chunk_size
        )
    
    # Print final results
    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
    
    a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
    boundary_status = "TRAINED" if model.compensate.boundary_trainable else "FIXED"
    
    print(f"\n🎯 Final Results (Chunked Processing):")
    print(f"  Processed: {len(x):,} events (chunk size: {args.chunk_size:,})")
    print(f"  a_params ({a_status}): [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
    print(f"  b_params ({b_status}): [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
    print(f"  boundaries ({boundary_status})")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Count trainable parameters
    trainable_params = 0
    if not model.compensate.a_fixed:
        trainable_params += len(final_a_params)
    if not model.compensate.b_fixed:
        trainable_params += len(final_b_params)
    if model.compensate.boundary_trainable:
        trainable_params += len(final_a_params)
    total_params = len(final_a_params) * (3 if model.compensate.boundary_trainable else 2)
    
    print(f"  Trainable parameters: {trainable_params}/{total_params}")
    print(f"  Config: {args.num_params} params, temp={args.temperature:.0f}, chunk={args.chunk_size:,}")
    
    if args.debug:
        print(f"  Detailed a_params: {final_a_params}")
        print(f"  Detailed b_params: {final_b_params}")
        if model.compensate.boundary_trainable:
            boundary_offsets = model.compensate.boundary_offsets.detach().cpu().numpy()
            print(f"  Detailed boundary_offsets: {boundary_offsets}")
    
    # Save learned parameters if requested and if training was performed
    if args.save_params and args.load_params is None:
        print(f"\n💾 SAVING LEARNED PARAMETERS")
        print("="*50)
        param_file, json_file, csv_file = save_learned_parameters(model, args.output_dir, base_name, duration.item(), debug=args.debug)
        print(f"   💡 To reuse these parameters later, use: --load_params {param_file}")
        print(f"   📊 CSV file can be opened in Excel: {csv_file}")
    
    # Save results
    save_results(model, losses, a_params_history, b_params_history, args.output_dir, base_name)
    
    # Plot learned parameters if requested
    if args.plot_params:
        plot_learned_parameters_with_data(model, x, y, t, args.sensor_width, args.sensor_height, 
                                         args.output_dir, base_name)
    
    # Create time-binned frame comparison plots if requested
    if args.save_frames:
        create_time_binned_frame_comparison(model, x, y, t, p, args.sensor_height, args.sensor_width, 
                                           args.bin_width, original_t_start, original_t_end,
                                           args.output_dir, base_name, chunk_size=args.chunk_size, debug=args.debug)
    
    # Visualize if requested
    if args.visualize:
        visualize_results(model, x, y, t, p, losses, variance_losses, smoothness_losses, a_params_history, b_params_history,
                         args.bin_width, args.sensor_width, args.sensor_height, 
                         original_t_start, original_t_end, args.output_dir, base_name)
    
    print("\n✅ Multi-Window Scan compensation complete (CHUNKED PROCESSING - UNIFIED VARIANCE)!")
    if args.load_params:
        print("🔄 Used pre-trained parameters - no optimization performed!")
    else:
        print("🚀 All events processed with chunked processing for memory efficiency and unified variance for proper learning!")
    if args.save_frames:
        print("📸 Time-binned frame comparisons saved to time_binned_frames/ subfolder!")
    if not args.debug:
        print("💡 Use --load_params <file.npz/.json/.csv> to reuse saved parameters and skip training next time!")
        print("📊 CSV files can be opened in Excel for easy parameter inspection and editing!")
        print("🖼️  Time-binned frame plots show temporal evolution of compensation across scan duration!")

if __name__ == "__main__":
    main()