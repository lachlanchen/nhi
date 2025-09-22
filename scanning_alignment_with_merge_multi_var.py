#!/usr/bin/env python3
"""
Enhanced scan compensation code for NPZ event files with Multi-Window Compensation
Replaces simple linear compensation with our advanced Compensate class
Added support for fixing a_params or b_params during training
Fixed: Skip training when no parameters are trainable
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import glob
from matplotlib.gridspec import GridSpec

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import our Compensate class (copy from the optimized 3D code)
class Compensate(nn.Module):
    def __init__(self, a_params, b_params, duration, temperature=5000, device='cpu', a_fixed=True, b_fixed=False):
        """
        3D Multi-window compensation class with PyTorch and gradients
        Added support for fixing parameters during training
        """
        super(Compensate, self).__init__()
        
        self.device = device
        self.duration = float(duration)
        self.temperature = float(temperature)
        self.a_fixed = a_fixed
        self.b_fixed = b_fixed
        
        # Convert parameters to PyTorch tensors with conditional gradient tracking
        if a_fixed:
            # Fixed parameters - no gradients
            self.a_params = torch.tensor(a_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('a_params_buffer', self.a_params)
            print(f"  a_params: FIXED at {a_params[0]:.3f} (no gradients)")
        else:
            # Trainable parameters - with gradients
            self.a_params = nn.Parameter(torch.tensor(a_params, dtype=torch.float32, device=device))
            print(f"  a_params: TRAINABLE (with gradients)")
        
        if b_fixed:
            # Fixed parameters - no gradients
            self.b_params = torch.tensor(b_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('b_params_buffer', self.b_params)
            print(f"  b_params: FIXED at {b_params[0]:.3f} (no gradients)")
        else:
            # Trainable parameters - with gradients
            self.b_params = nn.Parameter(torch.tensor(b_params, dtype=torch.float32, device=device))
            print(f"  b_params: TRAINABLE (with gradients)")
        
        # Validate input
        if len(a_params) != len(b_params):
            raise ValueError("a_params and b_params must have the same length")
        
        # Calculate window structure
        self.num_boundaries = len(a_params)
        self.num_main_windows = self.num_boundaries - 3  # n_main = len(ax) - 3
        self.num_total_windows = self.num_main_windows + 2  # Add 2 edge windows
        
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")
        
        # Calculate main window size based on main windows only (in microseconds)
        self.main_window_size = self.duration / self.num_main_windows
        
        # Calculate boundary offsets as fixed tensor (no gradients needed)
        # These offsets define the window positions and should NOT be learned
        boundary_offsets = torch.tensor([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ], dtype=torch.float32, device=device, requires_grad=False)
        
        # Register as buffer to ensure it's NOT a parameter
        self.register_buffer('boundary_offsets', boundary_offsets)
        
        # Minimal initialization output
        if self.device == 'cuda:0':
            device_str = "GPU"
        else:
            device_str = "CPU"
        
        # Count trainable parameters
        trainable_params = 0
        if not a_fixed:
            trainable_params += len(a_params)
        if not b_fixed:
            trainable_params += len(b_params)
            
        print(f"  Compensate ready: {self.num_main_windows} main windows, {device_str}, temp={self.temperature:.0f}μs")
        print(f"  Trainable parameters: {trainable_params}/{len(a_params) + len(b_params)}")
    
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
    
    def compute_window_memberships(self, x, y, t_shifted):
        """
        Compute soft membership for each window - OPTIMIZED VERSION
        """
        boundary_values = self.get_boundary_surfaces(x, y)
        
        # Convert t_shifted to tensor
        if not isinstance(t_shifted, torch.Tensor):
            t_shifted = torch.tensor(t_shifted, dtype=torch.float32, device=self.device)
        
        # Convert spatial coordinates to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            
        X, Y, T = torch.meshgrid(x, y, t_shifted, indexing='ij')
        
        # OPTIMIZED: Vectorized sigmoid computation using tensor shifting
        # Get lower and upper boundaries for all windows at once
        lower_bounds = boundary_values[:-1, :, :, None]  # [num_total_windows, len_x, len_y, 1]
        upper_bounds = boundary_values[1:, :, :, None]   # [num_total_windows, len_x, len_y, 1]
        
        # Compute sigmoids for all windows simultaneously
        lower_sigmoids = torch.sigmoid((T[None, :, :, :] - lower_bounds) / self.temperature)
        upper_sigmoids = torch.sigmoid((upper_bounds - T[None, :, :, :]) / self.temperature)
        
        # Element-wise multiplication for all windows
        memberships = lower_sigmoids * upper_sigmoids  # [num_total_windows, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = torch.sum(memberships, dim=0, keepdim=True)  # [1, len_x, len_y, len_t]
        memberships_sum = torch.clamp(memberships_sum, min=1e-8)
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t_shifted):
        """
        Interpolate parameters within each window - OPTIMIZED VERSION
        """
        boundary_values = self.get_boundary_surfaces(x, y)
        
        # Convert to tensors
        if not isinstance(t_shifted, torch.Tensor):
            t_shifted = torch.tensor(t_shifted, dtype=torch.float32, device=self.device)
        
        # Convert spatial coordinates to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            
        X, Y, T = torch.meshgrid(x, y, t_shifted, indexing='ij')
        
        # OPTIMIZED: Vectorized interpolation
        lower_bounds = boundary_values[:-1, :, :, None]  # [num_total_windows, len_x, len_y, 1]
        upper_bounds = boundary_values[1:, :, :, None]   # [num_total_windows, len_x, len_y, 1]
        
        window_widths = upper_bounds - lower_bounds
        window_widths = torch.clamp(window_widths, min=1e-8)
        
        # Interpolation parameter (no clamping)
        alpha = (T[None, :, :, :] - lower_bounds) / window_widths
        
        # Get current parameters
        a_params = self.get_a_params()
        b_params = self.get_b_params()
        
        # Vectorized parameter interpolation
        a_lower = a_params[:-1, None, None, None]  # [num_total_windows, 1, 1, 1]
        a_upper = a_params[1:, None, None, None]   # [num_total_windows, 1, 1, 1]
        b_lower = b_params[:-1, None, None, None]  # [num_total_windows, 1, 1, 1]
        b_upper = b_params[1:, None, None, None]   # [num_total_windows, 1, 1, 1]
        
        interpolated_slopes_a = (1 - alpha) * a_lower + alpha * a_upper
        interpolated_slopes_b = (1 - alpha) * b_lower + alpha * b_upper
        # interpolated_slopes_a = a_lower
        # interpolated_slopes_b = b_lower
        
        return interpolated_slopes_a, interpolated_slopes_b
    
    def forward(self, x, y, t):
        """
        Compute compensation for given (x, y, t) coordinates
        Uses ultra-small batching to prevent memory overflow
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
        
        # Use ultra-small batches to manage computation graph size
        num_events = len(x)
        
        if num_events <= 5000:  # 5K events - process at once
            return self.compute_event_compensation(x, y, t_shifted)
        else:
            # Use ultra-small batches to minimize memory footprint
            batch_size = 5000  # Ultra-small batches
            compensations = []
            
            for i in range(0, num_events, batch_size):
                end_idx = min(i + batch_size, num_events)
                x_batch = x[i:end_idx]
                y_batch = y[i:end_idx]
                t_batch = t_shifted[i:end_idx]
                
                # Process ultra-small batch
                compensation_batch = self.compute_event_compensation(x_batch, y_batch, t_batch)
                compensations.append(compensation_batch)
                
                # Very frequent memory cleanup
                if (i // batch_size) % 10 == 0:  # Every 10 batches (50K events)
                    torch.cuda.empty_cache()
            
            # Concatenate results
            compensation = torch.cat(compensations, dim=0)
            
            # Final cleanup
            torch.cuda.empty_cache()
            return compensation
    
    def compute_event_compensation(self, x, y, t_shifted):
        """
        Compute compensation directly for event coordinates (no meshgrids)
        """
        # Get current parameters
        a_params = self.get_a_params()
        b_params = self.get_b_params()
        
        # Compute boundary values for each event
        # boundary_i = a_i * x + b_i * y + offset_i
        num_events = len(x)
        
        # Expand parameters for broadcasting: [num_boundaries, num_events]
        a_expanded = a_params[:, None].expand(-1, num_events)  # [13, num_events]
        b_expanded = b_params[:, None].expand(-1, num_events)  # [13, num_events]
        offset_expanded = self.boundary_offsets[:, None].expand(-1, num_events)  # [13, num_events]
        
        # Expand coordinates for broadcasting: [1, num_events]
        x_expanded = x[None, :].expand(self.num_boundaries, -1)  # [13, num_events]
        y_expanded = y[None, :].expand(self.num_boundaries, -1)  # [13, num_events]
        
        # Compute boundary values: [num_boundaries, num_events]
        boundary_values = a_expanded * x_expanded + b_expanded * y_expanded + offset_expanded
        
        # Compute window memberships for each event
        t_expanded = t_shifted[None, :].expand(self.num_total_windows, -1)  # [12, num_events]
        
        # Get boundaries for each window: [num_total_windows, num_events]
        lower_bounds = boundary_values[:-1, :]  # [12, num_events]
        upper_bounds = boundary_values[1:, :]   # [12, num_events]
        
        # Compute sigmoid memberships
        lower_sigmoids = torch.sigmoid((t_expanded - lower_bounds) / self.temperature)
        upper_sigmoids = torch.sigmoid((upper_bounds - t_expanded) / self.temperature)
        memberships = lower_sigmoids * upper_sigmoids  # [12, num_events]
        
        # Normalize memberships
        memberships_sum = torch.sum(memberships, dim=0, keepdim=True)  # [1, num_events]
        memberships_sum = torch.clamp(memberships_sum, min=1e-8)
        memberships = memberships / memberships_sum  # [12, num_events]
        
        # Compute interpolation weights for each window
        window_widths = upper_bounds - lower_bounds  # [12, num_events]
        window_widths = torch.clamp(window_widths, min=1e-8)
        alpha = (t_expanded - lower_bounds) / window_widths  # [12, num_events]
        
        # Interpolate parameters for each window: [12, num_events]
        a_lower = a_params[:-1, None].expand(-1, num_events)  # [12, num_events]
        a_upper = a_params[1:, None].expand(-1, num_events)   # [12, num_events]
        b_lower = b_params[:-1, None].expand(-1, num_events)  # [12, num_events]
        b_upper = b_params[1:, None].expand(-1, num_events)   # [12, num_events]
        
        slopes_a = (1 - alpha) * a_lower + alpha * a_upper  # [12, num_events]
        slopes_b = (1 - alpha) * b_lower + alpha * b_upper  # [12, num_events]
        
        # Compute weighted compensation for each event
        x_contrib = memberships * slopes_a * x[None, :]  # [12, num_events]
        y_contrib = memberships * slopes_b * y[None, :]  # [12, num_events]
        
        # Sum over windows: [num_events]
        compensation_x = torch.sum(x_contrib, dim=0)
        compensation_y = torch.sum(y_contrib, dim=0)
        compensation = compensation_x + compensation_y
        
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

def load_npz_events(npz_path):
    """
    Load events from NPZ file
    Expected format: x, y, t, p arrays
    """
    print(f"Loading events from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Extract arrays
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)  # Should already be in microseconds
    p = data['p'].astype(np.float32)
    
    print(f"✓ Loaded {len(x):,} events")
    print(f"  Time: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f}s)")
    print(f"  Spatial: X=[{x.min():.0f}, {x.max():.0f}], Y=[{y.min():.0f}, {y.max():.0f}]")
    
    # Convert polarity to [-1, 1] if it's [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        print(f"  Converted polarity from [0,1] to [-1,1]")
    
    return x, y, t, p

def load_and_merge_segments(segments_folder):
    """
    Load and merge all scan segments with proper time normalization and polarity handling
    """
    print("\n" + "="*60)
    print("MERGING SCAN SEGMENTS")
    print("="*60)
    print(f"Loading from: {segments_folder}")
    
    # Find all segment files
    segment_files = glob.glob(os.path.join(segments_folder, "Scan_*_events.npz"))
    segment_files.sort()  # Ensure consistent ordering
    
    if not segment_files:
        raise FileNotFoundError(f"No segment files found in {segments_folder}")
    
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
    print(f"  Spatial: X=[{merged_x.min():.0f}, {merged_x.max():.0f}], Y=[{merged_y.min():.0f}, {merged_y.max():.0f}]")
    print(f"  Temporal: [{merged_t.min():.0f}, {merged_t.max():.0f}] μs")
    
    return merged_x, merged_y, merged_t, merged_p

class ScanCompensation(nn.Module):
    def __init__(self, duration, device='cuda', a_fixed=True, b_fixed=False, a_default=0.0, b_default=-76.0):
        super().__init__()
        
        # Initialize parameters based on defaults and fixed flags
        print(f"Initializing Multi-Window Compensation:")
        print(f"  Duration: {duration/1000:.1f} ms, Parameters: 26 total (13+13), Main windows: 10")
        print(f"  a_fixed: {a_fixed}, b_fixed: {b_fixed}")
        print(f"  a_default: {a_default}, b_default: {b_default}")
        
        # Initialize with default values
        a_params = [a_default] * 13  # Initialize to a_default
        b_params = [b_default] * 13  # Initialize to b_default
        
        # Create the compensate object - it will create its own Parameters
        self.compensate = Compensate(a_params, b_params, duration, device=device, a_fixed=a_fixed, b_fixed=b_fixed)
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps using multi-window compensation.
        """
        # Apply multi-window compensation
        compensation = self.compensate(x_coords, y_coords, timestamps)
        t_warped = timestamps - compensation
        
        return x_coords, y_coords, t_warped

    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, 
                original_t_start=None, original_t_end=None):
        """
        Process events through the model by warping them and then computing the loss.
        """
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        # Filter events to original time range if provided
        if original_t_start is not None and original_t_end is not None:
            valid_time_mask = (t_warped >= original_t_start) & (t_warped <= original_t_end)
            x_warped = x_warped[valid_time_mask]
            y_warped = y_warped[valid_time_mask]
            t_warped = t_warped[valid_time_mask]
            polarities = polarities[valid_time_mask]
            
            # Use original time range for binning
            t_start = original_t_start
            t_end = original_t_end
        else:
            # Use warped time range
            t_start = t_warped.min()
            t_end = t_warped.max()
        
        # Define time binning parameters
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        # Normalize time to [0, num_bins)
        t_norm = (t_warped - t_start) / time_bin_width

        # Compute floor and ceil indices for time bins
        t0 = torch.floor(t_norm)
        t1 = t0 + 1

        # Compute weights for linear interpolation over time
        wt = (t_norm - t0).float()  # Ensure float32

        # Clamping indices to valid range
        t0_clamped = t0.clamp(0, num_bins - 1)
        t1_clamped = t1.clamp(0, num_bins - 1)

        # Cast x and y to long for indexing
        x_indices = x_warped.long()
        y_indices = y_warped.long()

        # Ensure spatial indices are within bounds
        valid_mask = (x_indices >= 0) & (x_indices < W) & \
                     (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]

        # Compute linear indices for the event tensor
        spatial_indices = y_indices * W + x_indices
        spatial_indices = spatial_indices.long()

        # For t0
        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t0 = flat_indices_t0.long()
        weights_t0 = ((1 - wt) * polarities).float()

        # For t1
        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        flat_indices_t1 = flat_indices_t1.long()
        weights_t1 = (wt * polarities).float()

        # Combine indices and weights
        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        # Add explicit bounds checking to prevent CUDA errors
        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        # Create the flattened event tensor
        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        # Accumulate events into the flattened tensor using scatter_add
        if len(flat_indices) > 0:  # Only if we have valid indices
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        # Reshape back to (num_bins, H, W)
        event_tensor = event_tensor_flat.view(num_bins, H, W)

        # Compute the variance over x and y within each time bin
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        # Loss is the sum of variances
        loss = torch.sum(variances)

        return event_tensor, loss

def train_scan_compensation(x, y, t, p, sensor_width=1280, sensor_height=720, 
                          bin_width=1e5, num_iterations=1000, learning_rate=1.0, debug=False,
                          smoothness_weight=0.001, a_fixed=True, b_fixed=False, 
                          a_default=0.0, b_default=-76.0):
    """
    Train the multi-window scan compensation model with intelligent subsampling and smoothness regularization
    Added support for fixing parameters during training
    Fixed: Skip training when no parameters are trainable
    """
    print(f"Training multi-window scan compensation...")
    print(f"  Sensor: {sensor_width} x {sensor_height}")
    print(f"  Bin width: {bin_width/1000:.1f} ms")
    print(f"  Iterations: {num_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Smoothness weight: {smoothness_weight}")
    print(f"  Parameter settings: a_fixed={a_fixed}, b_fixed={b_fixed}")
    print(f"  Default values: a={a_default}, b={b_default}")
    
    # Intelligent subsampling for very large datasets
    max_training_events = 1000000  # 1M events max for training
    num_events = len(x)
    
    if num_events > max_training_events:
        print(f"  Large dataset: {num_events:,} events detected")
        print(f"  Subsampling to {max_training_events:,} events for efficient training...")
        
        # Create random indices for subsampling
        indices = np.random.choice(num_events, max_training_events, replace=False)
        indices = np.sort(indices)  # Keep temporal order somewhat
        
        x_train = x[indices]
        y_train = y[indices]
        t_train = t[indices]
        p_train = p[indices]
        
        print(f"  Subsampled to {len(x_train):,} events ({len(x_train)/num_events:.1%} of original)")
        if debug:
            print(f"  Subsampled time range: {t_train.min():.0f} - {t_train.max():.0f} μs")
    else:
        x_train = x
        y_train = y
        t_train = t
        p_train = p
        print(f"  Using all {num_events:,} events for training")
    
    # Convert to tensors with explicit dtype
    xs = torch.tensor(x_train, device=device, dtype=torch.float32)
    ys = torch.tensor(y_train, device=device, dtype=torch.float32)
    ts = torch.tensor(t_train, device=device, dtype=torch.float32)
    ps = torch.tensor(p_train, device=device, dtype=torch.float32)

    # Store original time range (from original data, not subsampled)
    original_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
    duration = original_t_end - original_t_start
    
    print(f"  Time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs ({duration.item()/1000:.1f} ms)")

    # Initialize multi-window compensation model with parameter fixing
    model = ScanCompensation(duration.item(), device=device, a_fixed=a_fixed, b_fixed=b_fixed, 
                            a_default=a_default, b_default=b_default)

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
                                 original_t_start, original_t_end)
            actual_loss_value = actual_loss.item()
            
            # Update loss histories with actual computed loss
            losses = [actual_loss_value] * num_iterations
            variance_losses = [actual_loss_value] * num_iterations
        
        print(f"\nFixed Parameters Evaluation:")
        print(f"  a_params (FIXED): [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
        print(f"  b_params (FIXED): [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
        print(f"  Computed loss: {actual_loss_value:.6f}")
        
        return model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start, original_t_end

    # Define the optimizer - only optimizes non-fixed parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining Progress:")
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
        
        # Compute main variance loss
        event_tensor, variance_loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                           original_t_start, original_t_end)
        
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
            print(f"  → Learning rate reduced to {optimizer.param_groups[0]['lr']:.4f}")
        elif i == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            print(f"  → Learning rate reduced to {optimizer.param_groups[0]['lr']:.4f}")

    print("-" * 90)
    print(f"Training completed!")
    print(f"  Final total loss: {losses[-1]:.6f}")
    print(f"  Final variance loss: {variance_losses[-1]:.6f}")
    print(f"  Final smoothness loss: {smoothness_losses[-1]:.6f}")
    
    return model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start, original_t_end

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
        boundary_offsets = model.compensate.boundary_offsets.cpu().numpy()
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
    max_plot_events = 50000  # Limit events for clear visualization
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
    
    plt.suptitle('Multi-Window Scan Compensation: Learned Boundaries with Event Data', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, f"{filename_prefix}_learned_parameters_with_data.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Parameters with data plot saved to: {plot_path}")
    
    plt.show()
    
    # Print parameter summary
    print(f"\nLearned Parameters Summary:")
    print(f"  a_params ({a_status}): {final_a_params}")
    print(f"  b_params ({b_status}): {final_b_params}")
    print(f"  a_params range: {final_a_params.min():.6f} to {final_a_params.max():.6f}")
    print(f"  b_params range: {final_b_params.min():.6f} to {final_b_params.max():.6f}")

def create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True):
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
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
        else:
            # Temporarily set all parameters to their initialization values
            if not model.compensate.a_fixed:
                original_a_params = model.compensate.a_params.clone()
                model.compensate.a_params.data.zero_()
            if not model.compensate.b_fixed:
                original_b_params = model.compensate.b_params.clone()
                model.compensate.b_params.data.zero_()
                
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
            
            # Restore parameters
            if not model.compensate.a_fixed:
                model.compensate.a_params.data = original_a_params
            if not model.compensate.b_fixed:
                model.compensate.b_params.data = original_b_params
    
    return event_tensor

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
    return f"_multiwindow_a{a_status}_{a_params.min():.4f}_{a_params.max():.4f}_b{b_status}_{b_params.min():.4f}_{b_params.max():.4f}"

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
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
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
    axes[1, 0].plot(var_comp_list, label='Multi-Window Compensated', alpha=0.7)
    axes[1, 0].set_xlabel('Time Bin')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Variance Comparison')
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
    axes[1, 1].set_title('Multi-Window Summary')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Add overall title with parameters
    fig.suptitle(f'Multi-Window Scan Compensation Results (with Parameter Fixing)\n{param_str}', fontsize=16, y=0.98)
    
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
            f.write("MULTI-WINDOW SCAN COMPENSATION RESULTS (WITH PARAMETER FIXING)\n")
            f.write("=" * 60 + "\n\n")
            
            a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
            b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
            
            f.write(f"Parameter Status:\n")
            f.write(f"  a_params: {a_status}\n")
            f.write(f"  b_params: {b_status}\n\n")
            
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
    parser = argparse.ArgumentParser(description='Multi-Window Scan compensation for NPZ event files with parameter fixing')
    parser.add_argument('input_path', help='Path to NPZ event file OR segments folder (when using --merge)')
    parser.add_argument('--merge', action='store_true', help='Merge all scan segments from folder instead of processing single file')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--smoothness_weight', type=float, default=0.001, help='Weight for smoothness regularization (default: 0.001)')
    
    # New parameter fixing arguments
    parser.add_argument('--a_fixed', action='store_true', default=True, help='Fix a_params during training (default: True)')
    parser.add_argument('--a_trainable', dest='a_fixed', action='store_false', help='Make a_params trainable during training')
    parser.add_argument('--b_fixed', action='store_true', help='Fix b_params during training (default: False)')
    parser.add_argument('--b_trainable', dest='b_fixed', action='store_false', default=True, help='Make b_params trainable during training (default)')
    parser.add_argument('--a_default', type=float, default=0.0, help='Default value for a_params (default: 0.0)')
    parser.add_argument('--b_default', type=float, default=-76.0, help='Default value for b_params (default: -76.0)')
    
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--plot_params', action='store_true', help='Plot learned parameters in X-T and Y-T planes')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
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
        base_name = f"{folder_name}_merged"
        
        print(f"Merging segments from: {segments_folder}")
        
        # Load and merge all segments
        x, y, t, p = load_and_merge_segments(segments_folder)
        
    else:
        # Input is a single NPZ file
        npz_file = args.input_path
        if not os.path.isfile(npz_file):
            raise ValueError(f"NPZ file not found: {npz_file}")
        
        # Set output directory
        if args.output_dir is None:
            args.output_dir = os.path.dirname(npz_file)
        
        # Create filename prefix
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        
        print(f"Analyzing: {npz_file}")
        
        # Load events from single file
        x, y, t, p = load_npz_events(npz_file)
    
    # Train multi-window model with parameter fixing
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
        a_default=args.a_default,
        b_default=args.b_default
    )
    
    # Print final results
    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
    
    a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
    
    print(f"\nFinal Results:")
    print(f"  a_params ({a_status}): [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
    print(f"  b_params ({b_status}): [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Count trainable parameters
    trainable_params = 0
    if not model.compensate.a_fixed:
        trainable_params += len(final_a_params)
    if not model.compensate.b_fixed:
        trainable_params += len(final_b_params)
    total_params = len(final_a_params) + len(final_b_params)
    
    print(f"  Trainable parameters: {trainable_params}/{total_params}")
    
    if args.debug:
        print(f"  Detailed a_params: {final_a_params}")
        print(f"  Detailed b_params: {final_b_params}")
    
    # Save results
    save_results(model, losses, a_params_history, b_params_history, args.output_dir, base_name)
    
    # Plot learned parameters if requested
    if args.plot_params:
        plot_learned_parameters_with_data(model, x, y, t, args.sensor_width, args.sensor_height, args.output_dir, base_name)
    
    # Visualize if requested - FIXED FUNCTION CALL WITH ALL REQUIRED PARAMETERS
    if args.visualize:
        visualize_results(model, x, y, t, p, losses, variance_losses, smoothness_losses, a_params_history, b_params_history,
                         args.bin_width, args.sensor_width, args.sensor_height, 
                         original_t_start, original_t_end, args.output_dir, base_name)
    
    print("\n✓ Multi-Window Scan compensation complete!")

if __name__ == "__main__":
    main()