#!/usr/bin/env python3
"""
Multi-Window Scan Compensation for NPZ event files - 
Uses sophisticated multi-window compensation with different scanning speeds for different temporal regions
Memory-efficient implementation for very long event sequences
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

# Device configuration with memory management
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set GPU memory management for better efficiency
if torch.cuda.is_available():
    # Enable memory fraction to avoid OOM
    torch.cuda.empty_cache()
    # Set CUDA memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("GPU memory management: expandable_segments enabled")

class Compensate(nn.Module):
    """
    Multi-window compensation class adapted for scan compensation
    """
    def __init__(self, a_params, b_params, duration, temperature=5000, device='cpu'):
        super(Compensate, self).__init__()
        
        self.device = device
        self.duration = float(duration)
        self.temperature = float(temperature)
        
        # Convert parameters to PyTorch tensors with gradient tracking
        self.a_params = nn.Parameter(torch.tensor(a_params, dtype=torch.float32, device=device))
        self.b_params = nn.Parameter(torch.tensor(b_params, dtype=torch.float32, device=device))
        
        # Validate input
        if len(a_params) != len(b_params):
            raise ValueError("a_params and b_params must have the same length")
        
        # Calculate window structure
        self.num_boundaries = len(a_params)
        self.num_main_windows = self.num_boundaries - 3
        self.num_total_windows = self.num_main_windows + 2
        
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")
        
        # Calculate main window size
        self.main_window_size = self.duration / self.num_main_windows
        
        # Calculate boundary offsets as fixed tensor
        boundary_offsets = torch.tensor([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ], dtype=torch.float32, device=device)
        
        self.register_buffer('boundary_offsets', boundary_offsets)
        
        print(f"Multi-window Compensate initialized:")
        print(f"  Duration: {self.duration:.0f} μs ({self.duration/1000:.1f} ms)")
        print(f"  Parameters per direction: {self.num_boundaries}")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Main window size: {self.main_window_size:.0f} μs ({self.main_window_size/1000:.1f} ms)")
        print(f"  A parameters: {self.a_params.data.cpu().numpy()}")
        print(f"  B parameters: {self.b_params.data.cpu().numpy()}")
    
    def get_boundary_surfaces(self, x, y):
        """Compute boundary surface values: t = a_i * x + b_i * y + offset_i"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # More memory-efficient computation
        n_events = len(x)
        n_boundaries = len(self.a_params)
        
        # Initialize result tensor
        boundary_values = torch.zeros(n_boundaries, n_events, device=self.device, dtype=torch.float32)
        
        # Compute boundary values one at a time to save memory
        for i in range(n_boundaries):
            boundary_values[i, :] = (
                self.a_params[i] * x + 
                self.b_params[i] * y + 
                self.boundary_offsets[i]
            )
        
        return boundary_values
    
    def compute_compensation_efficient(self, x_coords, y_coords, t_coords):
        """
        Memory-efficient compensation computation for long event sequences
        Uses small chunks and aggressive memory management
        """
        n_events = len(x_coords)
        
        # Use much smaller chunks to avoid GPU memory issues
        if n_events > 1000000:  # For very large datasets
            chunk_size = 5000
        elif n_events > 100000:
            chunk_size = 10000  
        else:
            chunk_size = min(20000, n_events)
        
        compensation_results = []
        
        print(f"Processing {n_events:,} events in chunks of {chunk_size:,}...")
        
        for start_idx in range(0, n_events, chunk_size):
            end_idx = min(start_idx + chunk_size, n_events)
            
            # Extract chunk
            x_chunk = x_coords[start_idx:end_idx]
            y_chunk = y_coords[start_idx:end_idx]
            t_chunk = t_coords[start_idx:end_idx]
            
            # Compute compensation for this chunk with memory management
            chunk_compensation = self.forward_chunk_memory_efficient(x_chunk, y_chunk, t_chunk)
            compensation_results.append(chunk_compensation.cpu())  # Move to CPU to save GPU memory
            
            # Clear GPU cache periodically
            if (start_idx // chunk_size) % 100 == 0:
                torch.cuda.empty_cache()
                print(f"  Processed {end_idx:,}/{n_events:,} events...")
        
        # Concatenate all results on CPU first, then move to GPU
        total_compensation_cpu = torch.cat(compensation_results, dim=0)
        total_compensation = total_compensation_cpu.to(self.device)
        
        print(f"Compensation complete: {len(total_compensation):,} events")
        
        return total_compensation
    
    def forward_chunk_memory_efficient(self, x_coords, y_coords, t_coords):
        """
        Memory-efficient computation for a chunk of events with aggressive memory management
        """
        # Ensure inputs are tensors
        if not isinstance(x_coords, torch.Tensor):
            x_coords = torch.tensor(x_coords, dtype=torch.float32, device=self.device)
        if not isinstance(y_coords, torch.Tensor):
            y_coords = torch.tensor(y_coords, dtype=torch.float32, device=self.device)
        if not isinstance(t_coords, torch.Tensor):
            t_coords = torch.tensor(t_coords, dtype=torch.float32, device=self.device)
        
        n_events = len(x_coords)
        
        # If still too large, process in even smaller sub-chunks
        if n_events > 20000:
            sub_chunk_size = 5000
            sub_results = []
            
            for sub_start in range(0, n_events, sub_chunk_size):
                sub_end = min(sub_start + sub_chunk_size, n_events)
                sub_x = x_coords[sub_start:sub_end]
                sub_y = y_coords[sub_start:sub_end]
                sub_t = t_coords[sub_start:sub_end]
                
                sub_compensation = self.forward_chunk_core(sub_x, sub_y, sub_t)
                sub_results.append(sub_compensation)
                
                # Clear intermediate tensors
                del sub_x, sub_y, sub_t, sub_compensation
            
            total_compensation = torch.cat(sub_results, dim=0)
            del sub_results
            return total_compensation
        else:
            return self.forward_chunk_core(x_coords, y_coords, t_coords)
    
    def forward_chunk_core(self, x_coords, y_coords, t_coords):
        """
        Core computation for a small chunk of events
        """
        n_events = len(x_coords)
        
        # Get boundary values for all events
        boundary_values = self.get_boundary_surfaces(x_coords, y_coords)  # [n_boundaries, n_events]
        
        # Compute window memberships efficiently
        lower_bounds = boundary_values[:-1, :]  # [n_windows, n_events]
        upper_bounds = boundary_values[1:, :]   # [n_windows, n_events]
        
        # Clear boundary_values to save memory
        del boundary_values
        
        # Broadcast t_coords to match boundary shapes
        t_expanded = t_coords[None, :].expand(self.num_total_windows, -1)  # [n_windows, n_events]
        
        # Compute sigmoid memberships
        lower_sigmoids = torch.sigmoid((t_expanded - lower_bounds) / self.temperature)
        upper_sigmoids = torch.sigmoid((upper_bounds - t_expanded) / self.temperature)
        memberships = lower_sigmoids * upper_sigmoids  # [n_windows, n_events]
        
        # Clear intermediate tensors
        del lower_sigmoids, upper_sigmoids
        
        # Normalize memberships
        memberships_sum = torch.sum(memberships, dim=0, keepdim=True)  # [1, n_events]
        memberships_sum = torch.clamp(memberships_sum, min=1e-8)
        memberships = memberships / memberships_sum  # [n_windows, n_events]
        del memberships_sum
        
        # Compute within-window interpolation
        window_widths = upper_bounds - lower_bounds
        window_widths = torch.clamp(window_widths, min=1e-8)
        alpha = (t_expanded - lower_bounds) / window_widths  # [n_windows, n_events]
        
        # Clear tensors no longer needed
        del upper_bounds, lower_bounds, window_widths, t_expanded
        
        # Interpolate parameters
        a_lower = self.a_params[:-1, None]  # [n_windows, 1]
        a_upper = self.a_params[1:, None]   # [n_windows, 1]
        b_lower = self.b_params[:-1, None]  # [n_windows, 1]
        b_upper = self.b_params[1:, None]   # [n_windows, 1]
        
        interpolated_a = (1 - alpha) * a_lower + alpha * a_upper  # [n_windows, n_events]
        interpolated_b = (1 - alpha) * b_lower + alpha * b_upper  # [n_windows, n_events]
        
        # Clear alpha and parameter tensors
        del alpha, a_lower, a_upper, b_lower, b_upper
        
        # Compute compensation for each window
        x_expanded = x_coords[None, :].expand(self.num_total_windows, -1)  # [n_windows, n_events]
        y_expanded = y_coords[None, :].expand(self.num_total_windows, -1)  # [n_windows, n_events]
        
        compensation_x = memberships * interpolated_a * x_expanded  # [n_windows, n_events]
        compensation_y = memberships * interpolated_b * y_expanded  # [n_windows, n_events]
        
        # Clear intermediate tensors
        del memberships, interpolated_a, interpolated_b, x_expanded, y_expanded
        
        # Sum across windows for final compensation
        total_compensation = torch.sum(compensation_x + compensation_y, dim=0)  # [n_events]
        
        # Clear final intermediate tensors
        del compensation_x, compensation_y
        
        return total_compensation
    
    def forward(self, x_coords, y_coords, t_coords):
        """
        Main forward pass - routes to efficient chunk processing
        """
        return self.compute_compensation_efficient(x_coords, y_coords, t_coords)


class MultiWindowScanCompensation(nn.Module):
    """
    Multi-window scan compensation model using the Compensate class
    """
    def __init__(self, a_params, b_params, duration, temperature=5000):
        super().__init__()
        
        # Create the multi-window compensator
        self.compensator = Compensate(a_params, b_params, duration, temperature, device)
        
        # Store parameters for easy access
        self.duration = duration
        self.num_windows = len(a_params) - 3
        
        print(f"MultiWindowScanCompensation initialized:")
        print(f"  Duration: {duration/1000:.1f} ms")
        print(f"  Number of main windows: {self.num_windows}")
        print(f"  Total parameters: {len(a_params)} a_params, {len(b_params)} b_params")
    
    def get_all_parameters(self):
        """Get all trainable parameters as a flat list"""
        params = []
        params.extend(self.compensator.a_params.tolist())
        params.extend(self.compensator.b_params.tolist())
        return params
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Warp timestamps using multi-window compensation
        t' = t - compensate(t, x, y)
        """
        # Shift time to start from 0
        t_min = torch.min(timestamps)
        t_shifted = timestamps - t_min
        
        # Compute compensation
        compensation = self.compensator(x_coords, y_coords, t_shifted)
        
        # Apply compensation: t' = t - compensation
        t_warped = timestamps - compensation
        
        print(f"Warp statistics:")
        print(f"  Original time range: {timestamps.min():.0f} - {timestamps.max():.0f} μs")
        print(f"  Compensation range: {compensation.min():.3f} - {compensation.max():.3f} μs")
        print(f"  Warped time range: {t_warped.min():.0f} - {t_warped.max():.0f} μs")
        
        return x_coords, y_coords, t_warped

    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, 
                original_t_start=None, original_t_end=None):
        """
        Process events through the multi-window model
        """
        # Warp the timestamps
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        # Filter events to original time range if provided
        if original_t_start is not None and original_t_end is not None:
            valid_time_mask = (t_warped >= original_t_start) & (t_warped <= original_t_end)
            x_warped = x_warped[valid_time_mask]
            y_warped = y_warped[valid_time_mask]
            t_warped = t_warped[valid_time_mask]
            polarities = polarities[valid_time_mask]
            
            t_start = original_t_start
            t_end = original_t_end
        else:
            t_start = t_warped.min()
            t_end = t_warped.max()
        
        # Create event tensor (same as original code)
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        # Normalize time to [0, num_bins)
        t_norm = (t_warped - t_start) / time_bin_width

        # Compute floor and ceil indices for time bins
        t0 = torch.floor(t_norm)
        t1 = t0 + 1

        # Compute weights for linear interpolation over time
        wt = (t_norm - t0).float()

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

        # Add explicit bounds checking
        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        # Create the flattened event tensor
        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        # Accumulate events
        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        # Reshape back to (num_bins, H, W)
        event_tensor = event_tensor_flat.view(num_bins, H, W)

        # Compute the variance loss
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        loss = torch.sum(variances)

        return event_tensor, loss


def create_initial_parameters(duration, num_main_windows=5, base_ax=0.0, base_ay=-76.0):
    """
    Create initial parameters for multi-window compensation
    Based on your successful single-window results (ax=0, ay=-76)
    """
    num_boundaries = num_main_windows + 3
    
    # Create parameter variations around your successful values
    # Small variations to allow for different scanning speeds in different windows
    ax_variation = 2.0  # Small variation around 0
    ay_variation = 10.0  # Variation around -76
    
    # Create a_params (ax values) with small variations around 0
    a_params = np.random.normal(base_ax, ax_variation, num_boundaries).astype(np.float32)
    
    # Create b_params (ay values) with variations around -76
    b_params = np.random.normal(base_ay, ay_variation, num_boundaries).astype(np.float32)
    
    print(f"Created initial parameters for {num_main_windows} main windows:")
    print(f"  A parameters (ax): {a_params}")
    print(f"  B parameters (ay): {b_params}")
    print(f"  Duration: {duration/1000:.1f} ms")
    
    return a_params.tolist(), b_params.tolist()


def load_npz_events(npz_path):
    """Load events from NPZ file"""
    print(f"Loading events from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    print(f"Available keys in NPZ file: {list(data.keys())}")
    
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)
    
    print(f"Loaded {len(x)} events")
    print(f"Time range: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f} seconds)")
    print(f"X range: {x.min():.0f} - {x.max():.0f}")
    print(f"Y range: {y.min():.0f} - {y.max():.0f}")
    
    # Convert polarity to [-1, 1] if needed
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        print("Converted polarity from [0,1] to [-1,1]")
    
    return x, y, t, p


def load_and_merge_segments(segments_folder):
    """Load and merge all scan segments (same as original)"""
    print(f"\n{'='*60}")
    print("MERGING SCAN SEGMENTS")
    print(f"{'='*60}")
    print(f"Loading segments from: {segments_folder}")
    
    segment_files = glob.glob(os.path.join(segments_folder, "Scan_*_events.npz"))
    segment_files.sort()
    
    if not segment_files:
        raise FileNotFoundError(f"No segment files found in {segments_folder}")
    
    print(f"Found {len(segment_files)} segment files:")
    for f in segment_files:
        print(f"  {os.path.basename(f)}")
    
    all_x, all_y, all_t, all_p = [], [], [], []
    
    for segment_file in segment_files:
        print(f"\nProcessing: {os.path.basename(segment_file)}")
        
        data = np.load(segment_file)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        t = data['t'].astype(np.float32)
        p = data['p'].astype(np.float32)
        
        start_time = float(data['start_time'])
        duration = float(data['duration_us'])
        direction = str(data['direction'])
        
        print(f"  Direction: {direction}")
        print(f"  Events: {len(x):,}")
        print(f"  Duration: {duration:.0f} μs ({duration/1000:.1f} ms)")
        
        # Normalize time
        t_normalized = t - start_time
        
        # Handle backward scans
        if 'Backward' in direction:
            t_normalized = duration - t_normalized
            p = -p
        
        # Convert polarity if needed
        if p.min() >= 0 and p.max() <= 1:
            p = (p - 0.5) * 2
        
        all_x.append(x)
        all_y.append(y)
        all_t.append(t_normalized)
        all_p.append(p)
    
    # Merge and sort
    merged_x = np.concatenate(all_x)
    merged_y = np.concatenate(all_y)
    merged_t = np.concatenate(all_t)
    merged_p = np.concatenate(all_p)
    
    sort_indices = np.argsort(merged_t)
    merged_x = merged_x[sort_indices]
    merged_y = merged_y[sort_indices]
    merged_t = merged_t[sort_indices]
    merged_p = merged_p[sort_indices]
    
    print(f"\nMerged {len(merged_x):,} events")
    print(f"Final time range: {merged_t.min():.0f} - {merged_t.max():.0f} μs")
    print(f"Total duration: {(merged_t.max() - merged_t.min())/1000:.1f} ms")
    
    return merged_x, merged_y, merged_t, merged_p


def train_multi_window_scan_compensation(x, y, t, p, sensor_width=1280, sensor_height=720, 
                                       bin_width=1e5, num_iterations=1000, learning_rate=0.1,
                                       num_main_windows=5, base_ax=0.0, base_ay=-76.0):
    """
    Train multi-window scan compensation model with memory management
    """
    print(f"Training multi-window scan compensation...")
    print(f"Sensor size: {sensor_width} x {sensor_height}")
    print(f"Bin width: {bin_width/1000:.1f} ms")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Main windows: {num_main_windows}")
    
    # Memory management for large datasets
    n_events = len(x)
    if n_events > 10000000:  # More than 10M events
        print(f"Large dataset detected ({n_events:,} events)")
        print("Using aggressive memory management...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to tensors
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    # Store original time range
    original_t_start = torch.tensor(float(ts.min().item()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(ts.max().item()), device=device, dtype=torch.float32)
    duration = float(ts.max().item() - ts.min().item())
    
    print(f"Original time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs")
    print(f"Duration: {duration:.0f} μs ({duration/1000:.1f} ms)")

    # Create initial parameters
    a_params, b_params = create_initial_parameters(duration, num_main_windows, base_ax, base_ay)
    
    # Create model
    model = MultiWindowScanCompensation(a_params, b_params, duration, temperature=5000)
    
    # Create optimizer with lower learning rate for large number of parameters
    actual_lr = learning_rate / (num_main_windows / 5.0)  # Scale down for more windows
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
    
    # Training loop
    losses = []
    params_history = []

    print(f"\nStarting training with {len(model.get_all_parameters())} parameters...")
    print(f"Adjusted learning rate: {actual_lr:.4f}")
    
    for i in range(num_iterations):
        try:
            optimizer.zero_grad()
            
            # Clear GPU cache every 10 iterations for large datasets
            if i % 10 == 0 and torch.cuda.is_available() and n_events > 5000000:
                torch.cuda.empty_cache()
            
            # Forward pass
            event_tensor, loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                     original_t_start, original_t_end)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability with many parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record progress
            current_loss = loss.item()
            losses.append(current_loss)
            current_params = model.get_all_parameters()
            params_history.append(current_params.copy())
            
            if i % 100 == 0:
                a_params_current = model.compensator.a_params.detach().cpu().numpy()
                b_params_current = model.compensator.b_params.detach().cpu().numpy()
                print(f"Iteration {i}, Loss: {current_loss:.6f}")
                print(f"  A params mean: {np.mean(a_params_current):.3f} (std: {np.std(a_params_current):.3f})")
                print(f"  B params mean: {np.mean(b_params_current):.3f} (std: {np.std(b_params_current):.3f})")
                
                # Memory status
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"  GPU memory: {memory_allocated:.2f} GB")
            
            # Clear intermediate tensors
            del event_tensor, loss
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU OOM at iteration {i}: {e}")
            print("Clearing cache and reducing chunk size...")
            torch.cuda.empty_cache()
            
            # Reduce chunk size in compensator
            if hasattr(model.compensator, 'max_chunk_size'):
                model.compensator.max_chunk_size //= 2
                print(f"Reduced chunk size to {model.compensator.max_chunk_size}")
            
            # Try again with cleared memory
            continue
            
        # Learning rate schedule
        if i == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            print("Reduced learning rate by 50%")
        elif i == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.2
            print("Reduced learning rate by 80%")

    return model, losses, params_history, original_t_start, original_t_end


def get_param_string(model):
    """Get formatted parameter string for plots"""
    a_params = model.compensator.a_params.detach().cpu().numpy()
    b_params = model.compensator.b_params.detach().cpu().numpy()
    
    a_str = f"ax=[{', '.join([f'{x:.2f}' for x in a_params])}]"
    b_str = f"ay=[{', '.join([f'{x:.2f}' for x in b_params])}]"
    
    return f"{a_str}, {b_str}"


def get_param_suffix(model):
    """Get filename-safe suffix"""
    a_params = model.compensator.a_params.detach().cpu().numpy()
    b_params = model.compensator.b_params.detach().cpu().numpy()
    
    a_mean = np.mean(a_params)
    b_mean = np.mean(b_params)
    
    return f"_multiwin_ax{a_mean:.2f}_ay{b_mean:.2f}_{len(a_params)}param"


def create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True):
    """Create event frames with or without compensation"""
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        if compensated:
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
        else:
            # Create temporary model with zero compensation
            duration = float(ts.max().item() - ts.min().item())
            num_boundaries = len(model.compensator.a_params)
            zero_a = [0.0] * num_boundaries
            zero_b = [0.0] * num_boundaries
            
            temp_model = MultiWindowScanCompensation(zero_a, zero_b, duration, temperature=5000)
            event_tensor, _ = temp_model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
    
    return event_tensor


def save_results(model, losses, params_history, output_dir, filename_prefix):
    """Save training results"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        param_suffix = get_param_suffix(model)
        
        # Get final parameters
        a_params_final = model.compensator.a_params.detach().cpu().numpy()
        b_params_final = model.compensator.b_params.detach().cpu().numpy()
        
        results_path = os.path.join(output_dir, f"{filename_prefix}_multi_window_results{param_suffix}.txt")
        
        with open(results_path, 'w') as f:
            f.write("MULTI-WINDOW SCAN COMPENSATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of main windows: {model.num_windows}\n")
            f.write(f"Total boundaries: {len(a_params_final)}\n")
            f.write(f"Duration: {model.duration/1000:.1f} ms\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Training iterations: {len(losses)}\n\n")
            
            f.write("Final A parameters (ax):\n")
            for i, param in enumerate(a_params_final):
                f.write(f"  a_{i}: {param:.6f}\n")
            
            f.write("\nFinal B parameters (ay):\n")
            for i, param in enumerate(b_params_final):
                f.write(f"  b_{i}: {param:.6f}\n")
            
            f.write(f"\nMean ax: {np.mean(a_params_final):.6f}\n")
            f.write(f"Mean ay: {np.mean(b_params_final):.6f}\n")
        
        print(f"Results saved to: {results_path}")
        
        # Save parameters as numpy arrays
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_a_params{param_suffix}.npy"), a_params_final)
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_b_params{param_suffix}.npy"), b_params_final)
        np.save(os.path.join(output_dir, f"{filename_prefix}_loss_history{param_suffix}.npy"), np.array(losses))


def visualize_results(model, x, y, t, p, losses, params_history, bin_width, 
                     sensor_width, sensor_height, original_t_start, original_t_end, 
                     output_dir=None, filename_prefix=""):
    """Visualize multi-window training results"""
    param_str = get_param_string(model)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Loss plot
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Parameters evolution
    params_array = np.array(params_history)
    num_a_params = len(model.compensator.a_params)
    
    # Plot A parameters
    for i in range(num_a_params):
        axes[0, 1].plot(params_array[:, i], label=f'a_{i}', alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('A Parameter Value')
    axes[0, 1].set_title('A Parameters Evolution')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True)
    
    # Plot B parameters  
    for i in range(num_a_params):
        axes[1, 1].plot(params_array[:, num_a_params + i], label=f'b_{i}', alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('B Parameter Value')
    axes[1, 1].set_title('B Parameters Evolution')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True)
    
    # Generate event frames
    event_tensor_orig = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
                                           original_t_start, original_t_end, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
                                           original_t_start, original_t_end, compensated=True)
    
    # Select middle frame
    bin_idx = event_tensor_orig.shape[0] // 2
    
    # Original frame
    frame_orig = event_tensor_orig[bin_idx].detach().cpu().numpy()
    im1 = axes[0, 2].imshow(frame_orig, cmap='inferno', aspect='auto')
    axes[0, 2].set_title(f'Original - Bin {bin_idx}')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Compensated frame
    frame_comp = event_tensor_comp[bin_idx].detach().cpu().numpy()
    im2 = axes[1, 2].imshow(frame_comp, cmap='inferno', aspect='auto')
    axes[1, 2].set_title(f'Compensated - Bin {bin_idx}')
    plt.colorbar(im2, ax=axes[1, 2])
    
    # Variance comparison
    with torch.no_grad():
        if event_tensor_orig.shape != event_tensor_comp.shape:
            min_bins = min(event_tensor_orig.shape[0], event_tensor_comp.shape[0])
            event_tensor_orig = event_tensor_orig[:min_bins]
            event_tensor_comp = event_tensor_comp[:min_bins]
        
        current_num_bins, H, W = event_tensor_orig.shape
        var_orig_tensor = torch.var(event_tensor_orig.reshape(current_num_bins, H * W), dim=1)
        var_comp_tensor = torch.var(event_tensor_comp.reshape(current_num_bins, H * W), dim=1)
        
        var_orig_list = var_orig_tensor.cpu().tolist()
        var_comp_list = var_comp_tensor.cpu().tolist()
        
        var_orig_mean = var_orig_tensor.mean().item()
        var_comp_mean = var_comp_tensor.mean().item()
    
    axes[1, 0].plot(var_orig_list, label='Original', alpha=0.7)
    axes[1, 0].plot(var_comp_list, label='Compensated', alpha=0.7)
    axes[1, 0].set_xlabel('Time Bin')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Variance Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary statistics
    a_params_final = model.compensator.a_params.detach().cpu().numpy()
    b_params_final = model.compensator.b_params.detach().cpu().numpy()
    improvement_pct = (var_comp_mean/var_orig_mean - 1) * 100
    
    summary_text = f"""Multi-Window Compensation Results:

Original mean variance: {var_orig_mean:.2f}
Compensated mean variance: {var_comp_mean:.2f}
Improvement: {improvement_pct:.1f}%

Final parameters:
Mean ax: {np.mean(a_params_final):.3f}
Mean ay: {np.mean(b_params_final):.3f}
Std ax: {np.std(a_params_final):.3f}
Std ay: {np.std(b_params_final):.3f}

Windows: {model.num_windows}
Total events: {len(x):,}
Final loss: {losses[-1]:.6f}"""
    
    axes[0, 1].text(1.1, 0.5, summary_text, transform=axes[0, 1].transAxes, 
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Multi-Window Scan Compensation Results\n{param_str}', fontsize=14)
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_multi_window_results{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Multi-window scan compensation for NPZ event files')
    parser.add_argument('input_path', help='Path to NPZ event file OR segments folder (when using --merge)')
    parser.add_argument('--merge', action='store_true', help='Merge all scan segments from folder')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num_windows', type=int, default=5, help='Number of main windows')
    parser.add_argument('--base_ax', type=float, default=0.0, help='Base ax value (from single-window results)')
    parser.add_argument('--base_ay', type=float, default=-76.0, help='Base ay value (from single-window results)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--subsample_for_training', type=int, default=None, help='Subsample events for training (e.g., 1000000 for 1M events)')
    
    args = parser.parse_args()
    
    # Load data
    if args.merge:
        if not os.path.isdir(args.input_path):
            raise ValueError(f"When using --merge, input_path must be a directory: {args.input_path}")
        
        if args.output_dir is None:
            args.output_dir = args.input_path
        
        folder_name = os.path.basename(args.input_path.rstrip('/'))
        base_name = f"{folder_name}_merged_multiwin"
        
        x, y, t, p = load_and_merge_segments(args.input_path)
    else:
        if not os.path.isfile(args.input_path):
            raise ValueError(f"NPZ file not found: {args.input_path}")
        
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.input_path)
        
        base_name = os.path.splitext(os.path.basename(args.input_path))[0] + "_multiwin"
        x, y, t, p = load_npz_events(args.input_path)
    
    print(f"\n{'='*60}")
    print("MULTI-WINDOW SCAN COMPENSATION")
    print(f"{'='*60}")
    print(f"Input: {args.input_path}")
    print(f"Events: {len(x):,}")
    print(f"Main windows: {args.num_windows}")
    print(f"Base parameters: ax={args.base_ax}, ay={args.base_ay}")
    
    # Subsample for training if dataset is very large
    if args.subsample_for_training is not None and len(x) > args.subsample_for_training:
        print(f"\nSubsampling {len(x):,} events to {args.subsample_for_training:,} for training...")
        
        # Create random indices for subsampling
        indices = np.random.choice(len(x), args.subsample_for_training, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        
        x_train = x[indices]
        y_train = y[indices]
        t_train = t[indices]
        p_train = p[indices]
        
        print(f"Training on {len(x_train):,} events ({100*len(x_train)/len(x):.1f}% of original)")
        
        # Keep original data for final evaluation
        x_full, y_full, t_full, p_full = x, y, t, p
    else:
        # Use all data for training
        x_train = x
        y_train = y
        t_train = t
        p_train = p
        x_full, y_full, t_full, p_full = x, y, t, p
    
    # Train model on training data
    model, losses, params_history, original_t_start, original_t_end = train_multi_window_scan_compensation(
        x_train, y_train, t_train, p_train,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        bin_width=args.bin_width,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        num_main_windows=args.num_windows,
        base_ax=args.base_ax,
        base_ay=args.base_ay
    )
    
    # Print final results
    a_params_final = model.compensator.a_params.detach().cpu().numpy()
    b_params_final = model.compensator.b_params.detach().cpu().numpy()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final A parameters: {a_params_final}")
    print(f"Final B parameters: {b_params_final}")
    print(f"Mean ax: {np.mean(a_params_final):.6f} (original single-window: {args.base_ax})")
    print(f"Mean ay: {np.mean(b_params_final):.6f} (original single-window: {args.base_ay})")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Save results
    save_results(model, losses, params_history, args.output_dir, base_name)
    
    # Visualize if requested (use full dataset for visualization)
    if args.visualize:
        print(f"\nCreating visualization using full dataset ({len(x_full):,} events)...")
        visualize_results(model, x_full, y_full, t_full, p_full, losses, params_history, 
                         args.bin_width, args.sensor_width, args.sensor_height, 
                         original_t_start, original_t_end, args.output_dir, base_name)
    
    print("\nMulti-window scan compensation complete!")
    print(f"Results saved to: {args.output_dir}")
    
    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")


if __name__ == "__main__":
    main()