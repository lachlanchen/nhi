#!/usr/bin/env python3
"""
Enhanced scan compensation code with time-dependent modulation phi(t)
t' = t - (ax*x + ay*y)*phi(t)
where phi(t) is restricted to [0.8, 1.2] and parameterized as a polynomial
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

def load_npz_events(npz_path):
    """
    Load events from NPZ file
    Expected format: x, y, t, p arrays
    """
    print(f"Loading events from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Print available keys for debugging
    print(f"Available keys in NPZ file: {list(data.keys())}")
    
    # Extract arrays
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)  # Should already be in microseconds
    p = data['p'].astype(np.float32)
    
    print(f"Loaded {len(x)} events")
    print(f"Time range: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f} seconds)")
    print(f"X range: {x.min():.0f} - {x.max():.0f}")
    print(f"Y range: {y.min():.0f} - {y.max():.0f}")
    print(f"Polarity range: {p.min():.0f} - {p.max():.0f}")
    
    # Convert polarity to [-1, 1] if it's [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        print("Converted polarity from [0,1] to [-1,1]")
    
    return x, y, t, p

def load_and_merge_segments(segments_folder):
    """
    Load and merge all scan segments with proper time normalization and polarity handling
    """
    print(f"\n{'='*60}")
    print("MERGING SCAN SEGMENTS")
    print(f"{'='*60}")
    print(f"Loading segments from: {segments_folder}")
    
    # Find all segment files
    segment_files = glob.glob(os.path.join(segments_folder, "Scan_*_events.npz"))
    segment_files.sort()  # Ensure consistent ordering
    
    if not segment_files:
        raise FileNotFoundError(f"No segment files found in {segments_folder}")
    
    print(f"Found {len(segment_files)} segment files:")
    for f in segment_files:
        print(f"  {os.path.basename(f)}")
    
    all_x = []
    all_y = []
    all_t = []
    all_p = []
    
    for segment_file in segment_files:
        print(f"\nProcessing: {os.path.basename(segment_file)}")
        
        # Load segment data
        data = np.load(segment_file)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        t = data['t'].astype(np.float32)
        p = data['p'].astype(np.float32)
        
        # Get metadata
        start_time = float(data['start_time'])
        end_time = float(data['end_time'])
        duration = float(data['duration_us'])
        direction = str(data['direction'])
        
        print(f"  Direction: {direction}")
        print(f"  Events: {len(x):,}")
        print(f"  Start time: {start_time:.0f} μs")
        print(f"  Duration: {duration:.0f} μs ({duration/1000:.1f} ms)")
        print(f"  Original time range: {t.min():.0f} - {t.max():.0f} μs")
        
        # Step 1: Normalize time by subtracting start time
        t_normalized = t - start_time
        print(f"  After start time subtraction: {t_normalized.min():.0f} - {t_normalized.max():.0f} μs")
        
        # Step 2: Convert polarity to [-1, 1] if needed
        if p.min() >= 0 and p.max() <= 1:
            p = (p - 0.5) * 2
            print(f"  Converted polarity from [0,1] to [-1,1]")
        
        # Step 3: Handle backward scans
        if 'Backward' in direction:
            print(f"  Processing backward scan:")
            # Reverse time: period - timestamps
            t_normalized = duration - t_normalized
            print(f"    After time reversal: {t_normalized.min():.0f} - {t_normalized.max():.0f} μs")
            
            # Flip polarity
            p = -p
            print(f"    Flipped polarity for backward scan")
            print(f"    New polarity range: {p.min():.0f} - {p.max():.0f}")
        
        # Add to lists
        all_x.append(x)
        all_y.append(y)
        all_t.append(t_normalized)
        all_p.append(p)
    
    # Concatenate all arrays
    print(f"\nMerging all segments...")
    merged_x = np.concatenate(all_x)
    merged_y = np.concatenate(all_y)
    merged_t = np.concatenate(all_t)
    merged_p = np.concatenate(all_p)
    
    print(f"Merged events: {len(merged_x):,}")
    print(f"Merged time range: {merged_t.min():.0f} - {merged_t.max():.0f} μs")
    print(f"Merged X range: {merged_x.min():.0f} - {merged_x.max():.0f}")
    print(f"Merged Y range: {merged_y.min():.0f} - {merged_y.max():.0f}")
    print(f"Merged polarity range: {merged_p.min():.0f} - {merged_p.max():.0f}")
    
    # Sort by time for better processing (optional but recommended)
    print("Sorting events by time...")
    sort_indices = np.argsort(merged_t)
    merged_x = merged_x[sort_indices]
    merged_y = merged_y[sort_indices]
    merged_t = merged_t[sort_indices]
    merged_p = merged_p[sort_indices]
    
    print(f"Final merged time range: {merged_t.min():.0f} - {merged_t.max():.0f} μs")
    print(f"Total duration: {(merged_t.max() - merged_t.min())/1000:.1f} ms")
    
    return merged_x, merged_y, merged_t, merged_p

class ScanCompensationWithPhi(nn.Module):
    def __init__(self, initial_params, phi_order=2):
        super().__init__()
        """
        Initialize with parameters: [ax, ay, p1, p2, ...] where p1, p2, ... are phi(t) polynomial coefficients
        phi(t) = 1 + p1*t_norm + p2*t_norm^2 + ...
        """
        self.phi_order = phi_order
        
        # Total parameters: ax, ay, and phi_order polynomial coefficients
        expected_params = 2 + phi_order
        
        if isinstance(initial_params, torch.Tensor):
            if len(initial_params) != expected_params:
                # Pad with zeros if not enough parameters
                padded_params = torch.zeros(expected_params, dtype=torch.float32)
                padded_params[:len(initial_params)] = initial_params
                initial_params = padded_params
            self.params = nn.Parameter(initial_params.clone().detach())
        else:
            if len(initial_params) != expected_params:
                # Pad with zeros if not enough parameters
                padded_params = list(initial_params) + [0.0] * (expected_params - len(initial_params))
                initial_params = padded_params
            self.params = nn.Parameter(torch.tensor(initial_params, dtype=torch.float32))
        
        print(f"Initialized with {expected_params} parameters: ax, ay, and {phi_order} phi coefficients")
    
    def compute_phi(self, timestamps, t_min, t_max):
        """
        Compute phi(t) as a polynomial: phi(t) = 1 + p1*t_norm + p2*t_norm^2 + ...
        Clamped to range [0.8, 1.2]
        """
        # Normalize time to [0, 1]
        t_norm = (timestamps - t_min) / (t_max - t_min + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Start with phi(t) = 1
        phi_t = torch.ones_like(t_norm)
        
        # Add polynomial terms
        for i in range(self.phi_order):
            coeff_idx = 2 + i  # Skip ax and ay
            if coeff_idx < len(self.params):
                phi_t += self.params[coeff_idx] * (t_norm ** (i + 1))
        
        # Clamp to [0.8, 1.2] range
        phi_t = torch.clamp(phi_t, 0.8, 1.2)
        
        return phi_t, t_norm
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps based on x and y positions with time-dependent modulation phi(t).
        t' = t - (ax*x + ay*y)*phi(t)
        """
        a_x = self.params[0]
        a_y = self.params[1]
        
        # Compute phi(t)
        t_min = timestamps.min()
        t_max = timestamps.max()
        phi_t, t_norm = self.compute_phi(timestamps, t_min, t_max)
        
        # Apply compensation with phi(t) modulation
        compensation = (a_x * x_coords + a_y * y_coords) * phi_t
        t_warped = timestamps - compensation
        
        return x_coords, y_coords, t_warped, phi_t, t_norm

    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, 
                original_t_start=None, original_t_end=None):
        """
        Process events through the model by warping them and then computing the loss.
        """
        x_warped, y_warped, t_warped, phi_t, t_norm = self.warp(x_coords, y_coords, timestamps)
        
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
        t_norm_bins = (t_warped - t_start) / time_bin_width

        # Compute floor and ceil indices for time bins
        t0 = torch.floor(t_norm_bins)
        t1 = t0 + 1

        # Compute weights for linear interpolation over time
        wt = (t_norm_bins - t0).float()  # Ensure float32

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

        return event_tensor, loss, phi_t, t_norm

def train_scan_compensation_with_phi(x, y, t, p, sensor_width=1280, sensor_height=720, 
                                   bin_width=1e5, num_iterations=1000, learning_rate=1.0,
                                   initial_params=None, phi_order=2):
    """
    Train the scan compensation model with phi(t)
    """
    print(f"Training scan compensation with phi(t)...")
    print(f"Sensor size: {sensor_width} x {sensor_height}")
    print(f"Bin width: {bin_width/1000:.1f} ms")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Phi(t) polynomial order: {phi_order}")
    
    # Convert to tensors with explicit dtype
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    # Store original time range
    original_t_start = torch.tensor(float(ts.min().item()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(ts.max().item()), device=device, dtype=torch.float32)
    print(f"Original time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs")

    # Initialize parameters: [ax, ay, p1, p2, ...]
    expected_params = 2 + phi_order
    if initial_params is None:
        initial_params = torch.zeros(expected_params, device=device, dtype=torch.float32, requires_grad=True)
    else:
        if len(initial_params) != expected_params:
            # Pad with zeros if not enough parameters
            padded_params = list(initial_params) + [0.0] * (expected_params - len(initial_params))
            initial_params = torch.tensor(padded_params, device=device, dtype=torch.float32, requires_grad=True)
        else:
            initial_params = torch.tensor(initial_params, device=device, dtype=torch.float32, requires_grad=True)
    
    model = ScanCompensationWithPhi(initial_params, phi_order=phi_order)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    params_history = []
    phi_history = []  # To store phi(t) evolution

    # Sample time points for phi(t) visualization
    sample_times = torch.linspace(ts.min(), ts.max(), 100, device=device, dtype=torch.float32)

    for i in range(num_iterations):
        optimizer.zero_grad()
        event_tensor, loss, phi_t, t_norm = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                                 original_t_start, original_t_end)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        current_params = model.params.detach().cpu()
        params_history.append([current_params[j].item() for j in range(len(current_params))])
        
        # Sample phi(t) for visualization
        if i % 50 == 0:  # Sample every 50 iterations to avoid too much memory usage
            with torch.no_grad():
                sample_phi, sample_t_norm = model.compute_phi(sample_times, ts.min(), ts.max())
                phi_history.append(sample_phi.cpu().numpy())
        
        if i % 100 == 0:
            param_str = f"ax={current_params[0].item():.6f}, ay={current_params[1].item():.6f}"
            phi_params = [f"p{j+1}={current_params[2+j].item():.6f}" for j in range(phi_order)]
            phi_str = ", ".join(phi_params)
            print(f"Iteration {i}, Loss: {current_loss:.6f}, {param_str}, {phi_str}")
        
        # Adjust the learning rate if needed
        if i == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            print("Reduced learning rate by 50%")
        elif i == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            print("Reduced learning rate by 90%")

    return model, losses, params_history, phi_history, sample_times, original_t_start, original_t_end

def create_event_frames_with_phi(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True):
    """
    Create event frames with or without phi(t) compensation
    """
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        if compensated:
            # Use current model parameters
            event_tensor, _, phi_t, t_norm = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
        else:
            # Temporarily set parameters to zero
            original_params = model.params.clone()
            model.params.data.zero_()
            event_tensor, _, phi_t, t_norm = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
            # Restore parameters
            model.params.data = original_params
    
    return event_tensor, phi_t, t_norm

def get_param_string_with_phi(model):
    """
    Get a formatted string of the model parameters including phi coefficients
    """
    params = model.params.detach().cpu()
    a_x = params[0].item()
    a_y = params[1].item()
    
    param_str = f"ax={a_x:.4f}, ay={a_y:.4f}"
    
    # Add phi parameters
    phi_params = []
    for i in range(model.phi_order):
        coeff_idx = 2 + i
        if coeff_idx < len(params):
            phi_params.append(f"p{i+1}={params[coeff_idx].item():.4f}")
    
    if phi_params:
        param_str += ", " + ", ".join(phi_params)
    
    return param_str

def get_param_suffix_with_phi(model):
    """
    Get a filename-safe suffix with the model parameters including phi
    """
    params = model.params.detach().cpu()
    a_x = params[0].item()
    a_y = params[1].item()
    
    suffix = f"_ax{a_x:.4f}_ay{a_y:.4f}"
    
    # Add phi parameters
    for i in range(model.phi_order):
        coeff_idx = 2 + i
        if coeff_idx < len(params):
            suffix += f"_p{i+1}{params[coeff_idx].item():.4f}"
    
    return suffix

def visualize_phi_function(model, sample_times, phi_history, losses, params_history, 
                          output_dir=None, filename_prefix=""):
    """
    Simplified phi(t) visualization that avoids numpy method calls
    """
    # Work with torch tensors and Python lists to avoid numpy conflicts
    times_tensor = sample_times.cpu()
    times_list = times_tensor.tolist()
    
    # Get current phi(t) values using torch operations
    with torch.no_grad():
        current_phi, t_norm = model.compute_phi(sample_times, sample_times.min(), sample_times.max())
        current_phi_list = current_phi.cpu().tolist()
        t_norm_list = t_norm.cpu().tolist()
    
    # Get current parameters
    params = model.params.detach().cpu()
    a_x = float(params[0].item())
    a_y = float(params[1].item())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Phi(t) vs time
    axes[0, 0].plot([t / 1000 for t in times_list], current_phi_list, 'b-', linewidth=2, label='φ(t)')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='φ=1')
    axes[0, 0].axhline(y=0.8, color='g', linestyle=':', alpha=0.7, label='φ=0.8')
    axes[0, 0].axhline(y=1.2, color='g', linestyle=':', alpha=0.7, label='φ=1.2')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('φ(t)')
    axes[0, 0].set_title('Time-dependent Modulation φ(t)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.75, 1.25)
    
    # 2. ax * phi(t) vs time
    ax_phi_t = [a_x * phi for phi in current_phi_list]
    axes[0, 1].plot([t / 1000 for t in times_list], ax_phi_t, 'r-', linewidth=2, label=f'ax·φ(t)')
    axes[0, 1].axhline(y=a_x, color='r', linestyle='--', alpha=0.7, label=f'ax={a_x:.4f}')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('ax·φ(t)')
    axes[0, 1].set_title('X-axis Compensation Factor')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. ay * phi(t) vs time
    ay_phi_t = [a_y * phi for phi in current_phi_list]
    axes[1, 0].plot([t / 1000 for t in times_list], ay_phi_t, 'g-', linewidth=2, label=f'ay·φ(t)')
    axes[1, 0].axhline(y=a_y, color='g', linestyle='--', alpha=0.7, label=f'ay={a_y:.4f}')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('ay·φ(t)')
    axes[1, 0].set_title('Y-axis Compensation Factor')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Training loss
    axes[1, 1].plot(losses, 'k-', linewidth=1)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Calculate simple statistics using Python
    phi_min = min(current_phi_list)
    phi_max = max(current_phi_list)
    phi_mean = sum(current_phi_list) / len(current_phi_list)
    phi_std = (sum((x - phi_mean) ** 2 for x in current_phi_list) / len(current_phi_list)) ** 0.5
    
    # Display polynomial equation
    phi_eq = "φ(t) = 1"
    for i in range(model.phi_order):
        coeff_idx = 2 + i
        if coeff_idx < len(params):
            coeff = float(params[coeff_idx].item())
            if coeff >= 0:
                phi_eq += f" + {coeff:.4f}·t^{i+1}"
            else:
                phi_eq += f" - {abs(coeff):.4f}·t^{i+1}"
    
    # Add title with equation and statistics
    fig.suptitle(f'Scan Compensation with φ(t) Modulation\n{phi_eq}\n' +
                f'φ(t) range: [{phi_min:.3f}, {phi_max:.3f}], mean: {phi_mean:.3f}±{phi_std:.3f}', 
                fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix_with_phi(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_phi_analysis{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Phi analysis plot saved to: {plot_path}")
    
    plt.show()
    
    return current_phi_list, times_list

def visualize_results_with_phi(model, x, y, t, p, losses, params_history, bin_width, 
                              sensor_width, sensor_height, original_t_start, original_t_end, 
                              output_dir=None, filename_prefix=""):
    """
    Visualize training results and compensated events with phi(t)
    """
    # Get parameter string for plots
    param_str = get_param_string_with_phi(model)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Parameters evolution
    params_array = np.array(params_history)
    axes[0, 1].plot(params_array[:, 0], label='ax', color='r')
    axes[0, 1].plot(params_array[:, 1], label='ay', color='g')
    
    # Plot phi parameters
    colors = ['b', 'orange', 'purple', 'brown']
    for i in range(model.phi_order):
        if 2 + i < params_array.shape[1]:
            color = colors[i % len(colors)]
            axes[0, 1].plot(params_array[:, 2 + i], label=f'p{i+1}', color=color)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Parameter Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Generate event frames with original time range
    event_tensor_orig, _, _ = create_event_frames_with_phi(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
                                                          original_t_start, original_t_end, compensated=False)
    event_tensor_comp, phi_t, t_norm = create_event_frames_with_phi(model, x, y, t, p, sensor_height, sensor_width, bin_width, 
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
    axes[1, 2].set_title(f'Compensated - Bin {bin_idx}')
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
    axes[1, 0].plot(var_comp_list, label='Compensated', alpha=0.7)
    axes[1, 0].set_xlabel('Time Bin')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Variance Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary statistics
    final_params_tensor = model.params.detach().cpu()
    improvement_pct = (var_comp_mean/var_orig_mean - 1) * 100
    
    stats_text = f'Original mean variance: {var_orig_mean:.2f}\n'
    stats_text += f'Compensated mean variance: {var_comp_mean:.2f}\n'
    stats_text += f'Improvement: {improvement_pct:.1f}%\n'
    stats_text += f'Final ax: {final_params_tensor[0].item():.4f}\n'
    stats_text += f'Final ay: {final_params_tensor[1].item():.4f}\n'
    
    for i in range(model.phi_order):
        coeff_idx = 2 + i
        if coeff_idx < len(final_params_tensor):
            stats_text += f'Final p{i+1}: {final_params_tensor[coeff_idx].item():.4f}\n'
    
    stats_text += f'Final loss: {losses[-1]:.6f}\n'
    stats_text += f'Total events: {len(x):,}\n'
    stats_text += f'Time bins: {actual_num_bins}'
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Add overall title with parameters
    fig.suptitle(f'Scan Compensation Results with φ(t)\n{param_str}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix_with_phi(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results_phi{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
    
    plt.show()

def save_results_with_phi(model, losses, params_history, output_dir, filename_prefix):
    """
    Save training results with phi parameters
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get parameter suffix for filenames
        param_suffix = get_param_suffix_with_phi(model)
        
        # Save final parameters
        final_params_tensor = model.params.detach().cpu()
        final_params = [final_params_tensor[j].item() for j in range(len(final_params_tensor))]
        results_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results_phi{param_suffix}.txt")
        
        with open(results_path, 'w') as f:
            f.write("SCAN COMPENSATION RESULTS WITH PHI(T)\n")
            f.write("=" * 50 + "\n\n")
            
            # Write polynomial equation
            phi_eq = "φ(t) = 1"
            for i in range(model.phi_order):
                coeff_idx = 2 + i
                if coeff_idx < len(final_params):
                    coeff = final_params[coeff_idx]
                    if coeff >= 0:
                        phi_eq += f" + {coeff:.6f}·t^{i+1}"
                    else:
                        phi_eq += f" - {abs(coeff):.6f}·t^{i+1}"
            
            f.write(f"Phi function: {phi_eq}\n\n")
            f.write(f"Final parameters:\n")
            f.write(f"  ax = {final_params[0]:.6f}\n")
            f.write(f"  ay = {final_params[1]:.6f}\n")
            
            for i in range(model.phi_order):
                coeff_idx = 2 + i
                if coeff_idx < len(final_params):
                    f.write(f"  p{i+1} = {final_params[coeff_idx]:.6f}\n")
            
            f.write(f"\nFinal loss: {losses[-1]:.6f}\n")
            f.write(f"Training iterations: {len(losses)}\n")
            f.write(f"Phi polynomial order: {model.phi_order}\n")
            
            f.write("\nParameter evolution (every 100 iterations):\n")
            for i, params in enumerate(params_history[::100]):
                param_str = f"Iteration {i*100}: ax={params[0]:.6f}, ay={params[1]:.6f}"
                for j in range(model.phi_order):
                    if 2 + j < len(params):
                        param_str += f", p{j+1}={params[2+j]:.6f}"
                f.write(param_str + "\n")
        
        print(f"Results saved to: {results_path}")
        
        # Save parameters as numpy arrays
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_params_phi{param_suffix}.npy"), np.array(final_params))
        np.save(os.path.join(output_dir, f"{filename_prefix}_loss_history_phi{param_suffix}.npy"), np.array(losses))
        np.save(os.path.join(output_dir, f"{filename_prefix}_params_history_phi{param_suffix}.npy"), np.array(params_history))

def main_with_phi():
    parser = argparse.ArgumentParser(description='Scan compensation with time-dependent phi(t) for NPZ event files')
    parser.add_argument('input_path', help='Path to NPZ event file OR segments folder (when using --merge)')
    parser.add_argument('--merge', action='store_true', help='Merge all scan segments from folder instead of processing single file')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--initial_a_x', type=float, default=0.0, help='Initial a_x parameter')
    parser.add_argument('--initial_a_y', type=float, default=0.0, help='Initial a_y parameter')
    parser.add_argument('--phi_order', type=int, default=2, help='Polynomial order for phi(t) (default: 2 for quadratic)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--visualize_phi', action='store_true', help='Show detailed phi(t) analysis plots')
    
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
        base_name = f"{folder_name}_merged_phi"
        
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
        base_name = os.path.splitext(os.path.basename(npz_file))[0] + "_phi"
        
        print(f"Analyzing: {npz_file}")
        
        # Load events from single file
        x, y, t, p = load_npz_events(npz_file)
    
    # Initial parameters: [ax, ay, p1, p2, ...] (phi coefficients start at 0)
    expected_params = 2 + args.phi_order
    initial_params = [args.initial_a_x, args.initial_a_y] + [0.0] * args.phi_order
    
    print(f"Training with phi(t) modulation...")
    print(f"Phi polynomial order: {args.phi_order}")
    print(f"Initial parameters: {initial_params}")
    
    # Train model with phi(t)
    model, losses, params_history, phi_history, sample_times, original_t_start, original_t_end = train_scan_compensation_with_phi(
        x, y, t, p,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        bin_width=args.bin_width,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        initial_params=initial_params,
        phi_order=args.phi_order
    )
    
    # Print final results
    final_params_tensor = model.params.detach().cpu()
    final_params = [final_params_tensor[j].item() for j in range(len(final_params_tensor))]
    
    print(f"\nFinal parameters:")
    print(f"  ax = {final_params[0]:.6f}")
    print(f"  ay = {final_params[1]:.6f}")
    for i in range(args.phi_order):
        if 2 + i < len(final_params):
            print(f"  p{i+1} = {final_params[2+i]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Save results
    save_results_with_phi(model, losses, params_history, args.output_dir, base_name)
    
    # Visualize phi(t) analysis if requested
    if args.visualize_phi:
        visualize_phi_function(model, sample_times, phi_history, losses, params_history, 
                              args.output_dir, base_name)
    
    # Visualize general results if requested
    if args.visualize:
        visualize_results_with_phi(model, x, y, t, p, losses, params_history, 
                                  args.bin_width, args.sensor_width, args.sensor_height, 
                                  original_t_start, original_t_end, args.output_dir, base_name)
    
    print("Scan compensation with phi(t) complete!")

if __name__ == "__main__":
    main_with_phi()