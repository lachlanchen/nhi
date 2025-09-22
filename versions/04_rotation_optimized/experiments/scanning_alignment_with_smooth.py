#!/usr/bin/env python3
"""
Enhanced scan compensation code for NPZ event files - 
Now saves compensated frames and creates frame-by-frame comparisons
Plus 3D smoothing and mean/median shifting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
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
        # t_warped = timestamps - torch.sqrt((a_x * x_coords)**2 - (a_y * y_coords)**2)
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
        # Loss is the mean variance (matching original working code)
        # loss = torch.mean(variances)
        loss = torch.sum(variances)

        return event_tensor, loss

def train_scan_compensation(x, y, t, p, sensor_width=1280, sensor_height=720, 
                          bin_width=1e5, num_iterations=1000, learning_rate=1.0,
                          initial_params=None):
    """
    Train the scan compensation model
    """
    print(f"Training scan compensation...")
    print(f"Sensor size: {sensor_width} x {sensor_height}")
    print(f"Bin width: {bin_width/1000:.1f} ms")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    
    # Convert to tensors with explicit dtype
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    # Store original time range
    original_t_start = torch.tensor(float(ts.min().item()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(ts.max().item()), device=device, dtype=torch.float32)
    print(f"Original time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs")

    # Initialize parameters
    if initial_params is None:
        initial_params = torch.zeros(2, device=device, dtype=torch.float32, requires_grad=True)
    else:
        initial_params = torch.tensor(initial_params, device=device, dtype=torch.float32, requires_grad=True)
    
    model = ScanCompensation(initial_params)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    params_history = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        event_tensor, loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                 original_t_start, original_t_end)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        current_params = model.params.detach().cpu()
        params_history.append([current_params[0].item(), current_params[1].item()])
        
        if i % 100 == 0:
            params_vals = model.params.detach().cpu()
            print(f"Iteration {i}, Loss: {current_loss:.6f}, Params: a_x={params_vals[0].item():.6f}, a_y={params_vals[1].item():.6f}")
        
        # Adjust the learning rate if needed
        if i == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            print("Reduced learning rate by 50%")
        elif i == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            print("Reduced learning rate by 90%")

    return model, losses, params_history, original_t_start, original_t_end

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
            # Temporarily set parameters to zero
            original_params = model.params.clone()
            model.params.data.zero_()
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end)
            # Restore parameters
            model.params.data = original_params
    
    return event_tensor

def apply_3d_smoothing(tensor, kernel_size=3):
    """
    Apply 3D mean filtering to the tensor using convolution
    """
    # Create 3D mean kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=tensor.device, dtype=tensor.dtype)
    kernel = kernel / (kernel_size ** 3)  # Normalize
    
    # Add batch and channel dimensions for conv3d: (B, C, T, H, W)
    tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
    
    # Apply 3D convolution with padding to maintain size
    padding = kernel_size // 2
    smoothed_5d = F.conv3d(tensor_5d, kernel, padding=padding)
    
    # Remove batch and channel dimensions
    smoothed = smoothed_5d.squeeze(0).squeeze(0)
    
    return smoothed

def compute_frame_statistics_and_shift(tensor, use_median=True):
    """
    Compute frame-wise mean or median and subtract from each frame
    Returns: shifted_tensor, statistics_values
    """
    T, H, W = tensor.shape
    statistics_values = []
    
    # Compute statistics for each frame
    for t in range(T):
        frame = tensor[t]
        if use_median:
            stat_val = torch.median(frame).item()
        else:
            stat_val = torch.mean(frame).item()
        statistics_values.append(stat_val)
    
    # Convert to tensor for broadcasting
    stats_tensor = torch.tensor(statistics_values, device=tensor.device, dtype=tensor.dtype)
    stats_tensor = stats_tensor.view(T, 1, 1)  # Shape for broadcasting
    
    # Subtract statistics from each frame
    shifted_tensor = tensor - stats_tensor
    
    return shifted_tensor, statistics_values

def save_enhanced_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, 
                              output_dir, filename_prefix, use_median=True):
    """
    Save original, compensated, smoothed, and shifted event frames
    """
    print("Generating and saving enhanced event frames...")
    
    # Create frames directory with bin_width in name
    bin_width_ms = int(bin_width / 1000)
    frames_dir = os.path.join(output_dir, f"{filename_prefix}_frames_bin{bin_width_ms}ms")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate original and compensated frames
    event_tensor_orig = create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True)
    
    # Apply 3D smoothing to compensated tensor
    print("Applying 3D smoothing (3x3x3 mean kernel)...")
    event_tensor_smooth = apply_3d_smoothing(event_tensor_comp, kernel_size=3)
    
    # Compute frame statistics and shift
    stat_type = "median" if use_median else "mean"
    print(f"Computing frame-wise {stat_type} and shifting...")
    event_tensor_shifted, frame_statistics = compute_frame_statistics_and_shift(event_tensor_smooth, use_median=use_median)
    
    # Convert to numpy arrays
    frames_orig = np.array(event_tensor_orig.detach().cpu().numpy(), dtype=np.float32)
    frames_comp = np.array(event_tensor_comp.detach().cpu().numpy(), dtype=np.float32)
    frames_smooth = np.array(event_tensor_smooth.detach().cpu().numpy(), dtype=np.float32)
    frames_shifted = np.array(event_tensor_shifted.detach().cpu().numpy(), dtype=np.float32)
    
    num_frames = frames_orig.shape[0]
    print(f"Saving {num_frames} frame sets...")
    print(f"Original frames shape: {frames_orig.shape}")
    print(f"Compensated frames shape: {frames_comp.shape}")
    print(f"Smoothed frames shape: {frames_smooth.shape}")
    print(f"Shifted frames shape: {frames_shifted.shape}")
    
    # Save as numpy arrays
    np.save(os.path.join(frames_dir, "frames_original.npy"), frames_orig)
    np.save(os.path.join(frames_dir, "frames_compensated.npy"), frames_comp)
    np.save(os.path.join(frames_dir, "frames_smoothed.npy"), frames_smooth)
    np.save(os.path.join(frames_dir, "frames_shifted.npy"), frames_shifted)
    np.save(os.path.join(frames_dir, f"frame_{stat_type}_values.npy"), np.array(frame_statistics))
    
    # Create subdirectories for individual frame images
    orig_dir = os.path.join(frames_dir, "original")
    comp_dir = os.path.join(frames_dir, "compensated")
    smooth_dir = os.path.join(frames_dir, "smoothed")
    shifted_dir = os.path.join(frames_dir, "shifted")
    
    for dir_path in [orig_dir, comp_dir, smooth_dir, shifted_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Determine colormap ranges for consistency
    all_frames = [frames_orig, frames_comp, frames_smooth, frames_shifted]
    vmin = min(float(np.min(f)) for f in all_frames)
    vmax = max(float(np.max(f)) for f in all_frames)
    
    print(f"Value range across all frame types: {vmin:.3f} to {vmax:.3f}")
    
    # Save individual frames as images
    for i in range(num_frames):
        # Original frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_orig[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Original Frame {i}')
        plt.savefig(os.path.join(orig_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Compensated frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_comp[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Compensated Frame {i}')
        plt.savefig(os.path.join(comp_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Smoothed frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_smooth[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Smoothed Frame {i}')
        plt.savefig(os.path.join(smooth_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Shifted frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_shifted[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Shifted Frame {i} ({stat_type} subtracted)')
        plt.savefig(os.path.join(shifted_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot frame statistics
    plt.figure(figsize=(12, 6))
    plt.plot(frame_statistics, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Frame Index')
    plt.ylabel(f'Frame {stat_type.capitalize()} Value')
    plt.title(f'Frame-wise {stat_type.capitalize()} Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    stats_plot_path = os.path.join(frames_dir, f"frame_{stat_type}_plot.png")
    plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced frames saved to: {frames_dir}")
    print(f"Frame {stat_type} plot saved to: {stats_plot_path}")
    
    return frames_orig, frames_comp, frames_smooth, frames_shifted, frame_statistics

def save_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, output_dir, filename_prefix):
    """
    Original save_event_frames function (for backward compatibility)
    """
    print("Generating and saving event frames...")
    
    # Create frames directory with bin_width in name
    bin_width_ms = int(bin_width / 1000)
    frames_dir = os.path.join(output_dir, f"{filename_prefix}_frames_bin{bin_width_ms}ms")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate both original and compensated frames
    event_tensor_orig = create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True)
    
    # Convert to numpy with explicit conversion
    frames_orig = np.array(event_tensor_orig.detach().cpu().numpy(), dtype=np.float32)
    frames_comp = np.array(event_tensor_comp.detach().cpu().numpy(), dtype=np.float32)
    
    num_frames = frames_orig.shape[0]
    print(f"Saving {num_frames} frame pairs...")
    print(f"Original frames shape: {frames_orig.shape}")
    print(f"Compensated frames shape: {frames_comp.shape}")
    
    # Save as numpy arrays
    np.save(os.path.join(frames_dir, "frames_original.npy"), frames_orig)
    np.save(os.path.join(frames_dir, "frames_compensated.npy"), frames_comp)
    
    # Save individual frames as images
    orig_dir = os.path.join(frames_dir, "original")
    comp_dir = os.path.join(frames_dir, "compensated")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    
    # Determine colormap range for consistency - use explicit numpy operations
    vmin_orig = float(np.min(frames_orig))
    vmin_comp = float(np.min(frames_comp))
    vmax_orig = float(np.max(frames_orig))
    vmax_comp = float(np.max(frames_comp))
    
    vmin = min(vmin_orig, vmin_comp)
    vmax = max(vmax_orig, vmax_comp)
    
    print(f"Value range: {vmin:.3f} to {vmax:.3f}")
    
    for i in range(num_frames):
        # Save original frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_orig[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Original Frame {i}')
        plt.savefig(os.path.join(orig_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save compensated frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frames_comp[i], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar()
        plt.title(f'Compensated Frame {i}')
        plt.savefig(os.path.join(comp_dir, f"frame_{i:04d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Frames saved to: {frames_dir}")
    return frames_orig, frames_comp

def create_frame_comparison_plots(frames_orig, frames_comp, output_dir, filename_prefix, 
                                 max_frames_to_show=None, skip_frames=None, bin_width=None):
    """
    Create side-by-side comparison plots of original vs compensated frames
    Only creates individual comparison plots (no overview plot)
    """
    num_frames = frames_orig.shape[0]
    
    # Determine which frames to show - default to all frames
    if max_frames_to_show is None:
        frame_indices = list(range(num_frames))
        skip_frames = 1  # Show every frame
    else:
        if skip_frames is None:
            skip_frames = max(1, num_frames // max_frames_to_show)
        frame_indices = list(range(0, num_frames, skip_frames))[:max_frames_to_show]
    
    print(f"Creating individual comparison plots for {len(frame_indices)} frames: {frame_indices}")
    
    # Determine colormap range for consistency - use explicit numpy operations
    vmin_orig = float(np.min(frames_orig))
    vmin_comp = float(np.min(frames_comp))
    vmax_orig = float(np.max(frames_orig))
    vmax_comp = float(np.max(frames_comp))
    
    vmin = min(vmin_orig, vmin_comp)
    vmax = max(vmax_orig, vmax_comp)
    
    # Create individual comparison plots directory
    bin_width_ms = int(bin_width / 1000) if bin_width else "unknown"
    comparison_dir = os.path.join(output_dir, f"{filename_prefix}_frame_comparisons_bin{bin_width_ms}ms")
    os.makedirs(comparison_dir, exist_ok=True)
    
    for frame_idx in frame_indices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original
        im1 = ax1.imshow(frames_orig[frame_idx], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        ax1.set_title(f'Original Frame {frame_idx}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1)
        
        # Compensated
        im2 = ax2.imshow(frames_comp[frame_idx], cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title(f'Compensated Frame {frame_idx}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2)
        
        # Calculate and display statistics
        orig_var = float(np.var(frames_orig[frame_idx]))
        comp_var = float(np.var(frames_comp[frame_idx]))
        improvement = (comp_var / orig_var - 1) * 100 if orig_var > 0 else 0
        
        plt.suptitle(f'Frame {frame_idx} Comparison - Variance: {orig_var:.3f} → {comp_var:.3f} ({improvement:.1f}%)', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"comparison_frame_{frame_idx:04d}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Individual comparison plots saved to: {comparison_dir}")

def visualize_results(model, x, y, t, p, losses, params_history, bin_width, 
                     sensor_width, sensor_height, original_t_start, original_t_end, 
                     output_dir=None, filename_prefix=""):
    """
    Visualize training results and compensated events
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Parameters evolution
    params_array = np.array(params_history)
    axes[0, 1].plot(params_array[:, 0], label='a_x')
    axes[0, 1].plot(params_array[:, 1], label='a_y')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Parameter Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
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
    axes[1, 1].text(0.1, 0.9, f'Original mean variance: {var_orig_mean:.2f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.8, f'Compensated mean variance: {var_comp_mean:.2f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'Improvement: {improvement_pct:.1f}%', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Final a_x: {final_params_tensor[0].item():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'Final a_y: {final_params_tensor[1].item():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Final loss: {losses[-1]:.6f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.3, f'Total events: {len(x):,}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'Time bins: {actual_num_bins}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
    
    plt.show()

def save_results(model, losses, params_history, output_dir, filename_prefix):
    """
    Save training results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final parameters
        final_params_tensor = model.params.detach().cpu()
        final_params = [final_params_tensor[0].item(), final_params_tensor[1].item()]
        results_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results.txt")
        
        with open(results_path, 'w') as f:
            f.write("SCAN COMPENSATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Final parameters (a_x, a_y): [{final_params[0]:.6f}, {final_params[1]:.6f}]\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Training iterations: {len(losses)}\n")
            f.write("\nParameter evolution (every 100 iterations):\n")
            for i, params in enumerate(params_history[::100]):
                f.write(f"Iteration {i*100}: a_x={params[0]:.6f}, a_y={params[1]:.6f}\n")
        
        print(f"Results saved to: {results_path}")
        
        # Save parameters as numpy arrays
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_params.npy"), np.array(final_params))
        np.save(os.path.join(output_dir, f"{filename_prefix}_loss_history.npy"), np.array(losses))
        np.save(os.path.join(output_dir, f"{filename_prefix}_params_history.npy"), np.array(params_history))

def main():
    parser = argparse.ArgumentParser(description='Scan compensation for NPZ event files')
    parser.add_argument('npz_file', help='Path to NPZ event file')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--initial_a_x', type=float, default=0.0, help='Initial a_x parameter')
    parser.add_argument('--initial_a_y', type=float, default=0.0, help='Initial a_y parameter')
    parser.add_argument('--direct_params', action='store_true', help='Use provided a_x and a_y directly without optimization')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--save_frames', action='store_true', help='Save all event frames as images and arrays (basic version)')
    parser.add_argument('--save_enhanced_frames', action='store_true', help='Save enhanced frames with smoothing and shifting')
    parser.add_argument('--use_mean', action='store_true', help='Use mean instead of median for frame shifting (default: median)')
    parser.add_argument('--max_comparison_frames', type=int, default=None, help='Maximum number of frames to show in comparison plots (default: None = show all frames)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.npz_file)
    
    # Create filename prefix
    base_name = os.path.splitext(os.path.basename(args.npz_file))[0]
    
    print(f"Analyzing: {args.npz_file}")
    
    # Load events
    x, y, t, p = load_npz_events(args.npz_file)
    
    # Store original time range - convert numpy scalars to Python floats first
    original_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
    
    # Initial parameters
    initial_params = [args.initial_a_x, args.initial_a_y]
    
    if args.direct_params:
        # Direct parameter mode - no optimization
        print(f"Using direct parameters without optimization:")
        print(f"a_x = {args.initial_a_x}, a_y = {args.initial_a_y}")
        
        # Create model with specified parameters
        model_params = torch.tensor(initial_params, device=device, dtype=torch.float32, requires_grad=False)
        model = ScanCompensation(model_params)
        
        # Create dummy training history for compatibility
        losses = [0.0]  # Single dummy loss
        params_history = [initial_params]  # Single entry
        
        # Print final results
        print(f"\nUsing direct parameters (a_x, a_y): [{args.initial_a_x:.6f}, {args.initial_a_y:.6f}]")
        
    else:
        # Train model
        model, losses, params_history, original_t_start, original_t_end = train_scan_compensation(
            x, y, t, p,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            bin_width=args.bin_width,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            initial_params=initial_params
        )
        
        # Print final results
        final_params_tensor = model.params.detach().cpu()
        final_params = [final_params_tensor[0].item(), final_params_tensor[1].item()]
        print(f"\nFinal parameters (a_x, a_y): [{final_params[0]:.6f}, {final_params[1]:.6f}]")
        print(f"Final loss: {losses[-1]:.6f}")
    
    # Save results
    save_results(model, losses, params_history, args.output_dir, base_name)
    
    # Save enhanced frames if requested
    if args.save_enhanced_frames:
        use_median = not args.use_mean  # Default to median unless --use_mean is specified
        frames_orig, frames_comp, frames_smooth, frames_shifted, frame_stats = save_enhanced_event_frames(
            model, x, y, t, p, args.sensor_height, args.sensor_width, 
            args.bin_width, original_t_start, original_t_end, args.output_dir, base_name,
            use_median=use_median
        )
    
    # Save basic frames if requested (backward compatibility)
    elif args.save_frames:
        frames_orig, frames_comp = save_event_frames(
            model, x, y, t, p, args.sensor_height, args.sensor_width, 
            args.bin_width, original_t_start, original_t_end, args.output_dir, base_name
        )
        
        # Create frame comparison plots
        create_frame_comparison_plots(
            frames_orig, frames_comp, args.output_dir, base_name,
            max_frames_to_show=args.max_comparison_frames, bin_width=args.bin_width
        )
    
    # Visualize if requested
    if args.visualize:
        visualize_results(model, x, y, t, p, losses, params_history, 
                         args.bin_width, args.sensor_width, args.sensor_height, 
                         original_t_start, original_t_end, args.output_dir, base_name)
    
    print("Scan compensation complete!")

if __name__ == "__main__":
    main()