#!/usr/bin/env python3
"""
Updated scan compensation code for NPZ event files - Fixed CUDA bounds issue
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os

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
        self.params = nn.Parameter(initial_params)
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps based on x and y positions.
        """
        a_x = self.params[0]
        a_y = self.params[1]
        t_warped = timestamps - a_x * x_coords - a_y * y_coords
        # t_warped = timestamps - torch.sqrt((a_x * x_coords)**2 - (a_y * y_coords)**2)
        return x_coords, y_coords, t_warped


    
    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width):
        """
        Process events through the model by warping them and then computing the loss.
        """
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        # Define time binning parameters
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        t_start = t_warped.min()
        t_end = t_warped.max()
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
    
    # Convert to tensors
    xs = torch.tensor(x, device=device)
    ys = torch.tensor(y, device=device)
    ts = torch.tensor(t, device=device)
    ps = torch.tensor(p, device=device)

    # Initialize parameters
    if initial_params is None:
        initial_params = torch.zeros(2, device=device, requires_grad=True)
    else:
        initial_params = torch.tensor(initial_params, device=device, requires_grad=True)
    
    model = ScanCompensation(initial_params)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    params_history = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        event_tensor, loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width)
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

    return model, losses, params_history

def create_event_frames(model, x, y, t, p, H, W, bin_width, compensated=True):
    """
    Create event frames with or without compensation
    """
    xs = torch.tensor(x, device=device)
    ys = torch.tensor(y, device=device)
    ts = torch.tensor(t, device=device)
    ps = torch.tensor(p, device=device)
    
    with torch.no_grad():
        if compensated:
            # Use current model parameters
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width)
        else:
            # Temporarily set parameters to zero
            original_params = model.params.clone()
            model.params.data.zero_()
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width)
            # Restore parameters
            model.params.data = original_params
    
    return event_tensor

def visualize_results(model, x, y, t, p, losses, params_history, bin_width, 
                     sensor_width, sensor_height, output_dir=None, filename_prefix=""):
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
    
    # Generate event frames
    event_tensor_orig = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width, compensated=True)
    
    # Get actual number of bins from tensor shape
    actual_num_bins = event_tensor_orig.shape[0]
    
    # Select a middle time bin to visualize
    bin_idx = actual_num_bins // 2
    
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
        # Check tensor shapes before variance calculation
        print(f"Original tensor shape: {event_tensor_orig.shape}")
        print(f"Compensated tensor shape: {event_tensor_comp.shape}")
        
        # Ensure both tensors have the same shape
        if event_tensor_orig.shape != event_tensor_comp.shape:
            print("Warning: Original and compensated tensors have different shapes!")
            min_bins = min(event_tensor_orig.shape[0], event_tensor_comp.shape[0])
            event_tensor_orig = event_tensor_orig[:min_bins]
            event_tensor_comp = event_tensor_comp[:min_bins]
            print(f"Trimmed to {min_bins} bins")
            actual_num_bins = min_bins  # Update the actual number of bins
        
        # Ensure we have the right shape (num_bins, H, W)
        if len(event_tensor_orig.shape) == 3:
            current_num_bins, H, W = event_tensor_orig.shape
            # Reshape to (num_bins, H*W) for variance calculation
            var_orig_tensor = torch.var(event_tensor_orig.reshape(current_num_bins, H * W), dim=1)
            var_comp_tensor = torch.var(event_tensor_comp.reshape(current_num_bins, H * W), dim=1)
        else:
            # Fallback: calculate variance over the flattened spatial dimensions
            var_orig_tensor = torch.var(event_tensor_orig.view(event_tensor_orig.shape[0], -1), dim=1)
            var_comp_tensor = torch.var(event_tensor_comp.view(event_tensor_comp.shape[0], -1), dim=1)
        
        # Convert to lists for plotting (avoiding numpy arrays)
        var_orig_list = var_orig_tensor.cpu().tolist()
        var_comp_list = var_comp_tensor.cpu().tolist()
        
        # Calculate mean values using PyTorch
        var_orig_mean = var_orig_tensor.mean().item()
        var_comp_mean = var_comp_tensor.mean().item()
        
        # Update actual_num_bins after potential trimming
        actual_num_bins = event_tensor_orig.shape[0]
    
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
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.npz_file)
    
    # Create filename prefix
    base_name = os.path.splitext(os.path.basename(args.npz_file))[0]
    
    print(f"Analyzing: {args.npz_file}")
    
    # Load events
    x, y, t, p = load_npz_events(args.npz_file)
    
    # Initial parameters
    initial_params = [args.initial_a_x, args.initial_a_y]
    
    # Train model
    model, losses, params_history = train_scan_compensation(
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
    
    # Visualize if requested
    if args.visualize:
        visualize_results(model, x, y, t, p, losses, params_history, 
                         args.bin_width, args.sensor_width, args.sensor_height, 
                         args.output_dir, base_name)
    
    print("Scan compensation complete!")

if __name__ == "__main__":
    main()