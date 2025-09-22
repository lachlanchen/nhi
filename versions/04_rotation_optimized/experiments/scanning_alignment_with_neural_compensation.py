#!/usr/bin/env python3
"""
FIXED: Enhanced scan compensation code with RESIDUAL neural network-based time-dependent modulation phi(t)
t' = t - (ax*x + ay*y)*phi(t)
where phi(t) = 1 + epsilon * NN(t), forcing small smooth adjustments around 1

Key fixes:
1. Better network architecture with skip connections
2. Improved time input features (multiple frequencies)
3. Better initialization strategy
4. Adaptive learning rates
5. Loss that encourages beneficial temporal variation
6. Better regularization
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

class ImprovedPhiNetwork(nn.Module):
    """
    Improved Neural Network for phi(t) with better architecture and features
    """
    def __init__(self, hidden_dim=32, num_layers=3, epsilon=0.1, num_frequencies=4):
        super().__init__()
        
        self.epsilon = epsilon
        self.num_frequencies = num_frequencies
        
        # Input features: t, sin(2πft), cos(2πft) for multiple frequencies
        input_dim = 1 + 2 * num_frequencies
        
        print(f"Improved Phi Network: phi(t) = 1 + {epsilon} * tanh(NN(features))")
        print(f"Input features: time + {num_frequencies} frequency components")
        print(f"Output range: [{1-epsilon:.3f}, {1+epsilon:.3f}]")
        print(f"Architecture: {input_dim} -> {hidden_dim} -> ... ({num_layers} layers)")
        
        # Build network with residual connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize for stability
        self._init_weights()
    
    def _init_weights(self):
        """Careful initialization to start near phi=1"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with small scale
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Make output layer especially small to start near 1
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer.bias)
    
    def _create_time_features(self, t_norm):
        """Create rich time features with multiple frequency components"""
        features = [t_norm.unsqueeze(-1)]  # Base time feature
        
        # Add sinusoidal features at different frequencies
        for i in range(self.num_frequencies):
            freq = 2 ** i  # Frequencies: 1, 2, 4, 8, ...
            features.append(torch.sin(2 * np.pi * freq * t_norm).unsqueeze(-1))
            features.append(torch.cos(2 * np.pi * freq * t_norm).unsqueeze(-1))
        
        return torch.cat(features, dim=-1)
    
    def forward(self, t_norm):
        """
        Forward pass with rich features and residual connections
        """
        # Create rich input features
        x = self._create_time_features(t_norm)
        
        # Input layer
        x = torch.tanh(self.input_layer(x))
        identity = x
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            x_new = torch.tanh(layer(x))
            
            # Add residual connection every 2 layers
            if i % 2 == 1:
                x = x_new + identity
                identity = x
            else:
                x = x_new
        
        # Output layer with small activation
        raw_output = self.output_layer(x).squeeze(-1)
        
        # Bounded output: phi(t) = 1 + epsilon * tanh(output)
        phi_t = 1.0 + self.epsilon * torch.tanh(raw_output)
        
        return phi_t

class ScanCompensationWithImprovedPhi(nn.Module):
    def __init__(self, initial_params, phi_network_params=None, smoothness_weight=0.01, 
                 variation_weight=0.001):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight  # Encourage beneficial variation
        
        # Compensation parameters
        if isinstance(initial_params, torch.Tensor):
            if len(initial_params) < 2:
                padded_params = torch.zeros(2, dtype=torch.float32)
                padded_params[:len(initial_params)] = initial_params
                initial_params = padded_params
            self.compensation_params = nn.Parameter(initial_params[:2].clone().detach())
        else:
            if len(initial_params) < 2:
                padded_params = list(initial_params) + [0.0] * (2 - len(initial_params))
                initial_params = padded_params
            self.compensation_params = nn.Parameter(torch.tensor(initial_params[:2], dtype=torch.float32))
        
        # Improved phi network
        if phi_network_params is None:
            phi_network_params = {'hidden_dim': 32, 'num_layers': 3, 'epsilon': 0.1, 'num_frequencies': 4}
        
        self.phi_network = ImprovedPhiNetwork(**phi_network_params)
        
        print(f"Initialized compensation parameters (ax, ay) and improved phi network")
        print(f"Smoothness weight: {smoothness_weight}, Variation weight: {variation_weight}")
    
    def compute_phi(self, timestamps, t_start_original):
        """
        Compute phi(t) with improved features
        """
        # Subtract start time and normalize to [0, 1]
        t_relative = timestamps - t_start_original
        t_min = t_relative.min()
        t_max = t_relative.max()
        t_norm = (t_relative - t_min) / (t_max - t_min + 1e-8)
        
        # Compute phi(t) using improved network
        phi_t = self.phi_network(t_norm)
        
        return phi_t, t_norm, t_relative
    
    def warp(self, x_coords, y_coords, timestamps, t_start_original):
        """
        Apply compensation: t' = t - (ax*x + ay*y)*phi(t)
        """
        a_x = self.compensation_params[0]
        a_y = self.compensation_params[1]
        
        phi_t, t_norm, t_relative = self.compute_phi(timestamps, t_start_original)
        
        # Apply compensation
        compensation = (a_x * x_coords + a_y * y_coords) * phi_t
        t_warped = timestamps - compensation
        
        return x_coords, y_coords, t_warped, phi_t, t_norm, t_relative

    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, 
                original_t_start=None, original_t_end=None):
        """
        Forward pass with improved loss function
        """
        # Use original start time for phi computation
        t_start_for_phi = timestamps.min() if original_t_start is None else original_t_start
        
        x_warped, y_warped, t_warped, phi_t, t_norm, t_relative = self.warp(
            x_coords, y_coords, timestamps, t_start_for_phi)
        
        # Filter events to time range if provided
        if original_t_start is not None and original_t_end is not None:
            valid_time_mask = (t_warped >= original_t_start) & (t_warped <= original_t_end)
            x_warped = x_warped[valid_time_mask]
            y_warped = y_warped[valid_time_mask]
            t_warped = t_warped[valid_time_mask]
            polarities = polarities[valid_time_mask]
            phi_t = phi_t[valid_time_mask]
            t_norm = t_norm[valid_time_mask]
            t_start = original_t_start
            t_end = original_t_end
        else:
            t_start = t_warped.min()
            t_end = t_warped.max()
        
        # Create event tensor (same as original)
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1
        t_norm_bins = (t_warped - t_start) / time_bin_width
        t0 = torch.floor(t_norm_bins)
        t1 = t0 + 1
        wt = (t_norm_bins - t0).float()
        t0_clamped = t0.clamp(0, num_bins - 1).long()
        t1_clamped = t1.clamp(0, num_bins - 1).long()

        x_indices = x_warped.long()
        y_indices = y_warped.long()
        valid_mask = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]
        phi_t = phi_t[valid_mask]
        t_norm = t_norm[valid_mask]

        spatial_indices = y_indices * W + x_indices
        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        weights_t0 = ((1 - wt) * polarities).float()
        weights_t1 = (wt * polarities).float()

        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)
        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        event_tensor = event_tensor_flat.view(num_bins, H, W)
        
        # Primary loss: variance (negative because we want to maximize)
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        baseline_variance = torch.sum(variances)
        
        # Smoothness regularization: penalize rapid changes in phi(t)
        smoothness_loss = torch.tensor(0.0, device=device)
        if len(phi_t) > 1 and len(t_norm) > 1:
            # Sort by normalized time for proper smoothness calculation
            sorted_indices = torch.argsort(t_norm)
            phi_sorted = phi_t[sorted_indices]
            
            # Calculate smoothness based on adjacent time points
            if len(phi_sorted) > 1:
                phi_diffs = phi_sorted[1:] - phi_sorted[:-1]
                smoothness_loss = torch.mean(phi_diffs ** 2)
        
        # Variation penalty: discourage phi(t) from being constant unless beneficial
        variation_loss = torch.tensor(0.0, device=device)
        if len(phi_t) > 1:
            phi_std = torch.std(phi_t)
            # Small penalty if phi is too constant but variance reduction is minimal
            if phi_std < 0.01:  # Very little variation
                variation_loss = 0.01 - phi_std  # Encourage some variation
        
        # Total loss
        variance_loss = baseline_variance
        total_loss = (variance_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.variation_weight * variation_loss)
        
        return event_tensor, total_loss, phi_t, t_norm, variance_loss, smoothness_loss, variation_loss

def train_scan_compensation_with_improved_phi(x, y, t, p, sensor_width=1280, sensor_height=720, 
                                            bin_width=1e5, num_iterations=1000, learning_rate=0.5,
                                            initial_params=None, phi_network_params=None, 
                                            smoothness_weight=0.01, variation_weight=0.001):
    """
    Train with improved phi network
    """
    print(f"Training scan compensation with Improved Neural Network phi(t)...")
    print(f"Sensor size: {sensor_width} x {sensor_height}")
    print(f"Bin width: {bin_width/1000:.1f} ms")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Smoothness weight: {smoothness_weight}")
    print(f"Variation weight: {variation_weight}")
    
    # Convert to tensors
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    # Store time range and normalize
    original_t_start = torch.tensor(float(ts.min().item()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(ts.max().item()), device=device, dtype=torch.float32)
    
    ts_normalized = ts - original_t_start
    original_t_start_normalized = torch.tensor(0.0, device=device, dtype=torch.float32)
    original_t_end_normalized = original_t_end - original_t_start
    
    print(f"Original time range: {original_t_start.item():.0f} - {original_t_end.item():.0f} μs")
    print(f"Normalized time range: {original_t_start_normalized.item():.0f} - {original_t_end_normalized.item():.0f} μs")

    # Initialize parameters
    if initial_params is None:
        initial_params = torch.zeros(2, device=device, dtype=torch.float32)
    else:
        initial_params = torch.tensor(initial_params[:2], device=device, dtype=torch.float32)
    
    model = ScanCompensationWithImprovedPhi(initial_params, phi_network_params, 
                                           smoothness_weight, variation_weight)
    model.to(device)

    # Optimizer with adaptive learning rates
    optimizer = torch.optim.Adam([
        {'params': [model.compensation_params], 'lr': learning_rate},
        {'params': model.phi_network.parameters(), 'lr': learning_rate * 0.01}  # Much smaller for phi
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                                          patience=100)

    # Training loop
    losses = []
    variance_losses = []
    smoothness_losses = []
    variation_losses = []
    params_history = []
    phi_history = []

    sample_times = torch.linspace(ts_normalized.min(), ts_normalized.max(), 100, device=device, dtype=torch.float32)
    best_loss = float('inf')
    patience_counter = 0

    for i in range(num_iterations):
        optimizer.zero_grad()
        
        event_tensor, total_loss, phi_t, t_norm, var_loss, smooth_loss, var_penalty = model(
            xs, ys, ts_normalized, ps, sensor_height, sensor_width, bin_width, 
            original_t_start_normalized, original_t_end_normalized)
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        current_loss = total_loss.item()
        current_var_loss = var_loss.item()
        current_smooth_loss = smooth_loss.item()
        current_var_penalty = var_penalty.item()
        
        losses.append(current_loss)
        variance_losses.append(current_var_loss)
        smoothness_losses.append(current_smooth_loss)
        variation_losses.append(current_var_penalty)
        
        current_params = model.compensation_params.detach().cpu()
        params_history.append([current_params[j].item() for j in range(len(current_params))])
        
        # Sample phi(t) for visualization
        if i % 25 == 0:
            with torch.no_grad():
                sample_phi, sample_t_norm, sample_t_rel = model.compute_phi(sample_times, torch.tensor(0.0, device=device))
                phi_history.append(sample_phi.cpu().numpy())
        
        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if i % 50 == 0:
            param_str = f"ax={current_params[0].item():.6f}, ay={current_params[1].item():.6f}"
            phi_min = phi_t.min().item()
            phi_max = phi_t.max().item()
            phi_mean = phi_t.mean().item()
            phi_std = phi_t.std().item()
            print(f"Iteration {i}: Loss={current_loss:.6f}, Var={current_var_loss:.6f}, "
                  f"Smooth={current_smooth_loss:.6f}, VarPen={current_var_penalty:.6f}")
            print(f"  {param_str}")
            print(f"  phi(t): [{phi_min:.4f}, {phi_max:.4f}], mean={phi_mean:.4f}±{phi_std:.4f}")
        
        # Early stopping
        if patience_counter > 200:
            print(f"Early stopping at iteration {i}")
            break

    loss_components = {
        'total': losses,
        'variance': variance_losses,
        'smoothness': smoothness_losses,
        'variation': variation_losses
    }

    return model, loss_components, params_history, phi_history, sample_times, original_t_start, original_t_end

def create_event_frames_with_improved_phi(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end, compensated=True):
    """Create event frames with/without compensation"""
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    
    ts_normalized = ts - original_t_start
    original_t_start_normalized = torch.tensor(0.0, device=device, dtype=torch.float32)
    original_t_end_normalized = original_t_end - original_t_start
    
    with torch.no_grad():
        if compensated:
            event_tensor, _, phi_t, t_norm, _, _, _ = model(xs, ys, ts_normalized, ps, H, W, bin_width, 
                                                        original_t_start_normalized, original_t_end_normalized)
        else:
            original_params = model.compensation_params.clone()
            model.compensation_params.data.zero_()
            event_tensor, _, phi_t, t_norm, _, _, _ = model(xs, ys, ts_normalized, ps, H, W, bin_width, 
                                                        original_t_start_normalized, original_t_end_normalized)
            model.compensation_params.data = original_params
    
    return event_tensor, phi_t, t_norm

def get_param_string_with_improved_phi(model):
    """Get parameter string"""
    params = model.compensation_params.detach().cpu()
    a_x = params[0].item()
    a_y = params[1].item()
    epsilon = model.phi_network.epsilon
    return f"ax={a_x:.4f}, ay={a_y:.4f}, ImprovedNN_phi(ε={epsilon})"

def get_param_suffix_with_improved_phi(model):
    """Get filename suffix"""
    params = model.compensation_params.detach().cpu()
    a_x = params[0].item()
    a_y = params[1].item()
    epsilon = model.phi_network.epsilon
    return f"_ax{a_x:.4f}_ay{a_y:.4f}_ImpNN_eps{epsilon}"

def visualize_improved_phi_function(model, sample_times, phi_history, loss_components, params_history, 
                                   output_dir=None, filename_prefix=""):
    """Visualize improved phi results"""
    times_tensor = sample_times.cpu()
    times_list = times_tensor.tolist()
    
    with torch.no_grad():
        current_phi, t_norm, t_rel = model.compute_phi(sample_times, torch.tensor(0.0, device=device))
        current_phi_list = current_phi.cpu().tolist()
        t_norm_list = t_norm.cpu().tolist()
    
    params = model.compensation_params.detach().cpu()
    a_x = float(params[0].item())
    a_y = float(params[1].item())
    epsilon = model.phi_network.epsilon
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Phi(t) vs time
    axes[0, 0].plot([t / 1000 for t in times_list], current_phi_list, 'b-', linewidth=2, label='φ(t) Improved NN')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='φ=1')
    axes[0, 0].axhline(y=1-epsilon, color='g', linestyle=':', alpha=0.7, label=f'φ={1-epsilon:.2f}')
    axes[0, 0].axhline(y=1+epsilon, color='g', linestyle=':', alpha=0.7, label=f'φ={1+epsilon:.2f}')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('φ(t)')
    axes[0, 0].set_title(f'Improved Neural Network φ(t) (ε={epsilon})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(1-epsilon-0.02, 1+epsilon+0.02)
    
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
    axes[0, 2].plot([t / 1000 for t in times_list], ay_phi_t, 'g-', linewidth=2, label=f'ay·φ(t)')
    axes[0, 2].axhline(y=a_y, color='g', linestyle='--', alpha=0.7, label=f'ay={a_y:.4f}')
    axes[0, 2].set_xlabel('Time (ms)')
    axes[0, 2].set_ylabel('ay·φ(t)')
    axes[0, 2].set_title('Y-axis Compensation Factor')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # 4. Training losses
    axes[1, 0].plot(loss_components['total'], 'k-', linewidth=1, label='Total Loss')
    axes[1, 0].plot(loss_components['variance'], 'b-', linewidth=1, label='Variance Loss')
    axes[1, 0].plot(loss_components['smoothness'], 'r-', linewidth=1, label='Smoothness')
    axes[1, 0].plot(loss_components['variation'], 'm-', linewidth=1, label='Variation Penalty')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    
    # 5. Parameter evolution
    params_array = np.array(params_history)
    axes[1, 1].plot(params_array[:, 0], label='ax', color='r')
    axes[1, 1].plot(params_array[:, 1], label='ay', color='g')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('Parameter Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Phi evolution over training
    if phi_history and len(phi_history) > 1:
        phi_array = np.array(phi_history)
        for i in range(0, len(phi_history), max(1, len(phi_history)//8)):
            alpha = 0.3 + 0.7 * i / (len(phi_history) - 1)
            color = plt.cm.viridis(i / (len(phi_history) - 1))
            axes[1, 2].plot([t / 1000 for t in times_list], phi_array[i], alpha=alpha, 
                           color=color, label=f'Iter {i*25}' if i < 8 else None)
        axes[1, 2].set_xlabel('Time (ms)')
        axes[1, 2].set_ylabel('φ(t)')
        axes[1, 2].set_title('φ(t) Evolution During Training')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(1-epsilon-0.02, 1+epsilon+0.02)
    else:
        axes[1, 2].text(0.5, 0.5, 'Phi evolution\nnot available', transform=axes[1, 2].transAxes, 
                       ha='center', va='center')
        axes[1, 2].set_title('φ(t) Evolution')
    
    # Statistics
    phi_min = min(current_phi_list)
    phi_max = max(current_phi_list)
    phi_mean = sum(current_phi_list) / len(current_phi_list)
    phi_std = (sum((x - phi_mean) ** 2 for x in current_phi_list) / len(current_phi_list)) ** 0.5
    
    fig.suptitle(f'Improved Neural Network Scan Compensation\n' +
                f'φ(t) range=[{phi_min:.4f}, {phi_max:.4f}], mean={phi_mean:.4f}±{phi_std:.4f}', 
                fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix_with_improved_phi(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_improved_phi_analysis{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Improved phi analysis plot saved to: {plot_path}")
    
    plt.show()
    
    return current_phi_list, times_list

def visualize_results_with_improved_phi(model, x, y, t, p, loss_components, params_history, bin_width, 
                                       sensor_width, sensor_height, original_t_start, original_t_end, 
                                       output_dir=None, filename_prefix=""):
    """Visualize results"""
    param_str = get_param_string_with_improved_phi(model)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(loss_components['total'], 'k-', linewidth=1, label='Total')
    axes[0, 0].plot(loss_components['variance'], 'b-', linewidth=1, label='Variance')
    axes[0, 0].plot(loss_components['smoothness'], 'r-', linewidth=1, label='Smoothness')
    axes[0, 0].plot(loss_components['variation'], 'm-', linewidth=1, label='Variation')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    
    # Parameters evolution
    params_array = np.array(params_history)
    axes[0, 1].plot(params_array[:, 0], label='ax', color='r')
    axes[0, 1].plot(params_array[:, 1], label='ay', color='g')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Parameter Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Generate event frames
    event_tensor_orig, _, _ = create_event_frames_with_improved_phi(
        model, x, y, t, p, sensor_height, sensor_width, bin_width, 
        original_t_start, original_t_end, compensated=False)
    event_tensor_comp, phi_t, t_norm = create_event_frames_with_improved_phi(
        model, x, y, t, p, sensor_height, sensor_width, bin_width, 
        original_t_start, original_t_end, compensated=True)
    
    actual_num_bins = event_tensor_orig.shape[0]
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
        if event_tensor_orig.shape != event_tensor_comp.shape:
            min_bins = min(event_tensor_orig.shape[0], event_tensor_comp.shape[0])
            event_tensor_orig = event_tensor_orig[:min_bins]
            event_tensor_comp = event_tensor_comp[:min_bins]
            actual_num_bins = min_bins
        
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
    final_params_tensor = model.compensation_params.detach().cpu()
    improvement_pct = (var_comp_mean/var_orig_mean - 1) * 100
    
    phi_min = phi_t.min().item()
    phi_max = phi_t.max().item()
    phi_mean = phi_t.mean().item()
    phi_std = phi_t.std().item()
    
    stats_text = f'Original mean variance: {var_orig_mean:.2f}\n'
    stats_text += f'Compensated mean variance: {var_comp_mean:.2f}\n'
    stats_text += f'Improvement: {improvement_pct:.1f}%\n'
    stats_text += f'Final ax: {final_params_tensor[0].item():.4f}\n'
    stats_text += f'Final ay: {final_params_tensor[1].item():.4f}\n'
    stats_text += f'Final loss: {loss_components["total"][-1]:.6f}\n'
    stats_text += f'φ(t): [{phi_min:.4f}, {phi_max:.4f}]\n'
    stats_text += f'φ(t) mean: {phi_mean:.4f}±{phi_std:.4f}\n'
    stats_text += f'Improved NN (ε={model.phi_network.epsilon})\n'
    stats_text += f'Total events: {len(x):,}\n'
    stats_text += f'Time bins: {actual_num_bins}'
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    fig.suptitle(f'Improved Neural Network Scan Compensation Results\n{param_str}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if output_dir:
        param_suffix = get_param_suffix_with_improved_phi(model)
        plot_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results_improved_phi{param_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
    
    plt.show()

def save_results_with_improved_phi(model, loss_components, params_history, output_dir, filename_prefix):
    """Save results"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        param_suffix = get_param_suffix_with_improved_phi(model)
        final_params = model.compensation_params.detach().cpu()
        results_path = os.path.join(output_dir, f"{filename_prefix}_scan_compensation_results_improved_phi{param_suffix}.txt")
        
        with open(results_path, 'w') as f:
            f.write("SCAN COMPENSATION RESULTS WITH IMPROVED NEURAL NETWORK PHI(T)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Improved Neural Network Architecture:\n")
            f.write(f"  Formula: φ(t) = 1 + {model.phi_network.epsilon} * tanh(NN(features))\n")
            f.write(f"  Output range: [{1-model.phi_network.epsilon:.3f}, {1+model.phi_network.epsilon:.3f}]\n")
            f.write(f"  Features: time + {model.phi_network.num_frequencies} frequency components\n")
            f.write(f"  Hidden layers: {len(model.phi_network.hidden_layers) + 1}\n")
            f.write(f"  Smoothness weight: {model.smoothness_weight}\n")
            f.write(f"  Variation weight: {model.variation_weight}\n\n")
            
            f.write(f"Final compensation parameters:\n")
            f.write(f"  ax = {final_params[0].item():.6f}\n")
            f.write(f"  ay = {final_params[1].item():.6f}\n\n")
            
            f.write(f"Final losses:\n")
            f.write(f"  Total loss: {loss_components['total'][-1]:.6f}\n")
            f.write(f"  Variance loss: {loss_components['variance'][-1]:.6f}\n")
            f.write(f"  Smoothness loss: {loss_components['smoothness'][-1]:.6f}\n")
            f.write(f"  Variation loss: {loss_components['variation'][-1]:.6f}\n\n")
            
            f.write(f"Training iterations: {len(loss_components['total'])}\n")
            
            f.write("\nParameter evolution (every 100 iterations):\n")
            for i, params in enumerate(params_history[::100]):
                f.write(f"Iteration {i*100}: ax={params[0]:.6f}, ay={params[1]:.6f}\n")
        
        print(f"Results saved to: {results_path}")
        
        # Save model
        model_path = os.path.join(output_dir, f"{filename_prefix}_improved_phi_model{param_suffix}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'compensation_params': final_params,
            'loss_components': loss_components,
            'params_history': params_history,
            'epsilon': model.phi_network.epsilon,
            'smoothness_weight': model.smoothness_weight,
            'variation_weight': model.variation_weight,
            'num_frequencies': model.phi_network.num_frequencies
        }, model_path)
        print(f"Model saved to: {model_path}")

def main_with_improved_phi():
    parser = argparse.ArgumentParser(description='Scan compensation with Improved Neural Network phi(t)')
    parser.add_argument('input_path', help='Path to NPZ event file OR segments folder (when using --merge)')
    parser.add_argument('--merge', action='store_true', help='Merge all scan segments from folder')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    parser.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--initial_a_x', type=float, default=0.0, help='Initial a_x parameter')
    parser.add_argument('--initial_a_y', type=float, default=0.0, help='Initial a_y parameter')
    parser.add_argument('--phi_hidden_dim', type=int, default=32, help='Hidden dimension for phi network')
    parser.add_argument('--phi_num_layers', type=int, default=3, help='Number of layers for phi network')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Maximum deviation from phi=1')
    parser.add_argument('--num_frequencies', type=int, default=4, help='Number of frequency components')
    parser.add_argument('--smoothness_weight', type=float, default=0.01, help='Weight for smoothness regularization')
    parser.add_argument('--variation_weight', type=float, default=0.001, help='Weight for variation penalty')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--visualize_phi', action='store_true', help='Show detailed phi(t) analysis plots')
    
    args = parser.parse_args()
    
    # Handle input path
    if args.merge:
        segments_folder = args.input_path
        if not os.path.isdir(segments_folder):
            raise ValueError(f"When using --merge, input_path must be a directory: {segments_folder}")
        
        if args.output_dir is None:
            args.output_dir = segments_folder
        
        folder_name = os.path.basename(segments_folder.rstrip('/'))
        base_name = f"{folder_name}_merged_improved_phi"
        
        print(f"Merging segments from: {segments_folder}")
        x, y, t, p = load_and_merge_segments(segments_folder)
        
    else:
        npz_file = args.input_path
        if not os.path.isfile(npz_file):
            raise ValueError(f"NPZ file not found: {npz_file}")
        
        if args.output_dir is None:
            args.output_dir = os.path.dirname(npz_file)
        
        base_name = os.path.splitext(os.path.basename(npz_file))[0] + "_improved_phi"
        
        print(f"Analyzing: {npz_file}")
        x, y, t, p = load_npz_events(npz_file)
    
    # Parameters
    initial_params = [args.initial_a_x, args.initial_a_y]
    phi_network_params = {
        'hidden_dim': args.phi_hidden_dim,
        'num_layers': args.phi_num_layers,
        'epsilon': args.epsilon,
        'num_frequencies': args.num_frequencies
    }
    
    print(f"Training with Improved Neural Network phi(t)...")
    print(f"Initial compensation parameters: {initial_params}")
    print(f"Phi network parameters: {phi_network_params}")
    print(f"Smoothness weight: {args.smoothness_weight}")
    print(f"Variation weight: {args.variation_weight}")
    
    # Train model
    model, loss_components, params_history, phi_history, sample_times, original_t_start, original_t_end = train_scan_compensation_with_improved_phi(
        x, y, t, p,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        bin_width=args.bin_width,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        initial_params=initial_params,
        phi_network_params=phi_network_params,
        smoothness_weight=args.smoothness_weight,
        variation_weight=args.variation_weight
    )
    
    # Print final results
    final_params_tensor = model.compensation_params.detach().cpu()
    
    print(f"\nFinal compensation parameters:")
    print(f"  ax = {final_params_tensor[0].item():.6f}")
    print(f"  ay = {final_params_tensor[1].item():.6f}")
    print(f"Final losses:")
    print(f"  Total loss: {loss_components['total'][-1]:.6f}")
    print(f"  Variance loss: {loss_components['variance'][-1]:.6f}")
    print(f"  Smoothness loss: {loss_components['smoothness'][-1]:.6f}")
    print(f"  Variation loss: {loss_components['variation'][-1]:.6f}")
    
    # Show phi(t) range
    with torch.no_grad():
        sample_times_full = torch.linspace(0, (original_t_end - original_t_start), 1000, device=device)
        phi_full, _, _ = model.compute_phi(sample_times_full, torch.tensor(0.0, device=device))
        phi_min = phi_full.min().item()
        phi_max = phi_full.max().item()
        phi_mean = phi_full.mean().item()
        phi_std = phi_full.std().item()
        print(f"Final phi(t): [{phi_min:.4f}, {phi_max:.4f}], mean={phi_mean:.4f}±{phi_std:.4f}")
        print(f"Theoretical range: [{1-args.epsilon:.3f}, {1+args.epsilon:.3f}]")
    
    # Save results
    save_results_with_improved_phi(model, loss_components, params_history, args.output_dir, base_name)
    
    # Visualizations
    if args.visualize_phi:
        visualize_improved_phi_function(model, sample_times, phi_history, loss_components, params_history, 
                                       args.output_dir, base_name)
    
    if args.visualize:
        visualize_results_with_improved_phi(model, x, y, t, p, loss_components, params_history, 
                                           args.bin_width, args.sensor_width, args.sensor_height, 
                                           original_t_start, original_t_end, args.output_dir, base_name)
    
    print("Scan compensation with Improved Neural Network phi(t) complete!")

if __name__ == "__main__":
    main_with_improved_phi()