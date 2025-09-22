import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

class Compensate(nn.Module):
    def __init__(self, a_params, b_params, duration, temperature=1000, device='cpu'):
        """
        3D Multi-window compensation class with PyTorch and gradients
        
        Args:
            a_params: X-direction coefficients (len determines number of boundaries)
            b_params: Y-direction coefficients (same length as a_params)  
            duration: Time duration in microseconds (max_t - min_t) after shifting
            temperature: Smoothness parameter in microseconds (default 1000)
            device: PyTorch device ('cpu' or 'cuda')
        """
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
        self.num_main_windows = self.num_boundaries - 3  # n_main = len(ax) - 3
        self.num_total_windows = self.num_main_windows + 2  # Add 2 edge windows
        
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")
        
        # Calculate main window size based on main windows only (in microseconds)
        self.main_window_size = self.duration / self.num_main_windows
        
        # Calculate boundary offsets as fixed tensor (no gradients needed)
        boundary_offsets = torch.tensor([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ], dtype=torch.float32, device=device)
        
        self.register_buffer('boundary_offsets', boundary_offsets)
        
        print(f"Compensate initialized (PyTorch):")
        print(f"  Duration: {self.duration:.0f} μs ({self.duration/1000:.1f} ms)")
        print(f"  Parameters per direction: {self.num_boundaries}")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.num_total_windows} (including 2 edge windows)")
        print(f"  Main window size: {self.main_window_size:.0f} μs ({self.main_window_size/1000:.1f} ms)")
        print(f"  Device: {device}")
        print(f"  A parameters: {self.a_params.data.cpu().numpy()}")
        print(f"  B parameters: {self.b_params.data.cpu().numpy()}")
        print(f"  Temperature: {self.temperature:.0f} μs")
    
    def get_boundary_surfaces(self, x, y, normalize=False, width=None, height=None):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        
        Args:
            x: X coordinates (tensor or array)
            y: Y coordinates (tensor or array)
            normalize: Whether to normalize x,y by width,height (default False)
            width: Sensor width for normalization (required if normalize=True)
            height: Sensor height for normalization (required if normalize=True)
        """
        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if normalize:
            if width is None or height is None:
                raise ValueError("width and height required when normalize=True")
            x = x / width
            y = y / height
        
        if x.dim() == 1 and y.dim() == 1:
            # Create meshgrid
            X, Y = torch.meshgrid(x, y, indexing='ij')
            boundary_values = torch.zeros(self.num_boundaries, len(x), len(y), 
                                        dtype=torch.float32, device=self.device)
            for i in range(self.num_boundaries):
                boundary_values[i] = (self.a_params[i] * X + 
                                    self.b_params[i] * Y + 
                                    self.boundary_offsets[i])
        else:
            raise ValueError("x and y must be 1D tensors/arrays")
        
        return boundary_values
    
    def compute_window_memberships(self, x, y, t_shifted, normalize=False, width=None, height=None):
        """
        Compute soft membership for each window
        """
        boundary_values = self.get_boundary_surfaces(x, y, normalize, width, height)
        
        # Convert t_shifted to tensor
        if not isinstance(t_shifted, torch.Tensor):
            t_shifted = torch.tensor(t_shifted, dtype=torch.float32, device=self.device)
        
        # Create 3D meshgrids
        if normalize and width is not None and height is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x / width
            y_use = y / height
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x
            y_use = y
            
        X, Y, T = torch.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_total_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, None]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, None]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions
            lower_sigmoid = torch.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = torch.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = torch.stack(memberships, dim=0)  # [num_total_windows, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = torch.sum(memberships, dim=0, keepdim=True)  # [1, len_x, len_y, len_t]
        memberships_sum = torch.clamp(memberships_sum, min=1e-8)
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t_shifted, normalize=False, width=None, height=None):
        """
        Interpolate parameters within each window
        """
        boundary_values = self.get_boundary_surfaces(x, y, normalize, width, height)
        
        # Convert to tensors
        if not isinstance(t_shifted, torch.Tensor):
            t_shifted = torch.tensor(t_shifted, dtype=torch.float32, device=self.device)
        
        # Create meshgrids
        if normalize and width is not None and height is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x / width
            y_use = y / height
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x
            y_use = y
            
        X, Y, T = torch.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        interpolated_slopes_a = []
        interpolated_slopes_b = []
        
        for i in range(self.num_total_windows):
            lower_bound = boundary_values[i][:, :, None]
            upper_bound = boundary_values[i + 1][:, :, None]
            
            window_width = upper_bound - lower_bound
            window_width = torch.clamp(window_width, min=1e-8)
            
            # Interpolation parameter (no clamping)
            alpha = (T - lower_bound) / window_width
            
            # Interpolate between parameters
            slope_a = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            slope_b = (1 - alpha) * self.b_params[i] + alpha * self.b_params[i + 1]
            
            interpolated_slopes_a.append(slope_a)
            interpolated_slopes_b.append(slope_b)
        
        return torch.stack(interpolated_slopes_a, dim=0), torch.stack(interpolated_slopes_b, dim=0)
    
    def forward(self, x, y, t, width=None, height=None, normalize=False):
        """
        Compute compensation for given (x, y, t) coordinates
        
        Args:
            x: X coordinates (tensor/array)
            y: Y coordinates (tensor/array)
            t: T coordinates (tensor/array)
            width: Sensor width (required if normalize=True)
            height: Sensor height (required if normalize=True)
            normalize: Whether to normalize spatial coordinates (default False)
            
        Returns:
            compensation: Total compensation tensor
        """
        # Convert to tensors
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        # Shift time to start from 0
        t_min = torch.min(t)
        t_shifted = t - t_min
        
        # Verify duration matches approximately
        actual_duration = torch.max(t_shifted).item()
        if abs(actual_duration - self.duration) > self.duration * 0.01:  # 1% tolerance
            print(f"Warning: Actual duration {actual_duration:.0f} μs != expected {self.duration:.0f} μs")
        
        # Get window memberships
        memberships = self.compute_window_memberships(x, y, t_shifted, normalize, width, height)
        
        # Get interpolated slopes
        slopes_a, slopes_b = self.compute_within_window_interpolation(x, y, t_shifted, normalize, width, height)
        
        # Create meshgrids
        if normalize and width is not None and height is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x / width
            y_use = y / height
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            x_use = x
            y_use = y
        
        X, Y, T = torch.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        # Weighted sum of compensations
        compensation_x = torch.sum(memberships * slopes_a * X[None, :, :, :], dim=0)
        compensation_y = torch.sum(memberships * slopes_b * Y[None, :, :, :], dim=0)
        compensation = compensation_x + compensation_y
        
        return compensation

def create_realistic_example():
    """
    Create realistic example: 884ms duration, 10 main windows, 1280x720 sensor
    ax: 0 to 10, ay: -60 to -80 (in μs units)
    Simple time range: 0 to 884ms
    """
    duration = 884000.0  # 884ms in μs
    n_main_windows = 10
    n_boundaries = n_main_windows + 3  # 13 boundaries
    
    # Create ax parameters: 0 to 10 μs
    a_params = np.linspace(0, 10, n_boundaries).tolist()
    
    # Create ay parameters: -60 to -80 μs
    b_params = np.linspace(-60, -80, n_boundaries).tolist()
    
    print(f"Realistic example parameters:")
    print(f"  Duration: {duration:.0f} μs ({duration/1000:.0f} ms)")
    print(f"  Time range: 0 to {duration/1000:.0f} ms")
    print(f"  Main windows: {n_main_windows}")
    print(f"  Boundaries needed: {n_boundaries}")
    print(f"  Window size: {duration/n_main_windows:.0f} μs ({duration/n_main_windows/1000:.1f} ms)")
    print(f"  A parameters (μs): {a_params}")
    print(f"  B parameters (μs): {b_params}")
    
    return a_params, b_params, duration

def visualize_3d_membership(comp, x_range, y_range, t_range):
    """
    Visualize 3D membership functions - show how membership varies with TIME
    """
    print("Visualizing 3D membership functions...")
    print(f"Time range: {t_range.min()/1000:.0f} to {t_range.max()/1000:.0f} ms")
    
    # Compute memberships
    t_shifted = t_range - t_range[0]
    with torch.no_grad():  # No gradients needed for visualization
        memberships = comp.compute_window_memberships(x_range, y_range, t_shifted, normalize=False)
    
    # Convert to numpy for plotting
    memberships_np = memberships.cpu().numpy()
    x_range_np = np.asarray(x_range)
    y_range_np = np.asarray(y_range)
    t_range_np = np.asarray(t_range)
    
    print(f"Membership shape: {memberships_np.shape}")
    print(f"Number of windows: {comp.num_total_windows}")
    print(f"Window structure: 1 edge + {comp.num_main_windows} main + 1 edge = {comp.num_total_windows} total")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Select different spatial points to show how membership varies with time
    x_positions = [len(x_range_np)//4, len(x_range_np)//2, 3*len(x_range_np)//4]
    y_positions = [len(y_range_np)//4, len(y_range_np)//2, 3*len(y_range_np)//4]
    
    # Show membership vs TIME at different spatial locations
    for i, (x_idx, y_idx) in enumerate([(x_positions[0], y_positions[0]),
                                        (x_positions[1], y_positions[1]), 
                                        (x_positions[2], y_positions[2])]):
        ax = plt.subplot(2, 3, i + 1)
        
        # Plot membership of each window vs time at this spatial location
        colors = plt.cm.Set1(np.linspace(0, 1, comp.num_total_windows))
        
        for w in range(comp.num_total_windows):
            membership_vs_time = memberships_np[w, x_idx, y_idx, :]
            if w == 0:
                window_type = "Edge-Start"
            elif w == comp.num_total_windows - 1:
                window_type = "Edge-End"
            else:
                window_type = f"Main-{w}"
                
            ax.plot(t_range_np/1000, membership_vs_time, 
                   color=colors[w], linewidth=2, 
                   label=f'W{w} ({window_type})')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membership (0 to 1)')
        ax.set_title(f'Window Membership vs Time\nX={x_range_np[x_idx]:.0f}, Y={y_range_np[y_idx]:.0f}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(-0.05, 1.05)
        
        # Add boundary markers
        with torch.no_grad():
            boundary_at_point = comp.get_boundary_surfaces([x_range_np[x_idx]], [y_range_np[y_idx]], normalize=False)
            boundary_at_point_np = boundary_at_point.cpu().numpy()
        
        for b_idx, boundary_t in enumerate(boundary_at_point_np[:, 0, 0]):
            if t_range_np.min() <= boundary_t <= t_range_np.max():
                ax.axvline(boundary_t/1000, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Print dominant windows at this location
        dominant_times = []
        for t_test in [0.2, 0.4, 0.6, 0.8]:  # Test at different fractions of time
            t_idx = int(t_test * len(t_range_np))
            window_memberships = memberships_np[:, x_idx, y_idx, t_idx]
            dominant_window = np.argmax(window_memberships)
            dominant_times.append(f"{t_range_np[t_idx]/1000:.0f}ms→W{dominant_window}")
        
        print(f"  At (X={x_range_np[x_idx]:.0f}, Y={y_range_np[y_idx]:.0f}): {', '.join(dominant_times)}")
    
    # Show total membership sum (should be 1.0 everywhere)
    ax4 = plt.subplot(2, 3, 4)
    total_membership = np.sum(memberships_np, axis=0)  # Sum over windows
    
    # Show total membership vs time at center point
    x_center, y_center = len(x_range_np)//2, len(y_range_np)//2
    ax4.plot(t_range_np/1000, total_membership[x_center, y_center, :], 'b-', linewidth=3)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Total Membership')
    ax4.set_title(f'Total Membership vs Time\n(should be 1.0 everywhere)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.98, 1.02)
    
    # Show window boundaries in X-T space (averaged over Y)
    ax5 = plt.subplot(2, 3, 5)
    
    # Show the boundary lines in X-T space
    y_avg_idx = len(y_range_np) // 2
    with torch.no_grad():
        boundary_surfaces = comp.get_boundary_surfaces(x_range_np, [y_range_np[y_avg_idx]], normalize=False)
        boundary_surfaces_np = boundary_surfaces.cpu().numpy()
    
    colors_boundary = plt.cm.tab10(np.linspace(0, 1, len(boundary_surfaces_np)))
    for b_idx in range(len(boundary_surfaces_np)):
        boundary_line = boundary_surfaces_np[b_idx][:, 0]  # Extract line at avg Y
        ax5.plot(x_range_np, boundary_line/1000, '--', alpha=0.8, linewidth=2, 
                color=colors_boundary[b_idx], label=f'Boundary {b_idx}')
    
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Time (ms)')
    ax5.set_title(f'Boundary Lines in X-T space\n(at Y={y_range_np[y_avg_idx]:.0f})')
    ax5.grid(True, alpha=0.3)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_ylim(t_range_np.min()/1000, t_range_np.max()/1000)
    
    # Show window boundaries in Y-T space (averaged over X)
    ax6 = plt.subplot(2, 3, 6)
    
    x_avg_idx = len(x_range_np) // 2
    with torch.no_grad():
        boundary_surfaces = comp.get_boundary_surfaces([x_range_np[x_avg_idx]], y_range_np, normalize=False)
        boundary_surfaces_np = boundary_surfaces.cpu().numpy()
    
    for b_idx in range(len(boundary_surfaces_np)):
        boundary_line = boundary_surfaces_np[b_idx][0, :]  # Extract line at avg X
        ax6.plot(y_range_np, boundary_line/1000, '--', alpha=0.8, linewidth=2, 
                color=colors_boundary[b_idx], label=f'Boundary {b_idx}')
    
    ax6.set_xlabel('Y (pixels)')
    ax6.set_ylabel('Time (ms)')
    ax6.set_title(f'Boundary Lines in Y-T space\n(at X={x_range_np[x_avg_idx]:.0f})')
    ax6.grid(True, alpha=0.3)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.set_ylim(t_range_np.min()/1000, t_range_np.max()/1000)
    
    plt.tight_layout()
    plt.show()
    
    # Verify membership sum
    total_membership_all = np.sum(memberships_np, axis=0)
    print(f"\nMembership Verification:")
    print(f"✓ Total membership range: {total_membership_all.min():.6f} to {total_membership_all.max():.6f}")
    print(f"✓ Mean total membership: {total_membership_all.mean():.6f} (should be 1.0)")
    print(f"✓ Max deviation from 1.0: {abs(total_membership_all - 1.0).max():.6f}")
    print(f"✓ Membership sum correct: {'✓' if abs(total_membership_all.mean() - 1.0) < 0.01 else '✗'}")
    
    return memberships

def visualize_3d_membership_bands(comp, x_range, y_range, t_range):
    """
    Visualize 3D membership as bands/boxes in 3D space (no thresholding)
    Shows windows 1, 3, 5, 7, 9 (main windows) as highlighted 3D regions
    """
    print("Visualizing 3D membership bands (full membership values)...")
    
    # Compute memberships
    t_shifted = t_range - t_range[0]
    with torch.no_grad():
        memberships = comp.compute_window_memberships(x_range, y_range, t_shifted, normalize=False)
    
    # Convert to numpy
    memberships_np = memberships.cpu().numpy()
    x_range_np = np.asarray(x_range)
    y_range_np = np.asarray(y_range)
    t_range_np = np.asarray(t_range)
    
    # Select windows to visualize (1, 3, 5, 7, 9 - main windows)
    windows_to_show = [1, 3, 5, 7, 9]
    
    print(f"Showing windows: {windows_to_show}")
    print(f"These correspond to main windows in the 10-window structure")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create 3D subplots for each window
    for i, window_idx in enumerate(windows_to_show):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        
        # Get membership for this window
        membership = memberships_np[window_idx]  # Shape: [x, y, t]
        
        # Sample points for visualization (no thresholding - use all points)
        step_x = max(1, len(x_range_np) // 8)  # Sample every 8th point
        step_y = max(1, len(y_range_np) // 8)
        step_t = max(1, len(t_range_np) // 8)
        
        # Create coordinate grids for sampling
        x_coords, y_coords, t_coords = np.meshgrid(
            range(0, len(x_range_np), step_x),
            range(0, len(y_range_np), step_y), 
            range(0, len(t_range_np), step_t),
            indexing='ij'
        )
        
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        t_coords = t_coords.flatten()
        
        # Map indices back to actual coordinates
        x_vals = x_range_np[x_coords]
        y_vals = y_range_np[y_coords]  
        t_vals = t_range_np[t_coords]
        
        # Get membership values at these points
        membership_vals = membership[x_coords, y_coords, t_coords]
        
        # Create 3D scatter plot with color coding by membership
        scatter = ax.scatter(x_vals, y_vals, t_vals/1000, 
                           c=membership_vals, 
                           cmap='viridis', 
                           alpha=0.6,
                           s=15)
        
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Membership')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('T (ms)')
        ax.set_title(f'Window {window_idx} Membership\n(Main Window {window_idx-1})')
        
        # Set consistent view
        ax.set_xlim(x_range_np.min(), x_range_np.max())
        ax.set_ylim(y_range_np.min(), y_range_np.max())
        ax.set_zlim(t_range_np.min()/1000, t_range_np.max()/1000)
        
        # Print window statistics
        avg_membership = membership.mean()
        max_membership = membership.max()
        std_membership = membership.std()
        
        print(f"Window {window_idx}: avg={avg_membership:.3f}, max={max_membership:.3f}, std={std_membership:.3f}")
    
    # Add summary plot showing all windows together
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Use coarser sampling for overview
    step_overview = max(1, min(len(x_range_np), len(y_range_np), len(t_range_np)) // 6)
    
    for i, window_idx in enumerate(windows_to_show):
        membership = memberships_np[window_idx]
        
        # Sample points
        x_coords = np.arange(0, len(x_range_np), step_overview)
        y_coords = np.arange(0, len(y_range_np), step_overview)
        t_coords = np.arange(0, len(t_range_np), step_overview)
        
        X_sample, Y_sample, T_sample = np.meshgrid(x_coords, y_coords, t_coords, indexing='ij')
        X_sample = X_sample.flatten()
        Y_sample = Y_sample.flatten()
        T_sample = T_sample.flatten()
        
        x_vals = x_range_np[X_sample]
        y_vals = y_range_np[Y_sample]
        t_vals = t_range_np[T_sample]
        
        # Color by membership, but use fixed color per window for distinction
        membership_vals = membership[X_sample, Y_sample, T_sample]
        
        ax6.scatter(x_vals, y_vals, t_vals/1000, 
                   c=colors[i], 
                   alpha=membership_vals * 0.8,  # Use membership for transparency
                   s=8,
                   label=f'W{window_idx}')
    
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    ax6.set_zlabel('T (ms)')
    ax6.set_title('All Windows Together\n(transparency by membership)')
    ax6.legend()
    
    # Set consistent view
    ax6.set_xlim(x_range_np.min(), x_range_np.max())
    ax6.set_ylim(y_range_np.min(), y_range_np.max())
    ax6.set_zlim(t_range_np.min()/1000, t_range_np.max()/1000)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n3D Membership Bands Visualization:")
    print(f"✓ Each plot shows full membership values (no thresholding)")
    print(f"✓ Points are colored by membership strength")
    print(f"✓ You should see 3D 'bands' or 'boxes' for each window")
    print(f"✓ Window positions depend on (x,y) location due to boundary surfaces")
    print(f"✓ PyTorch tensors used internally with gradient tracking")
    
    return memberships

def visualize_3d_compensation_clean(comp, x_range, y_range, t_range):
    """
    Clean 3D compensation visualization focusing on key aspects (PyTorch version)
    """
    print("Computing 3D compensation with PyTorch...")
    
    with torch.no_grad():  # No gradients for visualization
        compensation = comp(x_range, y_range, t_range, normalize=False)
        
        # Separate X and Y components for analysis
        t_shifted = t_range - t_range[0]
        memberships = comp.compute_window_memberships(x_range, y_range, t_shifted, normalize=False)
        slopes_a, slopes_b = comp.compute_within_window_interpolation(x_range, y_range, t_shifted, normalize=False)
        
        # Convert spatial arrays to tensors
        x_tensor = torch.tensor(x_range, dtype=torch.float32, device=comp.device)
        y_tensor = torch.tensor(y_range, dtype=torch.float32, device=comp.device)
        t_tensor = torch.tensor(t_shifted, dtype=torch.float32, device=comp.device)
        
        X, Y, T = torch.meshgrid(x_tensor, y_tensor, t_tensor, indexing='ij')
        compensation_x = torch.sum(memberships * slopes_a * X[None, :, :, :], dim=0)
        compensation_y = torch.sum(memberships * slopes_b * Y[None, :, :, :], dim=0)
    
    # Convert to numpy for plotting
    compensation_np = compensation.cpu().numpy()
    compensation_x_np = compensation_x.cpu().numpy()
    compensation_y_np = compensation_y.cpu().numpy()
    x_range_np = np.asarray(x_range)
    y_range_np = np.asarray(y_range)
    t_range_np = np.asarray(t_range)
    
    print(f"Compensation shape: {compensation_np.shape}")
    print(f"Total compensation range: {compensation_np.min():.3f} to {compensation_np.max():.3f} μs")
    print(f"X-component range: {compensation_x_np.min():.3f} to {compensation_x_np.max():.3f} μs") 
    print(f"Y-component range: {compensation_y_np.min():.3f} to {compensation_y_np.max():.3f} μs")
    print(f"Y-component is ~{abs(compensation_y_np.std()/compensation_x_np.std()):.1f}x larger than X-component")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Select middle slices
    x_mid = len(x_range_np) // 2
    y_mid = len(y_range_np) // 2
    t_mid = len(t_range_np) // 2
    
    # 1. 3D Total Compensation at fixed time
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    step = max(1, min(len(x_range_np), len(y_range_np)) // 20)
    X_3d, Y_3d = np.meshgrid(x_range_np[::step], y_range_np[::step], indexing='ij')
    Z_3d = compensation_np[::step, ::step, t_mid]
    
    surf = ax1.plot_surface(X_3d, Y_3d, Z_3d, alpha=0.8, cmap='coolwarm')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_zlabel('Compensation (μs)')
    ax1.set_title(f'3D Total Compensation\nat T={t_range_np[t_mid]/1000:.1f}ms')
    ax1.set_box_aspect([1,1,0.3])
    
    # 2. 3D X-component (should be small since ax~0)
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    Z_x = compensation_x_np[::step, ::step, t_mid]
    
    surf_x = ax2.plot_surface(X_3d, Y_3d, Z_x, alpha=0.8, cmap='coolwarm')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_zlabel('X-Compensation (μs)')
    ax2.set_title(f'3D X-Component\n(small since ax≈0)')
    ax2.set_box_aspect([1,1,0.3])
    
    # 3. 3D Y-component (should be dominant since ay~-60 to -80)
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    Z_y = compensation_y_np[::step, ::step, t_mid]
    
    surf_y = ax3.plot_surface(X_3d, Y_3d, Z_y, alpha=0.8, cmap='coolwarm')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_zlabel('Y-Compensation (μs)')
    ax3.set_title(f'3D Y-Component\n(dominant since ay≈-70)')
    ax3.set_box_aspect([1,1,0.3])
    
    # 4. Compensation vs X (at center Y, center T) - should be small variation
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x_range_np, compensation_np[:, y_mid, t_mid], 'b-', linewidth=2, label='Total')
    ax4.plot(x_range_np, compensation_x_np[:, y_mid, t_mid], 'r--', linewidth=2, label='X-component')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Compensation (μs)')
    ax4.set_title(f'Compensation vs X\n(Y={y_range_np[y_mid]:.0f}, T={t_range_np[t_mid]/1000:.1f}ms)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Compensation vs Y (at center X, center T) - should be large variation  
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(y_range_np, compensation_np[x_mid, :, t_mid], 'b-', linewidth=2, label='Total')
    ax5.plot(y_range_np, compensation_y_np[x_mid, :, t_mid], 'g--', linewidth=2, label='Y-component')
    ax5.set_xlabel('Y (pixels)')
    ax5.set_ylabel('Compensation (μs)')
    ax5.set_title(f'Compensation vs Y\n(X={x_range_np[x_mid]:.0f}, T={t_range_np[t_mid]/1000:.1f}ms)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Compensation vs Time (at center X, Y)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t_range_np/1000, compensation_np[x_mid, y_mid, :], 'b-', linewidth=2, label='Total')
    ax6.plot(t_range_np/1000, compensation_x_np[x_mid, y_mid, :], 'r--', linewidth=1, label='X-component')
    ax6.plot(t_range_np/1000, compensation_y_np[x_mid, y_mid, :], 'g--', linewidth=1, label='Y-component')
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Compensation (μs)')
    ax6.set_title(f'Compensation vs Time\n(X={x_range_np[x_mid]:.0f}, Y={y_range_np[y_mid]:.0f})')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Add boundary markers
    with torch.no_grad():
        boundary_at_center = comp.get_boundary_surfaces([x_range_np[x_mid]], [y_range_np[y_mid]], normalize=False)
        boundary_at_center_np = boundary_at_center.cpu().numpy()
    
    for i, boundary_t in enumerate(boundary_at_center_np[:, 0, 0]):
        if t_range_np.min() <= boundary_t <= t_range_np.max():
            ax6.axvline(boundary_t/1000, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return compensation, compensation_x, compensation_y

def test_realistic_scenario():
    """
    Test with the realistic scenario - PyTorch version with gradients
    """
    print("Testing Realistic Scenario (PyTorch with gradients):")
    print("=" * 50)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create realistic parameters
    a_params, b_params, duration = create_realistic_example()
    
    # Create compensator with PyTorch
    comp = Compensate(a_params, b_params, duration, temperature=5000.0, device=device)
    
    # Create realistic data ranges
    width, height = 1280, 720
    
    # Sample coordinates (reduced for performance)
    x_range = np.linspace(0, width, 40)
    y_range = np.linspace(0, height, 30) 
    
    # Simple time range: 0 to 884ms 
    t_range = np.linspace(0, duration, 50)  # 0 to 884000 μs
    
    print(f"\nTest data:")
    print(f"  Sensor: {width}×{height} pixels")
    print(f"  Time: {t_range.min()/1000:.0f} to {t_range.max()/1000:.0f} ms")
    print(f"  Duration: {duration/1000:.0f} ms")
    print(f"  Grid: {len(x_range)}×{len(y_range)}×{len(t_range)}")
    print(f"  A parameters (ax): {comp.a_params.data}")
    print(f"  B parameters (ay): {comp.b_params.data}")
    print(f"  Gradients enabled: {comp.a_params.requires_grad and comp.b_params.requires_grad}")
    
    print("\n" + "="*50)
    print("Step 1: Verify 3D Membership Functions")
    print("="*50)
    
    # First check memberships
    memberships = visualize_3d_membership(comp, x_range, y_range, t_range)
    
    print("\n" + "="*50)
    print("Step 1b: 3D Membership Bands (Windows 1,3,5,7,9)")
    print("="*50)
    
    # Show 3D membership bands
    visualize_3d_membership_bands(comp, x_range, y_range, t_range)
    
    print("\n" + "="*50)
    print("Step 2: Verify 3D Compensation")
    print("="*50)
    
    # Then check compensation
    compensation, comp_x, comp_y = visualize_3d_compensation_clean(comp, x_range, y_range, t_range)
    
    print("\nFinal Results Summary:")
    print(f"✓ PyTorch implementation with gradient tracking")
    print(f"✓ Device: {device}")
    print(f"✓ Membership functions verified: sum ≈ 1.0")
    print(f"✓ X-compensation (ax≈0): {comp_x.min().item():.1f} to {comp_x.max().item():.1f} μs")
    print(f"✓ Y-compensation (ay≈-70): {comp_y.min().item():.1f} to {comp_y.max().item():.1f} μs") 
    print(f"✓ Total compensation: {compensation.min().item():.1f} to {compensation.max().item():.1f} μs")
    print(f"✓ Y-component dominates X-component (expected due to ay >> ax)")
    print(f"✓ Ready for gradient-based optimization!")
    
    # Demonstrate gradient computation
    print(f"\n" + "="*50)
    print("Gradient Test")
    print("="*50)
    
    # Test gradient computation
    print("Testing gradient computation...")
    test_x = torch.tensor([640.0], device=device)
    test_y = torch.tensor([360.0], device=device) 
    test_t = torch.tensor([442000.0], device=device)  # Middle time
    
    # Forward pass with gradients
    test_compensation = comp(test_x, test_y, test_t)
    test_loss = test_compensation.sum()  # Simple loss
    
    # Backward pass
    test_loss.backward()
    
    print(f"✓ Forward pass: compensation = {test_compensation.item():.3f} μs")
    print(f"✓ Backward pass successful")
    print(f"✓ A parameters gradients: {comp.a_params.grad}")
    print(f"✓ B parameters gradients: {comp.b_params.grad}")
    print(f"✓ Ready for optimization with torch.optim!")
    
    return comp, compensation, comp_x, comp_y

if __name__ == "__main__":
    print("3D Multi-Window Compensation - PyTorch with Gradients")
    print("=" * 60)
    
    # Test with realistic scenario with PyTorch
    comp, compensation, comp_x, comp_y = test_realistic_scenario()
    
    print("\n" + "=" * 60)
    print("Summary of Key Findings:")
    print("=" * 60)
    print(f"1. PyTorch Implementation:")
    print(f"   - All operations use PyTorch tensors")
    print(f"   - Gradient tracking enabled for optimization")
    print(f"   - CUDA support if available")
    print(f"")
    print(f"2. 3D Membership Visualization:")
    print(f"   - No thresholding - shows full membership values")
    print(f"   - 3D bands/boxes clearly visible")
    print(f"   - Smooth transitions between windows")
    print(f"")
    print(f"3. Compensation Results:")
    print(f"   - X-component: {comp_x.min().item():.1f} to {comp_x.max().item():.1f} μs")
    print(f"   - Y-component: {comp_y.min().item():.1f} to {comp_y.max().item():.1f} μs")
    print(f"   - Total range: {compensation.min().item():.1f} to {compensation.max().item():.1f} μs")
    print(f"")
    print(f"Usage for Optimization:")
    print(f"  comp = Compensate(a_params, b_params, duration, device='cuda')")
    print(f"  optimizer = torch.optim.Adam(comp.parameters(), lr=0.01)")
    print(f"  loss = your_loss_function(comp(x, y, t))")
    print(f"  loss.backward()")
    print(f"  optimizer.step()")
    print(f"")
    print(f"✓ PyTorch implementation with gradients ready")
    print(f"✓ 3D membership bands visualization (no thresholding)")
    print(f"✓ Compatible with any PyTorch optimizer")
    print(f"✓ CUDA support for GPU acceleration")