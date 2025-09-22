import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Compensate:
    def __init__(self, a_params, b_params, duration, temperature=1000):
        """
        3D Multi-window compensation class
        
        Args:
            a_params: X-direction coefficients (len determines number of boundaries)
            b_params: Y-direction coefficients (same length as a_params)  
            duration: Time duration in microseconds (max_t - min_t) after shifting
            temperature: Smoothness parameter in microseconds (default 1000)
        """
        self.a_params = np.array(a_params, dtype=float)
        self.b_params = np.array(b_params, dtype=float)  
        self.duration = float(duration)
        self.temperature = float(temperature)
        
        # Validate input
        if len(self.a_params) != len(self.b_params):
            raise ValueError("a_params and b_params must have the same length")
        
        # Calculate window structure
        self.num_boundaries = len(self.a_params)
        self.num_main_windows = self.num_boundaries - 3  # n_main = len(ax) - 3
        self.num_total_windows = self.num_main_windows + 2  # Add 2 edge windows
        
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")
        
        # Calculate main window size based on main windows only (in microseconds)
        self.main_window_size = self.duration / self.num_main_windows
        
        # Calculate boundary offsets in microseconds
        # Structure: [edge_start, main_0, main_1, ..., main_n, edge_end]
        self.boundary_offsets = np.array([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ])
        
        print(f"Compensate initialized:")
        print(f"  Duration: {self.duration:.0f} μs ({self.duration/1000:.1f} ms)")
        print(f"  Parameters per direction: {self.num_boundaries}")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.num_total_windows} (including 2 edge windows)")
        print(f"  Main window size: {self.main_window_size:.0f} μs ({self.main_window_size/1000:.1f} ms)")
        print(f"  Boundary offsets: {self.boundary_offsets/1000}")
        print(f"  A parameters: {self.a_params}")
        print(f"  B parameters: {self.b_params}")
        print(f"  Temperature: {self.temperature:.0f} μs")
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_boundary_surfaces(self, x, y, normalize=False, width=None, height=None):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        
        Args:
            x: X coordinates (pixels or normalized)
            y: Y coordinates (pixels or normalized)
            normalize: Whether to normalize x,y by width,height (default False)
            width: Sensor width for normalization (required if normalize=True)
            height: Sensor height for normalization (required if normalize=True)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if normalize:
            if width is None or height is None:
                raise ValueError("width and height required when normalize=True")
            x = x / width
            y = y / height
        
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y, indexing='ij')
            boundary_values = np.zeros((self.num_boundaries, len(x), len(y)))
            for i in range(self.num_boundaries):
                boundary_values[i] = (self.a_params[i] * X + 
                                    self.b_params[i] * Y + 
                                    self.boundary_offsets[i])
        else:
            raise ValueError("x and y must be 1D arrays")
        
        return boundary_values
    
    def compute_window_memberships(self, x, y, t_shifted, normalize=False, width=None, height=None):
        """
        Compute soft membership for each window
        """
        boundary_values = self.get_boundary_surfaces(x, y, normalize, width, height)
        
        # Create 3D meshgrids
        if normalize and width is not None and height is not None:
            x_use = np.asarray(x) / width
            y_use = np.asarray(y) / height
        else:
            x_use = np.asarray(x)
            y_use = np.asarray(y)
            
        X, Y, T = np.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_total_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [num_total_windows, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)
        memberships_sum = np.maximum(memberships_sum, 1e-8)
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t_shifted, normalize=False, width=None, height=None):
        """
        Interpolate parameters within each window
        """
        boundary_values = self.get_boundary_surfaces(x, y, normalize, width, height)
        
        # Create meshgrids
        if normalize and width is not None and height is not None:
            x_use = np.asarray(x) / width
            y_use = np.asarray(y) / height
        else:
            x_use = np.asarray(x)
            y_use = np.asarray(y)
            
        X, Y, T = np.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        interpolated_slopes_a = []
        interpolated_slopes_b = []
        
        for i in range(self.num_total_windows):
            lower_bound = boundary_values[i][:, :, np.newaxis]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]
            
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)
            
            # Interpolation parameter (no clamping)
            alpha = (T - lower_bound) / window_width
            
            # Interpolate between parameters
            slope_a = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            slope_b = (1 - alpha) * self.b_params[i] + alpha * self.b_params[i + 1]
            
            interpolated_slopes_a.append(slope_a)
            interpolated_slopes_b.append(slope_b)
        
        return np.array(interpolated_slopes_a), np.array(interpolated_slopes_b)
    
    def __call__(self, x, y, t, width=None, height=None, normalize=False):
        """
        Compute compensation for given (x, y, t) coordinates
        
        Args:
            x: X coordinates (pixels)
            y: Y coordinates (pixels)
            t: T coordinates (microseconds)
            width: Sensor width (required if normalize=True)
            height: Sensor height (required if normalize=True)
            normalize: Whether to normalize spatial coordinates (default False)
            
        Returns:
            compensation: Total compensation in microseconds
        """
        # Shift time to start from 0
        t_array = np.asarray(t, dtype=float)
        t_min = t_array.min()
        t_shifted = t_array - t_min
        
        # Verify duration matches approximately
        actual_duration = t_shifted.max()
        if abs(actual_duration - self.duration) > self.duration * 0.01:  # 1% tolerance
            print(f"Warning: Actual duration {actual_duration:.0f} μs != expected {self.duration:.0f} μs")
        
        # Get window memberships
        memberships = self.compute_window_memberships(x, y, t_shifted, normalize, width, height)
        
        # Get interpolated slopes
        slopes_a, slopes_b = self.compute_within_window_interpolation(x, y, t_shifted, normalize, width, height)
        
        # Create meshgrids
        if normalize and width is not None and height is not None:
            x_use = np.asarray(x) / width
            y_use = np.asarray(y) / height
        else:
            x_use = np.asarray(x)
            y_use = np.asarray(y)
        
        X, Y, T = np.meshgrid(x_use, y_use, t_shifted, indexing='ij')
        
        # Weighted sum of compensations
        compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
        compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
        compensation = compensation_x + compensation_y
        
        return compensation

def create_realistic_example():
    """
    Create realistic example: 884ms duration, 10 main windows, 1280x720 sensor
    ax: 0 to 10, ay: -60 to -80 (in μs units)
    """
    duration = 884000.0  # 884ms in μs
    n_main_windows = 10
    n_boundaries = n_main_windows + 3  # 13 boundaries
    
    # Create ax parameters: 0 to 10 μs
    a_params = np.linspace(0, 10, n_boundaries).tolist()
    
    # Create ay parameters: -60 to -80 μs
    b_params = np.linspace(-60, -80, n_boundaries).tolist()
    
    print(f"Realistic example parameters:")
    print(f"  Duration: {duration:.0f} μs ({duration/1000:.1f} ms)")
    print(f"  Main windows: {n_main_windows}")
    print(f"  Boundaries needed: {n_boundaries}")
    print(f"  Window size: {duration/n_main_windows:.0f} μs ({duration/n_main_windows/1000:.1f} ms)")
    print(f"  A parameters (μs): {a_params}")
    print(f"  B parameters (μs): {b_params}")
    
    return a_params, b_params, duration

def visualize_3d_boundary_surfaces(comp, x_range, y_range, width, height):
    """
    Visualize the 3D boundary surfaces with proper scaling
    """
    print("Visualizing 3D boundary surfaces...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Get boundary surfaces (without normalization for better scaling)
    boundary_surfaces = comp.get_boundary_surfaces(x_range, y_range, normalize=False)
    
    # Create meshgrids for surfaces (subsample for performance)
    step_x = max(1, len(x_range) // 20)
    step_y = max(1, len(y_range) // 15)
    
    X_surf, Y_surf = np.meshgrid(x_range[::step_x], y_range[::step_y], indexing='ij')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_surfaces)))
    
    # 3D surface plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i in range(0, len(boundary_surfaces), 2):  # Show every 2nd surface
        Z_surf = boundary_surfaces[i][::step_x, ::step_y]
        ax1.plot_surface(X_surf, Y_surf, Z_surf/1000, alpha=0.4, color=colors[i])
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)') 
    ax1.set_zlabel('T (ms)')
    ax1.set_title(f'3D Boundary Surfaces\n({comp.num_boundaries} surfaces)')
    
    # Set better aspect ratio
    ax1.set_box_aspect([1,1,0.5])
    
    # Boundary lines at different Y positions
    y_positions = [y_range[len(y_range)//4], y_range[len(y_range)//2], y_range[3*len(y_range)//4]]
    
    for idx, y_pos in enumerate(y_positions):
        ax = fig.add_subplot(2, 3, 2 + idx)
        
        y_idx = np.argmin(np.abs(y_range - y_pos))
        
        for i, surface in enumerate(boundary_surfaces):
            ax.plot(x_range, surface[:, y_idx]/1000, '--', linewidth=2, 
                   color=colors[i % len(colors)], 
                   label=f'Boundary {i}' if idx == 0 and i < 5 else '')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('T (ms)')
        ax.set_title(f'Boundary Lines at Y = {y_pos:.0f}')
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Boundary lines at different X positions  
    x_positions = [x_range[len(x_range)//4], x_range[len(x_range)//2]]
    
    for idx, x_pos in enumerate(x_positions):
        ax = fig.add_subplot(2, 3, 5 + idx)
        
        x_idx = np.argmin(np.abs(x_range - x_pos))
        
        for i, surface in enumerate(boundary_surfaces):
            ax.plot(y_range, surface[x_idx, :]/1000, '--', linewidth=2, 
                   color=colors[i % len(colors)],
                   label=f'Boundary {i}' if idx == 0 and i < 5 else '')
        
        ax.set_xlabel('Y (pixels)')
        ax.set_ylabel('T (ms)')
        ax.set_title(f'Boundary Lines at X = {x_pos:.0f}')
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_compensation_3d(comp, x_range, y_range, t_range, width, height):
    """
    Comprehensive 3D visualization of compensation with proper scaling
    """
    print("Computing 3D compensation...")
    compensation = comp(x_range, y_range, t_range, width, height, normalize=False)
    
    print(f"Compensation shape: {compensation.shape}")
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f} μs")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Select middle slices
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # 1. Compensation at fixed T (XY slice)
    ax1 = plt.subplot(3, 4, 1)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    im1 = ax1.contourf(X, Y, compensation[:, :, t_mid], levels=20, cmap='coolwarm')
    ax1.set_title(f'Compensation at T = {t_range[t_mid]/1000:.1f} ms')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='μs')
    
    # 2. Compensation at fixed Y (XT slice)  
    ax2 = plt.subplot(3, 4, 2)
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    im2 = ax2.contourf(X_2d, T_2d/1000, compensation[:, y_mid, :], levels=20, cmap='coolwarm')
    ax2.set_title(f'Compensation at Y = {y_range[y_mid]:.0f}')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('T (ms)')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='μs')
    
    # 3. Compensation at fixed X (YT slice)
    ax3 = plt.subplot(3, 4, 3)
    Y_2d, T_2d = np.meshgrid(y_range, t_range, indexing='ij')
    im3 = ax3.contourf(Y_2d, T_2d/1000, compensation[x_mid, :, :], levels=20, cmap='coolwarm')
    ax3.set_title(f'Compensation at X = {x_range[x_mid]:.0f}')
    ax3.set_xlabel('Y (pixels)')
    ax3.set_ylabel('T (ms)')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='μs')
    
    # 4. 3D compensation surface
    ax4 = plt.subplot(3, 4, 4, projection='3d')
    step = max(1, min(len(x_range), len(y_range)) // 15)
    X_surf = X[::step, ::step]
    Y_surf = Y[::step, ::step]
    Z_surf = compensation[::step, ::step, t_mid]
    
    surf = ax4.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.8, cmap='coolwarm')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    ax4.set_zlabel('Compensation (μs)')
    ax4.set_title(f'3D Compensation at T={t_range[t_mid]/1000:.1f}ms')
    ax4.set_box_aspect([1,1,0.3])
    
    # 5-8. Window memberships at fixed T (show first 4 windows)
    memberships = comp.compute_window_memberships(x_range, y_range, 
                                                 t_range - t_range[0], normalize=False)
    
    for i in range(min(4, comp.num_total_windows)):
        ax = plt.subplot(3, 4, 5 + i)
        im = ax.contourf(X, Y, memberships[i, :, :, t_mid], levels=15, cmap='viridis')
        ax.set_title(f'Window {i} Membership\nat T = {t_range[t_mid]/1000:.1f} ms')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 9. Compensation along time at center point
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(t_range/1000, compensation[x_mid, y_mid, :], 'b-', linewidth=2)
    ax9.set_title(f'Compensation vs Time\nat ({x_range[x_mid]:.0f}, {y_range[y_mid]:.0f})')
    ax9.set_xlabel('T (ms)')
    ax9.set_ylabel('Compensation (μs)')
    ax9.grid(True, alpha=0.3)
    
    # Add boundary markers
    boundary_at_center = comp.get_boundary_surfaces([x_range[x_mid]], [y_range[y_mid]], normalize=False)
    colors_boundary = plt.cm.tab10(np.linspace(0, 1, len(boundary_at_center)))
    
    for i, boundary_t in enumerate(boundary_at_center[:, 0, 0]):
        if t_range.min() <= boundary_t <= t_range.max():
            ax9.axvline(boundary_t/1000, color=colors_boundary[i], linestyle='--', alpha=0.7)
    
    # 10. Compensation along X direction at center
    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(x_range, compensation[:, y_mid, t_mid], 'r-', linewidth=2)
    ax10.set_title(f'Compensation vs X\nat Y={y_range[y_mid]:.0f}, T={t_range[t_mid]/1000:.1f}ms')
    ax10.set_xlabel('X (pixels)')
    ax10.set_ylabel('Compensation (μs)')
    ax10.grid(True, alpha=0.3)
    
    # 11. Compensation along Y direction at center
    ax11 = plt.subplot(3, 4, 11)
    ax11.plot(y_range, compensation[x_mid, :, t_mid], 'g-', linewidth=2)
    ax11.set_title(f'Compensation vs Y\nat X={x_range[x_mid]:.0f}, T={t_range[t_mid]/1000:.1f}ms')
    ax11.set_xlabel('Y (pixels)')
    ax11.set_ylabel('Compensation (μs)')
    ax11.grid(True, alpha=0.3)
    
    # 12. Statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    stats_text = f"""3D Compensation Statistics:

Sensor: {width}×{height} pixels
Time: {t_range.min()/1000:.1f} to {t_range.max()/1000:.1f} ms
Duration: {comp.duration/1000:.1f} ms

Grid: {len(x_range)}×{len(y_range)}×{len(t_range)}

Compensation range: 
{compensation.min():.3f} to {compensation.max():.3f} μs

Model:
Main windows: {comp.num_main_windows}
Total windows: {comp.num_total_windows}  
Boundaries: {comp.num_boundaries}
Window size: {comp.main_window_size/1000:.1f} ms
Temperature: {comp.temperature:.0f} μs

A params: {comp.a_params[:5]}...
B params: {comp.b_params[:5]}..."""
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def test_realistic_scenario():
    """
    Test with the realistic scenario
    """
    print("Testing Realistic Scenario:")
    print("=" * 50)
    
    # Create realistic parameters
    a_params, b_params, duration = create_realistic_example()
    
    # Create compensator
    comp = Compensate(a_params, b_params, duration, temperature=5000.0)
    
    # Create realistic data ranges
    width, height = 1280, 720
    
    # Sample coordinates (reduced for performance)
    x_range = np.linspace(0, width, 40)
    y_range = np.linspace(0, height, 30) 
    
    # Time range with the specified duration
    t_start = 1000000  # 1 second in microseconds
    t_end = t_start + duration  # Add duration in microseconds
    t_range = np.linspace(t_start, t_end, 50)
    
    print(f"\nTest data:")
    print(f"  Sensor: {width}×{height} pixels")
    print(f"  Time: {t_start/1000:.0f} to {t_end/1000:.0f} ms")
    print(f"  Actual duration: {(t_end - t_start)/1000:.0f} ms")
    print(f"  Grid: {len(x_range)}×{len(y_range)}×{len(t_range)}")
    
    # Visualize boundary surfaces first
    visualize_3d_boundary_surfaces(comp, x_range, y_range, width, height)
    
    # Test compensation computation
    compensation = comp(x_range, y_range, t_range, width, height, normalize=False)
    
    print(f"\nCompensation Results:")
    print(f"  Compensation shape: {compensation.shape}")
    print(f"  Compensation range: {compensation.min():.3f} to {compensation.max():.3f} μs")
    
    # Full compensation visualization
    visualize_compensation_3d(comp, x_range, y_range, t_range, width, height)
    
    return comp, compensation

def test_normalization_comparison():
    """
    Compare normalized vs unnormalized compensation
    """
    print("Comparing Normalized vs Unnormalized:")
    print("=" * 50)
    
    a_params, b_params, duration = create_realistic_example()
    comp = Compensate(a_params, b_params, duration)
    
    width, height = 1280, 720
    x_range = np.linspace(0, width, 20)
    y_range = np.linspace(0, height, 15)
    t_range = np.linspace(1000000, 1000000 + duration, 25)
    
    # Test both modes
    comp_unnorm = comp(x_range, y_range, t_range, normalize=False)
    comp_norm = comp(x_range, y_range, t_range, width, height, normalize=True)
    
    print(f"Unnormalized: {comp_unnorm.min():.3f} to {comp_unnorm.max():.3f} μs")
    print(f"Normalized: {comp_norm.min():.6f} to {comp_norm.max():.6f} μs")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    t_mid = len(t_range) // 2
    
    im1 = axes[0].contourf(X, Y, comp_unnorm[:, :, t_mid], levels=15, cmap='coolwarm')
    axes[0].set_title('Unnormalized Compensation')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0], label='μs')
    
    im2 = axes[1].contourf(X, Y, comp_norm[:, :, t_mid], levels=15, cmap='coolwarm')
    axes[1].set_title('Normalized Compensation')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[1], label='μs')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("3D Multi-Window Compensation - Microsecond Units")
    print("=" * 60)
    
    # Test with realistic scenario  
    comp, compensation = test_realistic_scenario()
    
    print("\n" + "=" * 60)
    print("Normalization Comparison")
    
    # Compare normalization modes
    test_normalization_comparison()
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("# Unnormalized (recommended):")
    print("comp = Compensate(a_params, b_params, duration_us)")
    print("compensation = comp(x, y, t)")
    print("")
    print("# Normalized (optional):")
    print("compensation = comp(x, y, t, width, height, normalize=True)")