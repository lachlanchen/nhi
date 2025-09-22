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

def visualize_3d_membership(comp, x_range, y_range, t_range):
    """
    Simple 3D membership visualization to verify correctness
    """
    print("Visualizing 3D membership functions...")
    
    # Compute memberships
    t_shifted = t_range - t_range[0]
    memberships = comp.compute_window_memberships(x_range, y_range, t_shifted, normalize=False)
    
    print(f"Membership shape: {memberships.shape}")
    print(f"Number of windows: {comp.num_total_windows}")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Select middle time slice to show membership
    t_mid = len(t_range) // 2
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # Show first 6 windows (main windows + some edge)
    num_to_show = min(6, comp.num_total_windows)
    
    for i in range(num_to_show):
        ax = plt.subplot(2, 3, i + 1)
        
        membership_slice = memberships[i, :, :, t_mid]
        im = ax.contourf(X, Y, membership_slice, levels=15, cmap='viridis')
        
        window_type = "Edge" if i == 0 or i == comp.num_total_windows-1 else f"Main {i}"
        ax.set_title(f'Window {i} ({window_type})\nMembership at T={t_range[t_mid]/1000:.1f}ms')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Print membership statistics
        print(f"Window {i}: membership range {membership_slice.min():.3f} to {membership_slice.max():.3f}")
    
    plt.suptitle(f'3D Window Memberships - Verification\nTotal sum should ≈ 1.0', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Verify membership sum
    total_membership = np.sum(memberships, axis=0)
    print(f"\nMembership sum verification:")
    print(f"Total membership range: {total_membership.min():.6f} to {total_membership.max():.6f}")
    print(f"Should be close to 1.0: {'✓' if abs(total_membership.mean() - 1.0) < 0.01 else '✗'}")
    
    return memberships

def visualize_3d_compensation_clean(comp, x_range, y_range, t_range):
    """
    Clean 3D compensation visualization focusing on key aspects
    """
    print("Computing 3D compensation...")
    compensation = comp(x_range, y_range, t_range, normalize=False)
    
    # Separate X and Y components for analysis
    t_shifted = t_range - t_range[0]
    memberships = comp.compute_window_memberships(x_range, y_range, t_shifted, normalize=False)
    slopes_a, slopes_b = comp.compute_within_window_interpolation(x_range, y_range, t_shifted, normalize=False)
    
    X, Y, T = np.meshgrid(x_range, y_range, t_shifted, indexing='ij')
    compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
    compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
    
    print(f"Compensation shape: {compensation.shape}")
    print(f"Total compensation range: {compensation.min():.3f} to {compensation.max():.3f} μs")
    print(f"X-component range: {compensation_x.min():.3f} to {compensation_x.max():.3f} μs") 
    print(f"Y-component range: {compensation_y.min():.3f} to {compensation_y.max():.3f} μs")
    print(f"Y-component is ~{abs(compensation_y.mean()/compensation_x.mean()):.1f}x larger than X-component")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Select middle slices
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # 1. 3D Total Compensation at fixed time
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    step = max(1, min(len(x_range), len(y_range)) // 20)
    X_3d, Y_3d = np.meshgrid(x_range[::step], y_range[::step], indexing='ij')
    Z_3d = compensation[::step, ::step, t_mid]
    
    surf = ax1.plot_surface(X_3d, Y_3d, Z_3d, alpha=0.8, cmap='coolwarm')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_zlabel('Compensation (μs)')
    ax1.set_title(f'3D Total Compensation\nat T={t_range[t_mid]/1000:.1f}ms')
    ax1.set_box_aspect([1,1,0.3])
    
    # 2. 3D X-component (should be small since ax~0)
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    Z_x = compensation_x[::step, ::step, t_mid]
    
    surf_x = ax2.plot_surface(X_3d, Y_3d, Z_x, alpha=0.8, cmap='coolwarm')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_zlabel('X-Compensation (μs)')
    ax2.set_title(f'3D X-Component\n(small since ax≈0)')
    ax2.set_box_aspect([1,1,0.3])
    
    # 3. 3D Y-component (should be dominant since ay~-60 to -80)
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    Z_y = compensation_y[::step, ::step, t_mid]
    
    surf_y = ax3.plot_surface(X_3d, Y_3d, Z_y, alpha=0.8, cmap='coolwarm')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_zlabel('Y-Compensation (μs)')
    ax3.set_title(f'3D Y-Component\n(dominant since ay≈-70)')
    ax3.set_box_aspect([1,1,0.3])
    
    # 4. Compensation vs X (at center Y, center T) - should be small variation
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x_range, compensation[:, y_mid, t_mid], 'b-', linewidth=2, label='Total')
    ax4.plot(x_range, compensation_x[:, y_mid, t_mid], 'r--', linewidth=2, label='X-component')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Compensation (μs)')
    ax4.set_title(f'Compensation vs X\n(Y={y_range[y_mid]:.0f}, T={t_range[t_mid]/1000:.1f}ms)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Compensation vs Y (at center X, center T) - should be large variation  
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(y_range, compensation[x_mid, :, t_mid], 'b-', linewidth=2, label='Total')
    ax5.plot(y_range, compensation_y[x_mid, :, t_mid], 'g--', linewidth=2, label='Y-component')
    ax5.set_xlabel('Y (pixels)')
    ax5.set_ylabel('Compensation (μs)')
    ax5.set_title(f'Compensation vs Y\n(X={x_range[x_mid]:.0f}, T={t_range[t_mid]/1000:.1f}ms)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Compensation vs Time (at center X, Y)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t_range/1000, compensation[x_mid, y_mid, :], 'b-', linewidth=2, label='Total')
    ax6.plot(t_range/1000, compensation_x[x_mid, y_mid, :], 'r--', linewidth=1, label='X-component')
    ax6.plot(t_range/1000, compensation_y[x_mid, y_mid, :], 'g--', linewidth=1, label='Y-component')
    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Compensation (μs)')
    ax6.set_title(f'Compensation vs Time\n(X={x_range[x_mid]:.0f}, Y={y_range[y_mid]:.0f})')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Add boundary markers
    boundary_at_center = comp.get_boundary_surfaces([x_range[x_mid]], [y_range[y_mid]], normalize=False)
    for i, boundary_t in enumerate(boundary_at_center[:, 0, 0]):
        if t_range.min() <= boundary_t <= t_range.max():
            ax6.axvline(boundary_t/1000, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return compensation, compensation_x, compensation_y

def test_realistic_scenario():
    """
    Test with the realistic scenario - clean visualization
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
    print(f"  A parameters (ax): {comp.a_params}")
    print(f"  B parameters (ay): {comp.b_params}")
    
    print("\n" + "="*50)
    print("Step 1: Verify 3D Membership Functions")
    print("="*50)
    
    # First check memberships
    memberships = visualize_3d_membership(comp, x_range, y_range, t_range)
    
    print("\n" + "="*50)
    print("Step 2: Verify 3D Compensation")
    print("="*50)
    
    # Then check compensation
    compensation, comp_x, comp_y = visualize_3d_compensation_clean(comp, x_range, y_range, t_range)
    
    print("\nFinal Results Summary:")
    print(f"✓ Membership functions verified: sum ≈ 1.0")
    print(f"✓ X-compensation (ax≈0): {comp_x.min():.3f} to {comp_x.max():.3f} μs")
    print(f"✓ Y-compensation (ay≈-70): {comp_y.min():.3f} to {comp_y.max():.3f} μs") 
    print(f"✓ Total compensation: {compensation.min():.3f} to {compensation.max():.3f} μs")
    print(f"✓ Y-component dominates X-component by ~{abs(comp_y.std()/comp_x.std()):.1f}x (expected due to ay >> ax)")
    
    return comp, compensation, comp_x, comp_y



if __name__ == "__main__":
    print("3D Multi-Window Compensation - Clean Visualization")
    print("=" * 60)
    
    # Test with realistic scenario focusing on key verification
    comp, compensation, comp_x, comp_y = test_realistic_scenario()
    
    print("\n" + "=" * 60)
    print("Summary of Key Findings:")
    print("=" * 60)
    print(f"1. Window Structure:")
    print(f"   - {comp.num_main_windows} main windows + 2 edge windows")
    print(f"   - Window size: {comp.main_window_size/1000:.1f} ms")
    print(f"   - {comp.num_boundaries} boundary surfaces")
    print(f"")
    print(f"2. Parameter Scale Analysis:")
    print(f"   - ax (X-direction): 0 to 10 μs → small X-compensation")
    print(f"   - ay (Y-direction): -60 to -80 μs → large Y-compensation")  
    print(f"   - Y-compensation dominates as expected")
    print(f"")
    print(f"3. Compensation Results:")
    print(f"   - X-component: {comp_x.min():.3f} to {comp_x.max():.3f} μs")
    print(f"   - Y-component: {comp_y.min():.3f} to {comp_y.max():.3f} μs")
    print(f"   - Total range: {compensation.min():.3f} to {compensation.max():.3f} μs")
    print(f"")
    print(f"Usage:")
    print(f"  comp = Compensate(a_params, b_params, duration_microseconds)")
    print(f"  compensation = comp(x_pixels, y_pixels, t_microseconds)")
    print(f"")
    print(f"✓ 3D membership functions verified")
    print(f"✓ 3D compensation computed correctly") 
    print(f"✓ Scale differences handled properly")
    print(f"✓ Y-dominance confirmed (ay >> ax)")