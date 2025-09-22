import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Compensate:
    def __init__(self, a_params, b_params, duration, temperature=1):
        """
        3D Multi-window compensation class
        
        Args:
            a_params: X-direction coefficients (len determines number of boundaries)
            b_params: Y-direction coefficients (same length as a_params)
            duration: Time duration (max_t - min_t) after shifting
            temperature: Smoothness parameter (default 1)
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
        
        # Calculate main window size based on main windows only
        self.main_window_size = self.duration / self.num_main_windows
        
        # Calculate boundary offsets
        # Structure: [edge_start, main_0, main_1, ..., main_n, edge_end]
        # Boundaries at: [-main_window_size, 0, main_window_size, ..., duration, duration+main_window_size]
        self.boundary_offsets = np.array([
            (i - 1) * self.main_window_size
            for i in range(self.num_boundaries)
        ])
        
        print(f"Compensate initialized:")
        print(f"  Duration: {self.duration} (time units)")
        print(f"  Parameters per direction: {self.num_boundaries}")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.num_total_windows} (including 2 edge windows)")
        print(f"  Main window size: {self.main_window_size:.3f}")
        print(f"  Boundary offsets: {self.boundary_offsets}")
        print(f"  A parameters: {self.a_params}")
        print(f"  B parameters: {self.b_params}")
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_boundary_surfaces(self, x_norm, y_norm):
        """
        Compute boundary surface values: t = a_i * x_norm + b_i * y_norm + offset_i
        """
        x_norm = np.asarray(x_norm)
        y_norm = np.asarray(y_norm)
        
        if x_norm.ndim == 1 and y_norm.ndim == 1:
            X, Y = np.meshgrid(x_norm, y_norm, indexing='ij')
            boundary_values = np.zeros((self.num_boundaries, len(x_norm), len(y_norm)))
            for i in range(self.num_boundaries):
                boundary_values[i] = (self.a_params[i] * X + 
                                    self.b_params[i] * Y + 
                                    self.boundary_offsets[i])
        else:
            raise ValueError("x_norm and y_norm must be 1D arrays")
        
        return boundary_values
    
    def compute_window_memberships(self, x_norm, y_norm, t_shifted):
        """
        Compute soft membership for each window
        """
        boundary_values = self.get_boundary_surfaces(x_norm, y_norm)  # [num_boundaries, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x_norm, y_norm, t_shifted, indexing='ij')
        
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
    
    def compute_within_window_interpolation(self, x_norm, y_norm, t_shifted):
        """
        Interpolate parameters within each window
        """
        boundary_values = self.get_boundary_surfaces(x_norm, y_norm)  # [num_boundaries, len_x, len_y]
        
        X, Y, T = np.meshgrid(x_norm, y_norm, t_shifted, indexing='ij')
        
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
    
    def __call__(self, x, y, t, width, height):
        """
        Compute compensation for given (x, y, t) coordinates
        
        Args:
            x: X coordinates 
            y: Y coordinates 
            t: T coordinates 
            width: Sensor width for normalization
            height: Sensor height for normalization
            
        Returns:
            compensation: Total compensation 
        """
        # Normalize spatial coordinates
        x_norm = np.asarray(x, dtype=float) / width
        y_norm = np.asarray(y, dtype=float) / height
        
        # Shift time to start from 0
        t_array = np.asarray(t, dtype=float)
        t_min = t_array.min()
        t_shifted = t_array - t_min
        
        # Verify duration matches approximately
        actual_duration = t_shifted.max()
        if abs(actual_duration - self.duration) > self.duration * 0.01:  # 1% tolerance
            print(f"Warning: Actual duration {actual_duration:.3f} != expected {self.duration:.3f}")
        
        # Get window memberships
        memberships = self.compute_window_memberships(x_norm, y_norm, t_shifted)  
        
        # Get interpolated slopes
        slopes_a, slopes_b = self.compute_within_window_interpolation(x_norm, y_norm, t_shifted)
        
        # Create meshgrids (using normalized coordinates)
        X, Y, T = np.meshgrid(x_norm, y_norm, t_shifted, indexing='ij')
        
        # Weighted sum of compensations: C = Σᵢ wᵢ × (aᵢ×x_norm + bᵢ×y_norm)
        compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
        compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
        compensation = compensation_x + compensation_y
        
        return compensation

def create_realistic_example():
    """
    Create the realistic example: 884ms duration, 10 main windows, 1280x720 sensor
    ax: 0 to 10, ay: -60 to -80
    """
    duration = 884.0  # ms
    n_main_windows = 10
    n_boundaries = n_main_windows + 3  # 13 boundaries for 10 main + 2 edge windows
    
    # Create ax parameters: 0 to 10
    a_params = np.linspace(0, 10, n_boundaries).tolist()
    
    # Create ay parameters: -60 to -80
    b_params = np.linspace(-60, -80, n_boundaries).tolist()
    
    print(f"Realistic example parameters:")
    print(f"  Duration: {duration} ms")
    print(f"  Main windows: {n_main_windows}")
    print(f"  Boundaries needed: {n_boundaries}")
    print(f"  Window size: {duration/n_main_windows:.1f} ms")
    print(f"  A parameters: {a_params}")
    print(f"  B parameters: {b_params}")
    
    return a_params, b_params, duration

def test_realistic_scenario():
    """
    Test with the realistic scenario
    """
    print("Testing Realistic Scenario:")
    print("=" * 50)
    
    # Create realistic parameters
    a_params, b_params, duration = create_realistic_example()
    
    # Create compensator
    comp = Compensate(a_params, b_params, duration, temperature=10.0)
    
    # Create realistic data ranges
    width, height = 1280, 720
    
    # Sample some coordinates (reduced for faster computation)
    x_range = np.linspace(0, width, 30)
    y_range = np.linspace(0, height, 25) 
    
    # Time range with the specified duration
    t_start = 1000000  # 1 second in microseconds
    t_end = t_start + duration * 1000  # Add duration in microseconds
    t_range = np.linspace(t_start, t_end, 40)
    
    print(f"\nTest data:")
    print(f"  Sensor: {width}×{height}")
    print(f"  Time: {t_start} to {t_end} μs")
    print(f"  Actual duration: {(t_end - t_start)/1000:.1f} ms")
    print(f"  Grid: {len(x_range)}×{len(y_range)}×{len(t_range)}")
    
    # Test computation
    compensation = comp(x_range, y_range, t_range, width, height)
    
    print(f"\nResults:")
    print(f"  Compensation shape: {compensation.shape}")
    print(f"  Compensation range: {compensation.min():.6f} to {compensation.max():.6f}")
    
    # Visualize key slices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Select middle indices
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # XY slice at middle time
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    im1 = axes[0, 0].contourf(X, Y, compensation[:, :, t_mid], levels=15, cmap='coolwarm')
    axes[0, 0].set_title(f'Compensation at T = {t_range[t_mid]:.0f} μs')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # XT slice at middle Y
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    im2 = axes[0, 1].contourf(X_2d, T_2d/1000, compensation[:, y_mid, :], levels=15, cmap='coolwarm')
    axes[0, 1].set_title(f'Compensation at Y = {y_range[y_mid]:.0f}')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('T (ms)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # YT slice at middle X
    Y_2d, T_2d = np.meshgrid(y_range, t_range, indexing='ij')
    im3 = axes[1, 0].contourf(Y_2d, T_2d/1000, compensation[x_mid, :, :], levels=15, cmap='coolwarm')
    axes[1, 0].set_title(f'Compensation at X = {x_range[x_mid]:.0f}')
    axes[1, 0].set_xlabel('Y (pixels)')
    axes[1, 0].set_ylabel('T (ms)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Compensation along time at center point
    axes[1, 1].plot((t_range - t_start)/1000, compensation[x_mid, y_mid, :], 'b-', linewidth=2)
    axes[1, 1].set_title(f'Compensation at Center ({x_range[x_mid]:.0f}, {y_range[y_mid]:.0f})')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Compensation')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add boundary markers
    x_norm_center = x_range[x_mid] / width
    y_norm_center = y_range[y_mid] / height
    boundary_at_center = comp.get_boundary_surfaces([x_norm_center], [y_norm_center])
    
    for i, boundary_t in enumerate(boundary_at_center[:, 0, 0]):
        boundary_t_ms = boundary_t / 1000  # Convert to ms
        if 0 <= boundary_t_ms <= duration:
            axes[1, 1].axvline(boundary_t_ms, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return comp, compensation

def verify_window_structure():
    """
    Verify the window structure calculations
    """
    print("Verifying Window Structure:")
    print("=" * 50)
    
    # Test different configurations
    test_cases = [
        {"n_main": 3, "duration": 300},
        {"n_main": 5, "duration": 500},
        {"n_main": 10, "duration": 884},
    ]
    
    for case in test_cases:
        n_main = case["n_main"]
        duration = case["duration"]
        n_boundaries = n_main + 3
        
        print(f"\nConfiguration: {n_main} main windows, {duration} duration")
        
        # Create simple parameters
        a_params = list(range(n_boundaries))
        b_params = list(range(n_boundaries))
        
        comp = Compensate(a_params, b_params, duration)
        
        print(f"  Window structure verification:")
        print(f"    Main window size: {comp.main_window_size:.3f}")
        print(f"    Expected: {duration/n_main:.3f}")
        print(f"    Match: {abs(comp.main_window_size - duration/n_main) < 1e-10}")
        
        print(f"  Boundary coverage:")
        print(f"    Range: {comp.boundary_offsets[0]:.1f} to {comp.boundary_offsets[-1]:.1f}")
        print(f"    Main range: 0 to {duration}")
        print(f"    Edge coverage: {-comp.main_window_size:.1f} to {duration + comp.main_window_size:.1f}")

if __name__ == "__main__":
    print("3D Multi-Window Compensation - Realistic Example")
    print("=" * 60)
    
    # Verify window structure calculations
    verify_window_structure()
    
    print("\n" + "=" * 60)
    print("Realistic Test: 884ms, 10 windows, 1280×720")
    
    # Test with realistic scenario  
    comp, compensation = test_realistic_scenario()
    
    print("\n" + "=" * 60)
    print("Additional Tests")
    
    # Test different window numbers with same realistic parameters
    durations_and_windows = [
        (884, 5),   # 884ms, 5 main windows  
        (500, 8),   # 500ms, 8 main windows
        (1000, 12), # 1000ms, 12 main windows
    ]
    
    for duration, n_windows in durations_and_windows:
        print(f"\nTesting {duration}ms duration with {n_windows} main windows:")
        
        n_boundaries = n_windows + 3
        a_params = np.linspace(0, 5, n_boundaries)
        b_params = np.linspace(-30, -50, n_boundaries)
        
        comp_test = Compensate(a_params, b_params, duration)
        print(f"  Window size: {comp_test.main_window_size:.2f}")
        print(f"  Boundary range: {comp_test.boundary_offsets[0]:.1f} to {comp_test.boundary_offsets[-1]:.1f}")
    
    print(f"\nUsage:")
    print(f"  a_params = [list of {n_windows+3} x-coefficients]")
    print(f"  b_params = [list of {n_windows+3} y-coefficients]") 
    print(f"  comp = Compensate(a_params, b_params, duration)")
    print(f"  compensation = comp(x, y, t, width, height)")