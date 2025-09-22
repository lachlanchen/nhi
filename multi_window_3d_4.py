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
        self.a_params = np.array(a_params)
        self.b_params = np.array(b_params)  
        self.duration = duration
        self.temperature = temperature
        
        # Validate input
        if len(self.a_params) != len(self.b_params):
            raise ValueError("a_params and b_params must have the same length")
        
        # Calculate window structure
        self.num_boundaries = len(self.a_params)
        self.num_main_windows = self.num_boundaries - 3  # n_main = len(ax) - 1 - 2
        self.num_total_windows = self.num_main_windows + 2  # Add 2 edge windows
        
        # Calculate boundary offsets automatically
        # Window size is determined by main windows only
        self.window_size = duration / self.num_main_windows
        
        # Boundaries: [-window_size, 0, window_size, 2*window_size, ..., (n_main+1)*window_size]
        self.boundary_offsets = np.array([
            (i - 1) * self.window_size
            for i in range(self.num_boundaries)
        ])
        
        print(f"Compensate initialized:")
        print(f"  Parameters: {self.num_boundaries}")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.num_total_windows} (including 2 edge windows)")
        print(f"  Duration: {self.duration}")
        print(f"  Main window size: {self.window_size:.2f}")
        print(f"  Boundary offsets: {self.boundary_offsets}")
        print(f"  A parameters: {self.a_params}")
        print(f"  B parameters: {self.b_params}")
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_boundary_surfaces(self, x_norm, y_norm):
        """
        Compute boundary surface values: t = a_i * x_norm + b_i * y_norm + offset_i
        
        Args:
            x_norm: Normalized X coordinates (x/width) [len_x]
            y_norm: Normalized Y coordinates (y/height) [len_y]
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
        
        Args:
            x_norm: Normalized X coordinates [len_x]
            y_norm: Normalized Y coordinates [len_y]
            t_shifted: Time shifted to start from 0 [len_t]
        """
        boundary_values = self.get_boundary_surfaces(x_norm, y_norm)  # [num_boundaries, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x_norm, y_norm, t_shifted, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_total_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions - no clamping needed
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
        Interpolate parameters within each window - no clamping needed
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
            
            # No clamping - let the soft weighting handle boundaries naturally
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
            x: X coordinates [len_x]
            y: Y coordinates [len_y] 
            t: T coordinates [len_t]
            width: Sensor width for normalization
            height: Sensor height for normalization
            
        Returns:
            compensation: Total compensation [len_x, len_y, len_t]
        """
        # Normalize spatial coordinates
        x_norm = np.asarray(x) / width
        y_norm = np.asarray(y) / height
        
        # Shift time to start from 0
        t_array = np.asarray(t)
        t_min = t_array.min()
        t_shifted = t_array - t_min
        
        # Verify duration matches
        actual_duration = t_shifted.max()
        if abs(actual_duration - self.duration) > 1e-6:
            print(f"Warning: Actual duration {actual_duration:.2f} != expected {self.duration:.2f}")
        
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

def create_example_parameters(n_main_windows=3, duration=1000.0):
    """
    Create example parameters for testing
    
    Args:
        n_main_windows: Number of main windows
        duration: Time duration
    """
    # Calculate number of boundaries needed
    num_boundaries = n_main_windows + 3
    
    # Create example parameter arrays
    a_params = np.linspace(0.1, 0.5, num_boundaries)
    b_params = np.linspace(0.05, 0.3, num_boundaries)
    
    return a_params.tolist(), b_params.tolist()

def visualize_compensation_3d(comp, x_range, y_range, t_range, width, height):
    """
    Simple 3D visualization of compensation
    """
    print("Computing 3D compensation...")
    compensation = comp(x_range, y_range, t_range, width, height)
    
    print(f"Compensation shape: {compensation.shape}")
    print(f"Compensation range: {compensation.min():.6f} to {compensation.max():.6f}")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Select middle slices
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # 1. Compensation at fixed T (XY slice)
    ax1 = plt.subplot(2, 3, 1)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    im1 = ax1.contourf(X, Y, compensation[:, :, t_mid], levels=20, cmap='coolwarm')
    ax1.set_title(f'Compensation at T = {t_range[t_mid]:.1f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Compensation at fixed Y (XT slice)  
    ax2 = plt.subplot(2, 3, 2)
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    im2 = ax2.contourf(X_2d, T_2d, compensation[:, y_mid, :], levels=20, cmap='coolwarm')
    ax2.set_title(f'Compensation at Y = {y_range[y_mid]:.1f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('T')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Compensation at fixed X (YT slice)
    ax3 = plt.subplot(2, 3, 3)
    Y_2d, T_2d = np.meshgrid(y_range, t_range, indexing='ij')
    im3 = ax3.contourf(Y_2d, T_2d, compensation[x_mid, :, :], levels=20, cmap='coolwarm')
    ax3.set_title(f'Compensation at X = {x_range[x_mid]:.1f}')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('T')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 3D boundary surfaces (normalized coordinates)
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    x_norm = np.array(x_range) / width
    y_norm = np.array(y_range) / height
    t_shifted = np.array(t_range) - t_range[0]
    
    boundary_surfaces = comp.get_boundary_surfaces(x_norm[::3], y_norm[::3])
    X_surf, Y_surf = np.meshgrid(x_norm[::3], y_norm[::3], indexing='ij')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_surfaces)))
    for i in range(0, len(boundary_surfaces), 2):  # Show every 2nd surface
        Z_surf = boundary_surfaces[i]
        ax4.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.6, color=colors[i])
    
    ax4.set_xlabel('X (normalized)')
    ax4.set_ylabel('Y (normalized)')
    ax4.set_zlabel('T (shifted)')
    ax4.set_title('Boundary Surfaces')
    
    # 5. Compensation along line (fixed X, Y)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_range, compensation[x_mid, y_mid, :], 'b-', linewidth=2)
    ax5.set_title(f'Compensation at X={x_range[x_mid]:.1f}, Y={y_range[y_mid]:.1f}')
    ax5.set_xlabel('T')
    ax5.set_ylabel('Compensation')
    ax5.grid(True)
    
    # Add boundary intersections (in shifted time)
    x_norm_point = x_range[x_mid] / width
    y_norm_point = y_range[y_mid] / height
    boundary_at_point = comp.get_boundary_surfaces([x_norm_point], [y_norm_point])
    t_min_actual = t_range[0]
    
    for i, t_boundary_shifted in enumerate(boundary_at_point[:, 0, 0]):
        t_boundary_original = t_boundary_shifted + t_min_actual
        if t_range.min() <= t_boundary_original <= t_range.max():
            ax5.axvline(t_boundary_original, color='r', linestyle='--', alpha=0.7, 
                       label=f'Boundary {i}' if i < 3 else '')
    
    if len([b for b in boundary_at_point[:, 0, 0] if t_range.min() <= b + t_min_actual <= t_range.max()]) > 0:
        ax5.legend()
    
    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""3D Compensation Statistics:

Input data:
  Width: {width}, Height: {height}
  Time range: {t_range.min():.1f} to {t_range.max():.1f}
  Duration: {comp.duration:.2f}

Grid: {len(x_range)} × {len(y_range)} × {len(t_range)}

Compensation range: 
{compensation.min():.6f} to {compensation.max():.6f}

Model parameters:
  Main windows: {comp.num_main_windows}
  Total windows: {comp.num_total_windows}  
  Boundaries: {comp.num_boundaries}
  Main window size: {comp.window_size:.2f}
  Temperature: {comp.temperature}

A params: {comp.a_params}
B params: {comp.b_params}"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def test_arbitrary_data():
    """
    Test with arbitrary data sizes and ranges
    """
    print("Testing with arbitrary data configurations:")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Small sensor, short time",
            "width": 640, "height": 480,
            "x_range": np.linspace(0, 640, 30),
            "y_range": np.linspace(0, 480, 25), 
            "t_range": np.linspace(1000, 1500, 40),
            "n_windows": 2
        },
        {
            "name": "Large sensor, long time",
            "width": 1280, "height": 720,
            "x_range": np.linspace(0, 1280, 50),
            "y_range": np.linspace(0, 720, 40),
            "t_range": np.linspace(5000, 15000, 60),  
            "n_windows": 4
        },
        {
            "name": "Irregular ranges",
            "width": 800, "height": 600,
            "x_range": np.linspace(100, 700, 35),
            "y_range": np.linspace(50, 550, 30),
            "t_range": np.linspace(-1000, 2000, 50),
            "n_windows": 3
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['name']}")
        
        # Calculate duration
        duration = case['t_range'].max() - case['t_range'].min()
        print(f"  Duration: {duration}")
        
        # Create parameters
        a_params, b_params = create_example_parameters(case['n_windows'], duration)
        print(f"  Parameters: {len(a_params)} boundaries for {case['n_windows']} main windows")
        
        # Create compensator
        comp = Compensate(a_params, b_params, duration)
        
        # Test computation
        compensation = comp(case['x_range'], case['y_range'], case['t_range'], 
                          case['width'], case['height'])
        
        print(f"  Result shape: {compensation.shape}")
        print(f"  Compensation range: {compensation.min():.6f} to {compensation.max():.6f}")

def test_parameter_variations():
    """
    Test different parameter configurations
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fixed test data
    width, height = 1000, 800
    x_range = np.linspace(0, width, 40)
    y_range = np.linspace(0, height, 30)
    t_range = np.linspace(2000, 7000, 50)
    duration = t_range.max() - t_range.min()
    
    param_configs = [
        {
            "name": "2 Main Windows", 
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5 boundaries = 2 main windows
            "b": [0.05, 0.1, 0.15, 0.2, 0.25]
        },
        {
            "name": "3 Main Windows",
            "a": [0.1, 0.15, 0.3, 0.25, 0.4, 0.35],  # 6 boundaries = 3 main windows  
            "b": [0.05, 0.08, 0.15, 0.12, 0.2, 0.18]
        },
        {
            "name": "4 Main Windows",
            "a": [0.08, 0.12, 0.25, 0.2, 0.35, 0.3, 0.45],  # 7 boundaries = 4 main windows
            "b": [0.04, 0.06, 0.12, 0.1, 0.18, 0.15, 0.22]
        },
        {
            "name": "Negative A Parameters",
            "a": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2],  # 6 boundaries = 3 main windows
            "b": [0.05, 0.08, 0.12, 0.1, 0.15, 0.18]
        },
        {
            "name": "Strong Y-dependence", 
            "a": [0.05, 0.08, 0.12, 0.1, 0.15, 0.18],  # Small x coefficients
            "b": [0.2, 0.3, 0.5, 0.4, 0.6, 0.7]       # Large y coefficients
        },
        {
            "name": "Mixed Signs",
            "a": [-0.1, 0.05, 0.15, -0.05, 0.2, 0.1],
            "b": [0.08, -0.03, 0.12, 0.05, -0.08, 0.1]
        }
    ]
    
    # Fixed T slice for comparison
    t_slice_idx = len(t_range) // 2
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    for idx, config in enumerate(param_configs):
        comp = Compensate(config["a"], config["b"], duration)
        compensation = comp(x_range, y_range, t_range, width, height)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, Y, compensation[:, :, t_slice_idx], levels=15, cmap='coolwarm')
        title = f'{config["name"]}\n{comp.num_main_windows} main, {comp.num_boundaries} boundaries'
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
        
        print(f"{config['name']}: {compensation.min():.6f} to {compensation.max():.6f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Improved 3D Multi-Window Compensation")
    print("=" * 50)
    
    # Test with arbitrary data
    test_arbitrary_data()
    
    print("\n" + "=" * 50)
    print("Example with typical event camera data")
    
    # Typical event camera setup
    width, height = 1280, 720
    x_range = np.linspace(0, width, 50)
    y_range = np.linspace(0, height, 40) 
    t_range = np.linspace(1000000, 1005000, 60)  # 5ms duration in microseconds
    duration = t_range.max() - t_range.min()
    
    print(f"Sensor: {width}×{height}")
    print(f"Time range: {t_range.min()} to {t_range.max()} μs")
    print(f"Duration: {duration} μs ({duration/1000:.1f} ms)")
    
    # Create parameters for 3 main windows
    a_params, b_params = create_example_parameters(n_main_windows=3, duration=duration)
    comp = Compensate(a_params, b_params, duration)
    
    # Visualize
    visualize_compensation_3d(comp, x_range, y_range, t_range, width, height)
    
    print("\n" + "=" * 50) 
    print("Testing Parameter Variations")
    test_parameter_variations()
    
    print(f"\nUsage:")
    print(f"  a_params, b_params = create_example_parameters(n_main_windows, duration)")
    print(f"  comp = Compensate(a_params, b_params, duration)")
    print(f"  compensation = comp(x, y, t, width, height)")