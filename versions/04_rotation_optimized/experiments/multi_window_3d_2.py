import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Simple3DMultiWindow:
    def __init__(self, a_params, b_params, boundary_offsets, temperature=1):
        """
        Simple 3D multi-window compensation with pre-initialized parameters
        
        Args:
            a_params: X-direction coefficients [a_0, a_1, ..., a_n] for boundary surfaces
            b_params: Y-direction coefficients [b_0, b_1, ..., b_n] for boundary surfaces  
            boundary_offsets: Offset values [offset_0, offset_1, ..., offset_n] for boundary surfaces
            temperature: Smoothness parameter for soft assignments (default 1)
        """
        self.a_params = np.array(a_params)
        self.b_params = np.array(b_params)  
        self.boundary_offsets = np.array(boundary_offsets)
        self.temperature = temperature
        
        # Validate input
        if len(self.a_params) != len(self.b_params) or len(self.a_params) != len(self.boundary_offsets):
            raise ValueError("a_params, b_params, and boundary_offsets must have the same length")
        
        self.num_boundaries = len(self.a_params)
        self.num_windows = self.num_boundaries - 1
        
        print(f"3D Multi-Window Compensation initialized:")
        print(f"  Boundary surfaces: {self.num_boundaries}")
        print(f"  Windows: {self.num_windows}")
        print(f"  Total parameters: {2 * self.num_boundaries}")
        print(f"  Temperature: {self.temperature}")
        print(f"  A parameters (x-coeff): {self.a_params}")
        print(f"  B parameters (y-coeff): {self.b_params}")
        print(f"  Boundary offsets: {self.boundary_offsets}")
    
    def get_boundary_surfaces(self, x, y):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        
        Args:
            x: X coordinates [len_x] or scalar
            y: Y coordinates [len_y] or scalar
            
        Returns:
            boundary_values: [num_boundaries, len_x, len_y] array or [num_boundaries] for scalars
        """
        # Convert to numpy arrays if needed
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle different input shapes
        if x.ndim == 0 and y.ndim == 0:  # Both scalars
            boundary_values = self.a_params * x + self.b_params * y + self.boundary_offsets
        elif x.ndim == 1 and y.ndim == 1:  # Both arrays - create meshgrid
            X, Y = np.meshgrid(x, y, indexing='ij')
            boundary_values = np.zeros((self.num_boundaries, len(x), len(y)))
            for i in range(self.num_boundaries):
                boundary_values[i] = self.a_params[i] * X + self.b_params[i] * Y + self.boundary_offsets[i]
        elif x.ndim == 2 and y.ndim == 2:  # Already meshgrid format
            boundary_values = np.zeros((self.num_boundaries,) + x.shape)
            for i in range(self.num_boundaries):
                boundary_values[i] = self.a_params[i] * x + self.b_params[i] * y + self.boundary_offsets[i]
        else:
            raise ValueError("Unsupported input shapes for x and y")
        
        return boundary_values
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_window_memberships(self, x, y, t):
        """
        Compute soft membership for each window in 3D
        
        Args:
            x: X coordinates [len_x]
            y: Y coordinates [len_y] 
            t: T coordinates [len_t]
            
        Returns:
            memberships: [num_windows, len_x, len_y, len_t] array
        """
        # Get boundary surfaces for this x, y grid
        boundary_values = self.get_boundary_surfaces(x, y)  # [num_boundaries, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions
            # σ((t - lower)/τ) * σ((upper - t)/τ)
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [num_windows, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len_x, len_y, len_t]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t):
        """
        For each window, interpolate between boundary parameters in 3D
        
        Returns:
            interpolated_slopes_a: [num_windows, len_x, len_y, len_t] (x-direction slopes)
            interpolated_slopes_b: [num_windows, len_x, len_y, len_t] (y-direction slopes)
        """
        boundary_values = self.get_boundary_surfaces(x, y)  # [num_boundaries, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        interpolated_slopes_a = []
        interpolated_slopes_b = []
        
        for i in range(self.num_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]      # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Interpolation parameter α ∈ [0,1]
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)  # Avoid division by zero
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to [0,1]
            
            # Interpolate between parameters a_i, a_{i+1} for x-direction
            slope_a = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            
            # Interpolate between parameters b_i, b_{i+1} for y-direction
            slope_b = (1 - alpha) * self.b_params[i] + alpha * self.b_params[i + 1]
            
            interpolated_slopes_a.append(slope_a)
            interpolated_slopes_b.append(slope_b)
        
        return np.array(interpolated_slopes_a), np.array(interpolated_slopes_b)
    
    def __call__(self, x, y, t):
        """
        Compute the final compensation for given (x, y, t) coordinates
        
        Args:
            x: X coordinates [len_x]
            y: Y coordinates [len_y]
            t: T coordinates [len_t]
            
        Returns:
            compensation: [len_x, len_y, len_t] array (total compensation)
            compensation_x: [len_x, len_y, len_t] array (x-component)
            compensation_y: [len_x, len_y, len_t] array (y-component)
        """
        # Get window memberships
        memberships = self.compute_window_memberships(x, y, t)  # [num_windows, len_x, len_y, len_t]
        
        # Get interpolated slopes for each window
        slopes_a, slopes_b = self.compute_within_window_interpolation(x, y, t)  # [num_windows, len_x, len_y, len_t]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        # Weighted sum of compensations
        # C = Σᵢ wᵢ × (slope_a_i × x + slope_b_i × y)
        compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
        compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
        compensation = compensation_x + compensation_y
        
        return compensation, compensation_x, compensation_y

def create_example_parameters(num_windows=3, data_range=300, extend_range=100):
    """
    Create example parameters for testing
    
    Args:
        num_windows: Number of main windows
        data_range: Main data range
        extend_range: Extension on each side
        
    Returns:
        a_params, b_params, boundary_offsets
    """
    total_windows = num_windows + 2  # Including edge windows
    num_boundaries = total_windows + 1
    
    # Create boundary offsets
    total_range = data_range + 2 * extend_range
    window_width = total_range / total_windows
    boundary_offsets = []
    
    for i in range(num_boundaries):
        offset = -extend_range + i * window_width
        boundary_offsets.append(offset)
    
    # Create example parameters
    a_params = np.linspace(0.05, 0.35, num_boundaries)
    b_params = np.linspace(0.02, 0.18, num_boundaries)
    
    return a_params, b_params, np.array(boundary_offsets)

def visualize_3d_boundary_surfaces(comp, x_range, y_range):
    """
    Visualize the 3D boundary surfaces
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create meshgrids for surfaces
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # Get boundary surfaces
    boundary_surfaces = comp.get_boundary_surfaces(x_range, y_range)
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_surfaces)))
    
    # 3D surface plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i, surface in enumerate(boundary_surfaces):
        ax1.plot_surface(X, Y, surface, alpha=0.6, color=colors[i], label=f'Surface {i}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('T')
    ax1.set_title('3D Boundary Surfaces')
    
    # Contour plots at different Y slices
    y_slice_indices = [len(y_range)//4, len(y_range)//2, 3*len(y_range)//4]
    
    for idx, y_idx in enumerate(y_slice_indices):
        ax = fig.add_subplot(2, 3, 2 + idx)
        
        for i, surface in enumerate(boundary_surfaces):
            ax.plot(x_range, surface[:, y_idx], '--', linewidth=2, 
                   color=colors[i], label=f'Surface {i}')
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('T')
        ax.set_title(f'Boundary Lines at Y = {y_range[y_idx]:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Contour plots at different X slices
    x_slice_indices = [len(x_range)//4, len(x_range)//2]
    
    for idx, x_idx in enumerate(x_slice_indices):
        ax = fig.add_subplot(2, 3, 5 + idx)
        
        for i, surface in enumerate(boundary_surfaces):
            ax.plot(y_range, surface[x_idx, :], '--', linewidth=2, 
                   color=colors[i], label=f'Surface {i}')
        
        ax.set_xlabel('Y coordinate')
        ax.set_ylabel('T')
        ax.set_title(f'Boundary Lines at X = {x_range[x_idx]:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_3d_compensation(comp, x_range, y_range, t_range):
    """
    Visualize the 3D compensation function
    """
    print(f"Computing 3D compensation...")
    compensation, comp_x, comp_y = comp(x_range, y_range, t_range)
    memberships = comp.compute_window_memberships(x_range, y_range, t_range)
    
    print(f"Compensation shape: {compensation.shape}")
    print(f"Memberships shape: {memberships.shape}")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Select middle indices for slicing
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # Compensation at fixed T (XY slice)
    ax1 = plt.subplot(3, 4, 1)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    im1 = ax1.contourf(X, Y, compensation[:, :, t_mid], levels=20, cmap='coolwarm')
    ax1.set_title(f'Total Compensation at T = {t_range[t_mid]:.1f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # X-component at fixed T
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.contourf(X, Y, comp_x[:, :, t_mid], levels=20, cmap='coolwarm')
    ax2.set_title(f'X-Compensation at T = {t_range[t_mid]:.1f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Y-component at fixed T
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.contourf(X, Y, comp_y[:, :, t_mid], levels=20, cmap='coolwarm')
    ax3.set_title(f'Y-Compensation at T = {t_range[t_mid]:.1f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Total membership at fixed T
    ax4 = plt.subplot(3, 4, 4)
    total_membership = np.sum(memberships, axis=0)
    im4 = ax4.contourf(X, Y, total_membership[:, :, t_mid], levels=20, cmap='RdBu_r')
    ax4.set_title(f'Total Membership at T = {t_range[t_mid]:.1f}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # Plot window memberships at fixed T (up to 4 windows)
    num_windows_to_show = min(4, comp.num_windows)
    for i in range(num_windows_to_show):
        ax = plt.subplot(3, 4, 5 + i)
        im = ax.contourf(X, Y, memberships[i, :, :, t_mid], levels=20, cmap='viridis')
        ax.set_title(f'Window {i} at T = {t_range[t_mid]:.1f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Compensation slices at fixed X and Y
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(t_range, compensation[x_mid, y_mid, :], 'b-', linewidth=2, label='Total')
    ax9.plot(t_range, comp_x[x_mid, y_mid, :], 'r--', linewidth=2, label='X-component')
    ax9.plot(t_range, comp_y[x_mid, y_mid, :], 'g--', linewidth=2, label='Y-component')
    ax9.set_title(f'Compensation at X={x_range[x_mid]:.1f}, Y={y_range[y_mid]:.1f}')
    ax9.set_xlabel('Time T')
    ax9.set_ylabel('Compensation')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Add boundary surface intersections
    boundary_at_point = comp.get_boundary_surfaces(x_range[x_mid], y_range[y_mid])
    colors_lines = plt.cm.tab10(np.linspace(0, 1, len(boundary_at_point)))
    for i, t_boundary in enumerate(boundary_at_point):
        if t_range.min() <= t_boundary <= t_range.max():
            ax9.axvline(t_boundary, color=colors_lines[i], linestyle='--', alpha=0.7, label=f'Surface {i}')
    
    # Compensation at fixed Y (XT slice)
    ax10 = plt.subplot(3, 4, 10)
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    im10 = ax10.contourf(X_2d, T_2d, compensation[:, y_mid, :], levels=20, cmap='coolwarm')
    ax10.set_title(f'Compensation at Y = {y_range[y_mid]:.1f}')
    ax10.set_xlabel('X')
    ax10.set_ylabel('T')
    plt.colorbar(im10, ax=ax10, shrink=0.8)
    
    # Compensation at fixed X (YT slice)
    ax11 = plt.subplot(3, 4, 11)
    Y_2d, T_2d = np.meshgrid(y_range, t_range, indexing='ij')
    im11 = ax11.contourf(Y_2d, T_2d, compensation[x_mid, :, :], levels=20, cmap='coolwarm')
    ax11.set_title(f'Compensation at X = {x_range[x_mid]:.1f}')
    ax11.set_xlabel('Y')
    ax11.set_ylabel('T')
    plt.colorbar(im11, ax=ax11, shrink=0.8)
    
    # Statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    stats_text = f"""3D Compensation Stats:
Windows: {comp.num_windows}
Boundary surfaces: {comp.num_boundaries}
Total parameters: {2 * comp.num_boundaries}

Grid: {len(x_range)} × {len(y_range)} × {len(t_range)}

Compensation ranges:
Total: {compensation.min():.2f} to {compensation.max():.2f}
X-comp: {comp_x.min():.2f} to {comp_x.max():.2f}  
Y-comp: {comp_y.min():.2f} to {comp_y.max():.2f}

Membership sum error: 
{np.abs(total_membership - 1.0).max():.6f}

Parameters:
A: {comp.a_params}
B: {comp.b_params}
Offsets: {comp.boundary_offsets}"""
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return compensation, comp_x, comp_y, memberships

def test_different_parameter_sets():
    """
    Test different parameter configurations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    x_range = np.linspace(0, 200, 40)
    y_range = np.linspace(0, 150, 30)
    t_range = np.linspace(-150, 450, 50)
    
    # Different parameter sets
    param_configs = [
        {
            "name": "Positive Parameters",
            "a_params": [0.05, 0.15, 0.25, 0.35],
            "b_params": [0.02, 0.06, 0.10, 0.14],
            "offsets": [-100, 0, 100, 200]
        },
        {
            "name": "Negative Parameters", 
            "a_params": [-0.35, -0.25, -0.15, -0.05],
            "b_params": [-0.14, -0.10, -0.06, -0.02],
            "offsets": [-100, 0, 100, 200]
        },
        {
            "name": "Mixed Parameters",
            "a_params": [-0.1, 0.0, 0.1, 0.2],
            "b_params": [-0.05, 0.0, 0.05, 0.1],
            "offsets": [-100, 0, 100, 200]
        },
        {
            "name": "Uniform Slopes",
            "a_params": [0.1, 0.1, 0.1, 0.1],
            "b_params": [0.05, 0.05, 0.05, 0.05],
            "offsets": [-100, 0, 100, 200]
        },
        {
            "name": "Strong Y-dependence",
            "a_params": [0.05, 0.1, 0.15, 0.2],
            "b_params": [0.1, 0.2, 0.3, 0.4],
            "offsets": [-100, 0, 100, 200]
        },
        {
            "name": "Variable Spacing",
            "a_params": [0.1, 0.2, 0.15, 0.25],
            "b_params": [0.05, 0.1, 0.08, 0.12],
            "offsets": [-150, -50, 150, 350]
        }
    ]
    
    # Fixed T slice for visualization
    t_slice_idx = len(t_range) // 2
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    for idx, config in enumerate(param_configs):
        comp = Simple3DMultiWindow(
            config["a_params"], 
            config["b_params"], 
            config["offsets"],
            temperature=1
        )
        
        compensation, _, _ = comp(x_range, y_range, t_range)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, Y, compensation[:, :, t_slice_idx], levels=15, cmap='coolwarm')
        ax.set_title(f'{config["name"]}\n({comp.num_windows} windows)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        print(f"{config['name']}: Compensation range {compensation.min():.2f} to {compensation.max():.2f}")
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example of how to use the class
    """
    print("Example Usage:")
    print("=" * 50)
    
    # Method 1: Use helper function to create parameters
    print("Method 1: Using helper function")
    a_params, b_params, offsets = create_example_parameters(num_windows=3, data_range=300, extend_range=100)
    comp = Simple3DMultiWindow(a_params, b_params, offsets, temperature=1)
    
    # Method 2: Define parameters directly
    print("\nMethod 2: Direct parameter definition")
    a_params = [0.1, 0.2, 0.15, 0.25]  # 4 boundaries for 3 windows
    b_params = [0.05, 0.1, 0.08, 0.12]
    offsets = [-100, 0, 100, 200]
    comp2 = Simple3DMultiWindow(a_params, b_params, offsets, temperature=2)
    
    # Method 3: Create from existing parameters
    print("\nMethod 3: From numpy arrays")
    a_params = np.array([-0.1, 0.0, 0.1, 0.2, 0.3])  # 5 boundaries for 4 windows
    b_params = np.array([0.02, 0.04, 0.06, 0.08, 0.1])
    offsets = np.array([-200, -100, 0, 100, 200])
    comp3 = Simple3DMultiWindow(a_params, b_params, offsets)
    
    # Use the compensation function
    x_range = np.linspace(0, 100, 20)
    y_range = np.linspace(0, 80, 15)
    t_range = np.linspace(-50, 250, 30)
    
    print(f"\nComputing compensation on {len(x_range)}×{len(y_range)}×{len(t_range)} grid...")
    compensation, comp_x, comp_y = comp3(x_range, y_range, t_range)
    
    print(f"Results:")
    print(f"  Total compensation range: {compensation.min():.2f} to {compensation.max():.2f}")
    print(f"  X-component range: {comp_x.min():.2f} to {comp_x.max():.2f}")
    print(f"  Y-component range: {comp_y.min():.2f} to {comp_y.max():.2f}")
    
    return comp3

if __name__ == "__main__":
    print("3D Multi-Window Compensation - Clean Interface")
    print("=" * 60)
    
    # Show example usage
    comp = example_usage()
    
    print("\n" + "=" * 60)
    print("Main 3D Visualization")
    print("=" * 60)
    
    # Main visualization
    x_range = np.linspace(0, 200, 50)
    y_range = np.linspace(0, 150, 40)
    t_range = np.linspace(-200, 300, 60)
    
    # Visualize boundary surfaces
    visualize_3d_boundary_surfaces(comp, x_range, y_range)
    
    # Visualize compensation
    compensation, comp_x, comp_y, memberships = visualize_3d_compensation(comp, x_range, y_range, t_range)
    
    print("\n" + "=" * 60)
    print("Testing Different Parameter Sets")
    print("=" * 60)
    
    # Test different configurations
    test_different_parameter_sets()
    
    print("\n3D Visualization complete!")
    print("Usage: comp = Simple3DMultiWindow(a_params, b_params, offsets)")
    print("       compensation, comp_x, comp_y = comp(x, y, t)")