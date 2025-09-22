import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Compensate:
    def __init__(self, a_params, b_params, boundary_offsets, temperature=1):
        """
        3D Multi-window compensation class
        
        Args:
            a_params: X-direction coefficients [6 values for 3 main windows + 2 edge]
            b_params: Y-direction coefficients [6 values for 3 main windows + 2 edge]  
            boundary_offsets: Offset values [6 values for boundary surfaces]
            temperature: Smoothness parameter (default 1)
        """
        self.a_params = np.array(a_params)
        self.b_params = np.array(b_params)  
        self.boundary_offsets = np.array(boundary_offsets)
        self.temperature = temperature
        
        # For 3 main windows, we have 5 total windows (3 main + 2 edge)
        # This requires 6 boundary surfaces
        self.num_windows = 5  # 3 main + 2 edge
        self.num_boundaries = 6
        
        print(f"Compensate initialized: {self.num_windows} windows, {self.num_boundaries} boundary surfaces")
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_boundary_surfaces(self, x, y):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y, indexing='ij')
            boundary_values = np.zeros((self.num_boundaries, len(x), len(y)))
            for i in range(self.num_boundaries):
                boundary_values[i] = self.a_params[i] * X + self.b_params[i] * Y + self.boundary_offsets[i]
        else:
            raise ValueError("x and y must be 1D arrays")
        
        return boundary_values
    
    def compute_window_memberships(self, x, y, t):
        """
        Compute soft membership for each window
        """
        boundary_values = self.get_boundary_surfaces(x, y)  # [6, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [5, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)
        memberships_sum = np.maximum(memberships_sum, 1e-8)
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t):
        """
        Interpolate parameters within each window
        """
        boundary_values = self.get_boundary_surfaces(x, y)  # [6, len_x, len_y]
        
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        interpolated_slopes_a = []
        interpolated_slopes_b = []
        
        for i in range(self.num_windows):
            lower_bound = boundary_values[i][:, :, np.newaxis]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]
            
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # Interpolate between parameters
            slope_a = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            slope_b = (1 - alpha) * self.b_params[i] + alpha * self.b_params[i + 1]
            
            interpolated_slopes_a.append(slope_a)
            interpolated_slopes_b.append(slope_b)
        
        return np.array(interpolated_slopes_a), np.array(interpolated_slopes_b)
    
    def __call__(self, x, y, t):
        """
        Compute compensation for given (x, y, t) coordinates
        
        Returns:
            compensation: Total compensation [len_x, len_y, len_t]
        """
        # Get window memberships
        memberships = self.compute_window_memberships(x, y, t)  # [5, len_x, len_y, len_t]
        
        # Get interpolated slopes
        slopes_a, slopes_b = self.compute_within_window_interpolation(x, y, t)  # [5, len_x, len_y, len_t]
        
        # Create meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        # Weighted sum of compensations: C = Σᵢ wᵢ × (aᵢ×x + bᵢ×y)
        compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
        compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
        compensation = compensation_x + compensation_y
        
        return compensation

def create_standard_parameters():
    """
    Create standard parameters for 3 main windows
    """
    # 6 boundary surfaces for 5 windows (3 main + 2 edge)
    a_params = [0.05, 0.1, 0.2, 0.15, 0.25, 0.3]  # X-direction slopes
    b_params = [0.02, 0.05, 0.1, 0.08, 0.12, 0.15]  # Y-direction slopes  
    boundary_offsets = [-100, 0, 100, 200, 300, 400]  # Surface offsets
    
    return a_params, b_params, boundary_offsets

def visualize_compensation_3d(comp, x_range, y_range, t_range):
    """
    Simple 3D visualization of compensation
    """
    print("Computing 3D compensation...")
    compensation = comp(x_range, y_range, t_range)
    
    print(f"Compensation shape: {compensation.shape}")
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    
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
    
    # 4. 3D boundary surfaces
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    boundary_surfaces = comp.get_boundary_surfaces(x_range, y_range)
    X_surf, Y_surf = np.meshgrid(x_range[::3], y_range[::3], indexing='ij')  # Reduced resolution
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundary_surfaces)))
    for i in range(0, len(boundary_surfaces), 2):  # Show every 2nd surface
        Z_surf = boundary_surfaces[i][::3, ::3]
        ax4.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.6, color=colors[i])
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('T')
    ax4.set_title('Boundary Surfaces')
    
    # 5. Compensation along line (fixed X, Y)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_range, compensation[x_mid, y_mid, :], 'b-', linewidth=2)
    ax5.set_title(f'Compensation at X={x_range[x_mid]:.1f}, Y={y_range[y_mid]:.1f}')
    ax5.set_xlabel('T')
    ax5.set_ylabel('Compensation')
    ax5.grid(True)
    
    # Add boundary intersections
    boundary_at_point = comp.get_boundary_surfaces([x_range[x_mid]], [y_range[y_mid]])
    for i, t_boundary in enumerate(boundary_at_point[:, 0, 0]):
        if t_range.min() <= t_boundary <= t_range.max():
            ax5.axvline(t_boundary, color='r', linestyle='--', alpha=0.7)
    
    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""3D Compensation Statistics:

Grid size: {len(x_range)} × {len(y_range)} × {len(t_range)}

Compensation range: 
{compensation.min():.3f} to {compensation.max():.3f}

Parameters:
A (x-coeff): {comp.a_params}
B (y-coeff): {comp.b_params}
Offsets: {comp.boundary_offsets}

Windows: {comp.num_windows}
Boundary surfaces: {comp.num_boundaries}
Temperature: {comp.temperature}"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def test_different_parameters():
    """
    Test different parameter sets
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x_range = np.linspace(0, 200, 30)
    y_range = np.linspace(0, 150, 25)
    t_range = np.linspace(-150, 450, 40)
    
    # Different parameter sets
    param_sets = [
        {
            "name": "Standard",
            "a": [0.05, 0.1, 0.2, 0.15, 0.25, 0.3],
            "b": [0.02, 0.05, 0.1, 0.08, 0.12, 0.15],
            "offsets": [-100, 0, 100, 200, 300, 400]
        },
        {
            "name": "Negative A",
            "a": [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05],
            "b": [0.02, 0.05, 0.1, 0.08, 0.12, 0.15],
            "offsets": [-100, 0, 100, 200, 300, 400]
        },
        {
            "name": "Strong Y-dependence",
            "a": [0.05, 0.1, 0.15, 0.1, 0.2, 0.25],
            "b": [0.1, 0.2, 0.3, 0.25, 0.35, 0.4],
            "offsets": [-100, 0, 100, 200, 300, 400]
        },
        {
            "name": "Mixed Signs",
            "a": [-0.1, 0.05, 0.15, -0.05, 0.2, 0.1],
            "b": [0.05, -0.02, 0.1, 0.03, -0.05, 0.08],
            "offsets": [-100, 0, 100, 200, 300, 400]
        },
        {
            "name": "Variable Offsets",
            "a": [0.1, 0.15, 0.2, 0.18, 0.25, 0.3],
            "b": [0.05, 0.08, 0.1, 0.07, 0.12, 0.15],
            "offsets": [-150, -50, 50, 150, 250, 450]
        },
        {
            "name": "Uniform Parameters",
            "a": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "b": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "offsets": [-100, 0, 100, 200, 300, 400]
        }
    ]
    
    # Fixed T slice for comparison
    t_slice_idx = len(t_range) // 2
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    for idx, params in enumerate(param_sets):
        comp = Compensate(params["a"], params["b"], params["offsets"], temperature=1)
        compensation = comp(x_range, y_range, t_range)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, Y, compensation[:, :, t_slice_idx], levels=15, cmap='coolwarm')
        ax.set_title(f'{params["name"]}\nRange: {compensation.min():.2f} to {compensation.max():.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
        
        print(f"{params['name']}: Compensation range {compensation.min():.2f} to {compensation.max():.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Simple 3D Multi-Window Compensation")
    print("=" * 50)
    
    # Example usage
    print("\nExample 1: Using standard parameters")
    a_params, b_params, boundary_offsets = create_standard_parameters()
    comp = Compensate(a_params, b_params, boundary_offsets)
    
    # Test computation
    x_range = np.linspace(0, 200, 40)
    y_range = np.linspace(0, 150, 30)
    t_range = np.linspace(-150, 450, 50)
    
    print(f"\nComputing compensation on {len(x_range)}×{len(y_range)}×{len(t_range)} grid...")
    compensation = comp(x_range, y_range, t_range)
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    
    # Visualize
    print("\nGenerating 3D visualization...")
    visualize_compensation_3d(comp, x_range, y_range, t_range)
    
    # Example 2: Custom parameters
    print("\n" + "=" * 50)
    print("Example 2: Custom parameters")
    custom_a = [0.1, 0.2, 0.15, 0.25, 0.3, 0.35]
    custom_b = [0.05, 0.1, 0.08, 0.12, 0.15, 0.18]
    custom_offsets = [-120, -20, 80, 180, 280, 380]
    
    comp2 = Compensate(custom_a, custom_b, custom_offsets, temperature=2)
    compensation2 = comp2(x_range, y_range, t_range)
    print(f"Custom compensation range: {compensation2.min():.3f} to {compensation2.max():.3f}")
    
    # Test different parameter sets
    print("\n" + "=" * 50)
    print("Testing different parameter sets")
    test_different_parameters()
    
    print("\nUsage: comp = Compensate(a_params, b_params, offsets)")
    print("       compensation = comp(x, y, t)")