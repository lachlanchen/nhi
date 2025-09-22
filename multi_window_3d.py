import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Simple3DMultiWindow:
    def __init__(self, num_main_windows=3, data_range=300, extend_range=100, temperature=1):
        """
        Simple 3D multi-window compensation for visualization
        
        Args:
            num_main_windows: Number of main windows (default 3)
            data_range: Main data range width (default 300)
            extend_range: Extension beyond main range for edge windows (default 100)
            temperature: Smoothness parameter for soft assignments (default 1)
        """
        self.num_main_windows = num_main_windows
        self.data_range = data_range
        self.extend_range = extend_range
        self.temperature = temperature
        
        # Total coverage: from -extend_range to data_range + extend_range
        self.total_range = data_range + 2 * extend_range
        
        # Total windows: n main windows + 2 edge windows  
        self.total_windows = num_main_windows + 2
        
        # Total parameters: 2 * (n+3) for boundary surfaces t = a_i * x + b_i * y + offset_i
        self.num_params = num_main_windows + 3
        
        # Initialize x-direction parameters (a_i)
        self.a_params = np.linspace(0.05, 0.35, self.num_params)
        
        # Initialize y-direction parameters (b_i)  
        self.b_params = np.linspace(0.02, 0.18, self.num_params)
        
        # Calculate boundary offsets (same as 2D case)
        window_width = self.total_range / self.total_windows
        self.boundary_offsets = []
        
        for i in range(self.num_params):
            offset = -extend_range + i * window_width
            self.boundary_offsets.append(offset)
        
        self.boundary_offsets = np.array(self.boundary_offsets)
        
        print(f"3D Configuration:")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.total_windows} (including 2 edge windows)")
        print(f"  Boundary surfaces: {self.num_params}")
        print(f"  Total parameters: {2 * self.num_params} (a + b coefficients)")
        print(f"  Data range: [0, {self.data_range}]")
        print(f"  Extended range: [{-self.extend_range}, {self.data_range + self.extend_range}]")
        print(f"  Window width: {window_width:.1f}")
        print(f"  Boundary offsets: {self.boundary_offsets}")
        print(f"  A parameters (x-coeff): {self.a_params}")
        print(f"  B parameters (y-coeff): {self.b_params}")
    
    def get_boundary_surfaces(self, x, y):
        """
        Compute boundary surface values: t = a_i * x + b_i * y + offset_i
        
        Args:
            x: X coordinates [len_x] or scalar
            y: Y coordinates [len_y] or scalar
            
        Returns:
            boundary_values: [num_params, len_x, len_y] array or [num_params] for scalars
        """
        # Convert to numpy arrays if needed
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle different input shapes
        if x.ndim == 0 and y.ndim == 0:  # Both scalars
            boundary_values = self.a_params * x + self.b_params * y + self.boundary_offsets
        elif x.ndim == 1 and y.ndim == 1:  # Both arrays - create meshgrid
            X, Y = np.meshgrid(x, y, indexing='ij')
            boundary_values = np.zeros((self.num_params, len(x), len(y)))
            for i in range(self.num_params):
                boundary_values[i] = self.a_params[i] * X + self.b_params[i] * Y + self.boundary_offsets[i]
        elif x.ndim == 2 and y.ndim == 2:  # Already meshgrid format
            boundary_values = np.zeros((self.num_params,) + x.shape)
            for i in range(self.num_params):
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
            memberships: [total_windows, len_x, len_y, len_t] array
        """
        # Get boundary surfaces for this x, y grid
        boundary_values = self.get_boundary_surfaces(x, y)  # [num_params, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.total_windows):
            # Window i: between boundary surfaces i and i+1
            lower_bound = boundary_values[i][:, :, np.newaxis]  # [len_x, len_y, 1]
            upper_bound = boundary_values[i + 1][:, :, np.newaxis]  # [len_x, len_y, 1]
            
            # Soft membership using sigmoid functions
            # σ((t - lower)/τ) * σ((upper - t)/τ)
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [total_windows, len_x, len_y, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len_x, len_y, len_t]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, y, t):
        """
        For each window, interpolate between boundary parameters in 3D
        
        Returns:
            interpolated_slopes_a: [total_windows, len_x, len_y, len_t] (x-direction slopes)
            interpolated_slopes_b: [total_windows, len_x, len_y, len_t] (y-direction slopes)
        """
        boundary_values = self.get_boundary_surfaces(x, y)  # [num_params, len_x, len_y]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        interpolated_slopes_a = []
        interpolated_slopes_b = []
        
        for i in range(self.total_windows):
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
    
    def compute_compensation(self, x, y, t):
        """
        Compute the final compensation for given (x, y, t) coordinates
        
        Returns:
            compensation: [len_x, len_y, len_t] array
        """
        # Get window memberships
        memberships = self.compute_window_memberships(x, y, t)  # [total_windows, len_x, len_y, len_t]
        
        # Get interpolated slopes for each window
        slopes_a, slopes_b = self.compute_within_window_interpolation(x, y, t)  # [total_windows, len_x, len_y, len_t]
        
        # Create 3D meshgrids
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        # Weighted sum of compensations
        # C = Σᵢ wᵢ × (slope_a_i × x + slope_b_i × y)
        compensation_x = np.sum(memberships * slopes_a * X[np.newaxis, :, :, :], axis=0)
        compensation_y = np.sum(memberships * slopes_b * Y[np.newaxis, :, :, :], axis=0)
        compensation = compensation_x + compensation_y
        
        return compensation, compensation_x, compensation_y
    
    def get_main_window_memberships(self, x, y, t):
        """
        Get memberships for just the main windows (excluding edge windows)
        
        Returns:
            main_memberships: [num_main_windows, len_x, len_y, len_t] array
        """
        all_memberships = self.compute_window_memberships(x, y, t)
        # Edge window 0, then main windows 1 to num_main_windows, then edge window num_main_windows+1
        main_memberships = all_memberships[1:1+self.num_main_windows]
        return main_memberships

def visualize_3d_boundary_surfaces(model, x_range, y_range):
    """
    Visualize the 3D boundary surfaces
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create meshgrids for surfaces
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # Get boundary surfaces
    boundary_surfaces = model.get_boundary_surfaces(x_range, y_range)
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

def visualize_simple_3d_problem():
    """
    Visualize the simple 3D multi-window compensation problem
    """
    # Create 3D model
    model = Simple3DMultiWindow(num_main_windows=3, data_range=300, extend_range=100, temperature=1)
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 50)   # Reduced resolution for 3D
    y_range = np.linspace(0, 150, 40)   # Reduced resolution for 3D  
    t_range = np.linspace(-150, 450, 60) # Reduced resolution for 3D
    
    print(f"\n3D Visualization setup:")
    print(f"X range: {x_range.min()} to {x_range.max()} ({len(x_range)} points)")
    print(f"Y range: {y_range.min()} to {y_range.max()} ({len(y_range)} points)")
    print(f"T range: {t_range.min()} to {t_range.max()} ({len(t_range)} points)")
    
    # First, visualize the boundary surfaces
    print(f"\nVisualizing boundary surfaces...")
    visualize_3d_boundary_surfaces(model, x_range, y_range)
    
    # Compute memberships and compensation
    print(f"\nComputing 3D memberships and compensation...")
    all_memberships = model.compute_window_memberships(x_range, y_range, t_range)
    main_memberships = model.get_main_window_memberships(x_range, y_range, t_range)
    compensation, comp_x, comp_y = model.compute_compensation(x_range, y_range, t_range)
    
    print(f"All memberships shape: {all_memberships.shape}")
    print(f"Main memberships shape: {main_memberships.shape}")
    print(f"Compensation shape: {compensation.shape}")
    
    # Visualize compensation slices
    fig = plt.figure(figsize=(20, 15))
    
    # Select middle indices for slicing
    x_mid = len(x_range) // 2
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    
    # Compensation at fixed T (XY slice)
    ax1 = plt.subplot(3, 4, 1)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    im1 = ax1.contourf(X, Y, compensation[:, :, t_mid], levels=20, cmap='coolwarm')
    ax1.set_title(f'Compensation at T = {t_range[t_mid]:.1f}')
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
    total_membership = np.sum(all_memberships, axis=0)
    im4 = ax4.contourf(X, Y, total_membership[:, :, t_mid], levels=20, cmap='RdBu_r')
    ax4.set_title(f'Total Membership at T = {t_range[t_mid]:.1f}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # Plot main window memberships at fixed T
    for i in range(min(3, model.num_main_windows)):
        ax = plt.subplot(3, 4, 5 + i)
        im = ax.contourf(X, Y, main_memberships[i, :, :, t_mid], levels=20, cmap='viridis')
        ax.set_title(f'Window {i} at T = {t_range[t_mid]:.1f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Compensation slices at fixed X and Y
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(t_range, compensation[x_mid, y_mid, :], 'b-', linewidth=2, label='Total')
    ax8.plot(t_range, comp_x[x_mid, y_mid, :], 'r--', linewidth=2, label='X-component')
    ax8.plot(t_range, comp_y[x_mid, y_mid, :], 'g--', linewidth=2, label='Y-component')
    ax8.set_title(f'Compensation at X={x_range[x_mid]:.1f}, Y={y_range[y_mid]:.1f}')
    ax8.set_xlabel('Time T')
    ax8.set_ylabel('Compensation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add boundary surface intersections
    boundary_at_point = model.get_boundary_surfaces(x_range[x_mid], y_range[y_mid])
    colors_lines = plt.cm.tab10(np.linspace(0, 1, len(boundary_at_point)))
    for i, t_boundary in enumerate(boundary_at_point):
        if t_range.min() <= t_boundary <= t_range.max():
            ax8.axvline(t_boundary, color=colors_lines[i], linestyle='--', alpha=0.7, label=f'Surface {i}')
    
    # Compensation at fixed Y (XT slice)
    ax9 = plt.subplot(3, 4, 9)
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    im9 = ax9.contourf(X_2d, T_2d, compensation[:, y_mid, :], levels=20, cmap='coolwarm')
    ax9.set_title(f'Compensation at Y = {y_range[y_mid]:.1f}')
    ax9.set_xlabel('X')
    ax9.set_ylabel('T')
    plt.colorbar(im9, ax=ax9, shrink=0.8)
    
    # Compensation at fixed X (YT slice)
    ax10 = plt.subplot(3, 4, 10)
    Y_2d, T_2d = np.meshgrid(y_range, t_range, indexing='ij')
    im10 = ax10.contourf(Y_2d, T_2d, compensation[x_mid, :, :], levels=20, cmap='coolwarm')
    ax10.set_title(f'Compensation at X = {x_range[x_mid]:.1f}')
    ax10.set_xlabel('Y')
    ax10.set_ylabel('T')
    plt.colorbar(im10, ax=ax10, shrink=0.8)
    
    # Statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    stats_text = f"""3D Configuration:
Main windows: {model.num_main_windows}
Total windows: {model.total_windows}
Boundary surfaces: {model.num_params}
Total parameters: {2 * model.num_params}

Grid size: {len(x_range)} × {len(y_range)} × {len(t_range)}

Compensation range:
Total: {compensation.min():.2f} to {compensation.max():.2f}
X-comp: {comp_x.min():.2f} to {comp_x.max():.2f}  
Y-comp: {comp_y.min():.2f} to {comp_y.max():.2f}

Membership sum error: 
{np.abs(total_membership - 1.0).max():.6f}"""
    
    ax11.text(0.1, 0.9, stats_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return model, all_memberships, main_memberships, compensation, comp_x, comp_y

def test_3d_configurations():
    """
    Test different 3D configurations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    x_range = np.linspace(0, 200, 30)
    y_range = np.linspace(0, 150, 25)
    t_range = np.linspace(-150, 450, 40)
    
    configs = [
        {"num_main_windows": 2, "title": "2 Main Windows"},
        {"num_main_windows": 3, "title": "3 Main Windows"},
        {"num_main_windows": 4, "title": "4 Main Windows"},
        {"num_main_windows": 2, "data_range": 400, "title": "Extended Data Range"},
        {"num_main_windows": 3, "extend_range": 50, "title": "Smaller Extension"}, 
        {"num_main_windows": 5, "title": "5 Main Windows"}
    ]
    
    # Fixed T slice for visualization
    t_slice_idx = len(t_range) // 2
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    for idx, config in enumerate(configs):
        # Set defaults
        params = {"data_range": 300, "extend_range": 100, "temperature": 1}
        params.update({k:v for k,v in config.items() if k != 'title'})
        
        model = Simple3DMultiWindow(**params)
        compensation, _, _ = model.compute_compensation(x_range, y_range, t_range)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, Y, compensation[:, :, t_slice_idx], levels=15, cmap='coolwarm')
        ax.set_title(f'{config["title"]}\n({2*model.num_params} params, {model.total_windows} windows)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def compare_2d_vs_3d():
    """
    Compare 2D (y=0) vs full 3D compensation
    """
    # Import 2D model for comparison
    from multi_window_2 import Simple2DMultiWindow  # Assuming the 2D code is in this file
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    x_range = np.linspace(0, 200, 100)
    y_range = np.linspace(0, 150, 80)
    t_range = np.linspace(-150, 450, 200)
    
    # 2D model
    model_2d = Simple2DMultiWindow(num_main_windows=3, data_range=300, extend_range=100, temperature=1)
    compensation_2d = model_2d.compute_compensation(x_range, t_range)
    
    # 3D model
    model_3d = Simple3DMultiWindow(num_main_windows=3, data_range=300, extend_range=100, temperature=1)
    compensation_3d, comp_x, comp_y = model_3d.compute_compensation(x_range, y_range, t_range)
    
    # Select slices for comparison
    y_mid = len(y_range) // 2
    t_mid = len(t_range) // 2
    x_mid = len(x_range) // 2
    
    X_2d, T_2d = np.meshgrid(x_range, t_range, indexing='ij')
    X_xy, Y_xy = np.meshgrid(x_range, y_range, indexing='ij')
    
    # 2D compensation (XT)
    im1 = axes[0, 0].contourf(X_2d, T_2d, compensation_2d, levels=20, cmap='coolwarm')
    axes[0, 0].set_title('2D Compensation (XT plane)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('T')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 3D compensation at Y=mid (XT slice)
    im2 = axes[0, 1].contourf(X_2d, T_2d, compensation_3d[:, y_mid, :], levels=20, cmap='coolwarm')
    axes[0, 1].set_title(f'3D Compensation at Y={y_range[y_mid]:.1f}')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('T')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 3D compensation at Y=0 (XT slice)
    im3 = axes[0, 2].contourf(X_2d, T_2d, compensation_3d[:, 0, :], levels=20, cmap='coolwarm')
    axes[0, 2].set_title('3D Compensation at Y=0')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('T')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # 3D compensation components at fixed T
    im4 = axes[1, 0].contourf(X_xy, Y_xy, comp_x[:, :, t_mid], levels=20, cmap='coolwarm')
    axes[1, 0].set_title(f'X-Component at T={t_range[t_mid]:.1f}')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    im5 = axes[1, 1].contourf(X_xy, Y_xy, comp_y[:, :, t_mid], levels=20, cmap='coolwarm')
    axes[1, 1].set_title(f'Y-Component at T={t_range[t_mid]:.1f}')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    im6 = axes[1, 2].contourf(X_xy, Y_xy, compensation_3d[:, :, t_mid], levels=20, cmap='coolwarm')
    axes[1, 2].set_title(f'Total 3D at T={t_range[t_mid]:.1f}')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    print(f"\nComparison Statistics:")
    print(f"2D compensation range: {compensation_2d.min():.2f} to {compensation_2d.max():.2f}")
    print(f"3D total compensation range: {compensation_3d.min():.2f} to {compensation_3d.max():.2f}")
    print(f"3D X-component range: {comp_x.min():.2f} to {comp_x.max():.2f}")
    print(f"3D Y-component range: {comp_y.min():.2f} to {comp_y.max():.2f}")

if __name__ == "__main__":
    print("3D Multi-Window Compensation Visualization")
    print("=" * 60)
    
    # Main 3D visualization
    print("Example: 3 main windows in 3D with boundary surfaces")
    model, all_memberships, main_memberships, compensation, comp_x, comp_y = visualize_simple_3d_problem()
    
    print("\n" + "=" * 60)
    print("Testing Different 3D Configurations")
    print("=" * 60)
    
    test_3d_configurations()
    
    print("\n" + "=" * 60) 
    print("Comparing 2D vs 3D")
    print("=" * 60)
    
    try:
        compare_2d_vs_3d()
    except ImportError:
        print("Skipping 2D vs 3D comparison (2D module not available)")
    
    print("\n3D Visualization complete!")