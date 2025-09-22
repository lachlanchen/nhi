import numpy as np
import matplotlib.pyplot as plt

class Simple2DMultiWindow:
    def __init__(self, num_main_windows=3, data_range=300, extend_range=100, temperature=1):
        """
        Simple 2D multi-window compensation for visualization
        
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
        
        # Total parameters: n+3 (for n+3 boundary lines)
        self.num_params = num_main_windows + 3
        
        # Initialize parameters - evenly spaced slopes as example
        self.a_params = np.linspace(0.05, 0.35, self.num_params)
        
        # Calculate boundary offsets
        # We want to cover from -extend_range to data_range + extend_range
        # with total_windows windows
        window_width = self.total_range / self.total_windows
        self.boundary_offsets = []
        
        for i in range(self.num_params):
            # Boundary i is at position: -extend_range + i * window_width
            offset = -extend_range + i * window_width
            self.boundary_offsets.append(offset)
        
        self.boundary_offsets = np.array(self.boundary_offsets)
        
        print(f"Configuration:")
        print(f"  Main windows: {self.num_main_windows}")
        print(f"  Total windows: {self.total_windows} (including 2 edge windows)")
        print(f"  Parameters: {self.num_params}")
        print(f"  Data range: [0, {self.data_range}]")
        print(f"  Extended range: [{-self.extend_range}, {self.data_range + self.extend_range}]")
        print(f"  Window width: {window_width:.1f}")
        print(f"  Boundary offsets: {self.boundary_offsets}")
        print(f"  Parameters: {self.a_params}")
    
    def get_boundary_lines(self, x):
        """
        Compute boundary line values: t = a_i * x + offset_i
        
        Returns:
            boundary_values: [num_params, len(x)] array
        """
        # Convert x to numpy array if it isn't already
        x = np.asarray(x)
        
        # Compute boundary values for each x
        if x.ndim == 0:  # Single value
            boundary_values = self.a_params * x + self.boundary_offsets
        else:  # Array of values
            boundary_values = np.zeros((self.num_params, len(x)))
            for i in range(self.num_params):
                boundary_values[i] = self.a_params[i] * x + self.boundary_offsets[i]
        
        return boundary_values
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_window_memberships(self, x, t):
        """
        Compute soft membership for each window
        
        Returns:
            memberships: [total_windows, len(x), len(t)] array
        """
        boundary_values = self.get_boundary_lines(x)  # [num_params, len(x)]
        
        # Create meshgrids for broadcasting
        X, T = np.meshgrid(x, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.total_windows):
            # Window i: between boundary lines i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]  # [len(x), 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len(x), 1]
            
            # Soft membership using sigmoid functions
            # σ((t - lower)/τ) * σ((upper - t)/τ)
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [total_windows, len(x), len(t)]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len(x), len(t)]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, t):
        """
        For each window, interpolate between boundary parameters
        
        Returns:
            interpolated_slopes: [total_windows, len(x), len(t)]
        """
        boundary_values = self.get_boundary_lines(x)  # [num_params, len(x)]
        
        # Create meshgrids
        X, T = np.meshgrid(x, t, indexing='ij')
        
        interpolated_slopes = []
        
        for i in range(self.total_windows):
            # Window i: between boundary lines i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]      # [len(x), 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len(x), 1]
            
            # Interpolation parameter α ∈ [0,1]
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)  # Avoid division by zero
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to [0,1]
            
            # Interpolate between parameters a_i and a_{i+1}
            slope = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            
            interpolated_slopes.append(slope)
        
        return np.array(interpolated_slopes)
    
    def compute_compensation(self, x, t):
        """
        Compute the final compensation for given (x, t) coordinates
        
        Returns:
            compensation: [len(x), len(t)] array
        """
        # Get window memberships
        memberships = self.compute_window_memberships(x, t)  # [total_windows, len(x), len(t)]
        
        # Get interpolated slopes for each window
        slopes = self.compute_within_window_interpolation(x, t)  # [total_windows, len(x), len(t)]
        
        # Create meshgrid for x
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Weighted sum of compensations
        # C = Σᵢ wᵢ × slope_i × x
        compensation = np.sum(memberships * slopes * X[np.newaxis, :, :], axis=0)
        
        return compensation
    
    def get_main_window_memberships(self, x, t):
        """
        Get memberships for just the main windows (excluding edge windows)
        
        Returns:
            main_memberships: [num_main_windows, len(x), len(t)] array
        """
        all_memberships = self.compute_window_memberships(x, t)
        # Edge window 0, then main windows 1 to num_main_windows, then edge window num_main_windows+1
        main_memberships = all_memberships[1:1+self.num_main_windows]
        return main_memberships

def visualize_simple_2d_problem():
    """
    Visualize the simple 2D multi-window compensation problem
    """
    # Create model: 3 main windows covering 0-300, extended to -100 to 400
    model = Simple2DMultiWindow(num_main_windows=3, data_range=300, extend_range=100, temperature=1)
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 100)  # X coordinates
    t_range = np.linspace(-150, 450, 200)  # Time coordinates (slightly beyond extended range)
    
    print(f"\nVisualization setup:")
    print(f"X range: {x_range.min()} to {x_range.max()}")
    print(f"T range: {t_range.min()} to {t_range.max()}")
    
    # Test boundary lines computation
    print(f"\nTesting boundary computation...")
    test_x = np.array([50, 100, 150])
    test_boundaries = model.get_boundary_lines(test_x)
    print(f"Boundary values shape: {test_boundaries.shape}")
    print(f"Sample boundary values at x=[50,100,150]: \n{test_boundaries}")
    
    # Show boundary equations
    print(f"\nBoundary line equations:")
    for i in range(len(model.a_params)):
        print(f"  Line {i}: t = {model.a_params[i]:.3f}x + {model.boundary_offsets[i]:.1f}")
    
    # Compute memberships and compensation
    print(f"\nComputing memberships and compensation...")
    all_memberships = model.compute_window_memberships(x_range, t_range)
    main_memberships = model.get_main_window_memberships(x_range, t_range)
    compensation = model.compute_compensation(x_range, t_range)
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Create meshgrids for plotting
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Plot main window memberships 
    for i in range(model.num_main_windows):
        ax = plt.subplot(3, 3, i + 1)
        
        # Plot membership for main window i
        im = ax.contourf(X, T, main_memberships[i], levels=20, cmap='viridis')
        ax.set_title(f'Main Window {i} Membership')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add boundary lines for this main window
        boundary_vals = model.get_boundary_lines(x_range)
        # Main window i is between boundary lines i+1 and i+2
        ax.plot(x_range, boundary_vals[i+1], 'r--', linewidth=2, alpha=0.8, label=f'Line {i+1}')
        ax.plot(x_range, boundary_vals[i+2], 'r--', linewidth=2, alpha=0.8, label=f'Line {i+2}')
        ax.legend()
    
    # Plot all boundary lines together
    ax4 = plt.subplot(3, 3, 4)
    boundary_vals = model.get_boundary_lines(x_range)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, line_vals in enumerate(boundary_vals):
        color = colors[i % len(colors)]
        ax4.plot(x_range, line_vals, '--', linewidth=2, color=color, 
                label=f'Line {i}: t = {model.a_params[i]:.3f}x + {model.boundary_offsets[i]:.1f}')
    
    ax4.set_title('All Boundary Lines')
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Time t')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Plot total membership (should be 1 everywhere)
    ax5 = plt.subplot(3, 3, 5)
    total_membership = np.sum(all_memberships, axis=0)
    im5 = ax5.contourf(X, T, total_membership, levels=20, cmap='RdBu_r')
    ax5.set_title('Total Membership (should be ≈1)')
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Time t')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # Plot compensation function
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.contourf(X, T, compensation, levels=30, cmap='coolwarm')
    ax6.set_title('Compensation Function C(x,t)')
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Time t')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # Add boundary lines to compensation plot
    for i, line_vals in enumerate(boundary_vals):
        color = colors[i % len(colors)]
        ax6.plot(x_range, line_vals, '--', linewidth=1, alpha=0.6, color=color)
    
    # Plot compensation along x=100 (vertical slice)
    ax7 = plt.subplot(3, 3, 7)
    x_idx = np.argmin(np.abs(x_range - 100))  # Find index closest to x=100
    compensation_slice = compensation[x_idx, :]
    ax7.plot(t_range, compensation_slice, 'b-', linewidth=2)
    ax7.set_title(f'Compensation at x = {x_range[x_idx]:.1f}')
    ax7.set_xlabel('Time t')
    ax7.set_ylabel('Compensation C(x,t)')
    ax7.grid(True, alpha=0.3)
    
    # Add vertical lines for boundary intersections at x=100
    boundary_at_x100 = model.get_boundary_lines(x_range[x_idx])
    for i, t_boundary in enumerate(boundary_at_x100):
        if t_range.min() <= t_boundary <= t_range.max():
            color = colors[i % len(colors)]
            ax7.axvline(t_boundary, color=color, linestyle='--', alpha=0.7, label=f'B{i}')
    ax7.legend()
    
    # Plot compensation along t=150 (horizontal slice)
    ax8 = plt.subplot(3, 3, 8)
    t_idx = np.argmin(np.abs(t_range - 150))  # Find index closest to t=150
    compensation_slice = compensation[:, t_idx]
    ax8.plot(x_range, compensation_slice, 'r-', linewidth=2)
    ax8.set_title(f'Compensation at t = {t_range[t_idx]:.1f}')
    ax8.set_xlabel('X coordinate')
    ax8.set_ylabel('Compensation C(x,t)')
    ax8.grid(True, alpha=0.3)
    
    # Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = f"""Configuration:
Main windows: {model.num_main_windows}
Total windows: {model.total_windows}
Parameters: {model.num_params}

Data range: [0, {model.data_range}]
Extended: [{-model.extend_range}, {model.data_range + model.extend_range}]

Compensation range: 
{compensation.min():.2f} to {compensation.max():.2f}

Membership sum error: 
{np.abs(total_membership - 1.0).max():.6f}"""
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return model, all_memberships, main_memberships, compensation

def test_different_configurations():
    """
    Test different window configurations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-150, 450, 200)
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    configs = [
        {"num_main_windows": 2, "data_range": 300, "extend_range": 100, "title": "2 Main Windows"},
        {"num_main_windows": 3, "data_range": 300, "extend_range": 100, "title": "3 Main Windows"},
        {"num_main_windows": 4, "data_range": 300, "extend_range": 100, "title": "4 Main Windows"},
        {"num_main_windows": 3, "data_range": 400, "extend_range": 50, "title": "Extended Data Range"},
        {"num_main_windows": 3, "data_range": 200, "extend_range": 150, "title": "Extended Edge Range"},
        {"num_main_windows": 5, "data_range": 300, "extend_range": 100, "title": "5 Main Windows"}
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, config in enumerate(configs):
        model = Simple2DMultiWindow(**{k:v for k,v in config.items() if k != 'title'}, temperature=1)
        compensation = model.compute_compensation(x_range, t_range)
        boundary_vals = model.get_boundary_lines(x_range)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, T, compensation, levels=20, cmap='coolwarm')
        ax.set_title(f'{config["title"]}\n({model.num_params} params, {model.total_windows} windows)')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        
        # Add boundary lines
        for i, line_vals in enumerate(boundary_vals):
            color = colors[i % len(colors)]
            ax.plot(x_range, line_vals, '--', linewidth=1, alpha=0.7, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def test_negative_parameters():
    """
    Test with negative parameters
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-150, 450, 200)
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Test cases
    test_cases = [
        {"params": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "title": "Positive Parameters"},
        {"params": [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05], "title": "Negative Parameters"}, 
        {"params": [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15], "title": "Mixed Parameters"}
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for idx, case in enumerate(test_cases):
        model = Simple2DMultiWindow(num_main_windows=3, data_range=300, extend_range=100, temperature=1)
        model.a_params = np.array(case["params"])
        
        compensation = model.compute_compensation(x_range, t_range)
        boundary_vals = model.get_boundary_lines(x_range)
        
        ax = axes[idx]
        im = ax.contourf(X, T, compensation, levels=20, cmap='coolwarm')
        ax.set_title(f'{case["title"]}\n{case["params"]}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        
        # Add boundary lines
        for i, line_vals in enumerate(boundary_vals):
            color = colors[i % len(colors)]
            ax.plot(x_range, line_vals, '--', linewidth=1, alpha=0.7, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        print(f"{case['title']}: Compensation range {compensation.min():.2f} to {compensation.max():.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("General Multi-Window Compensation Visualization")
    print("=" * 60)
    
    # Main visualization with clear example
    print("Example: 3 main windows, data range 0-300, extended to -100 to 400")
    model, all_memberships, main_memberships, compensation = visualize_simple_2d_problem()
    
    print("\n" + "=" * 60)
    print("Testing Different Configurations")
    print("=" * 60)
    
    test_different_configurations()
    
    print("\n" + "=" * 60)
    print("Testing Negative Parameters")
    print("=" * 60)
    
    test_negative_parameters()
    
    print("\nVisualization complete!")