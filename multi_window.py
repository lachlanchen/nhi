import numpy as np
import matplotlib.pyplot as plt
import torch

class Simple2DMultiWindow:
    def __init__(self, num_windows=4, bin_width=100, temperature=1):
        """
        Simple 2D multi-window compensation for visualization
        
        Args:
            num_windows: Number of windows (default 4)
            bin_width: Width of each time bin  
            temperature: Smoothness parameter for soft assignments
        """
        self.num_windows = num_windows
        self.bin_width = bin_width
        self.temperature = temperature
        
        # Define boundary line parameters [a_0, a_1, a_2, a_3, a_4] for 4 windows
        # These create the lines: t = a_i*x + offset_i
        self.a_params = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        
        print(f"Initialized {num_windows} windows")
        print(f"Boundary parameters: {self.a_params}")
    
    def get_boundary_lines(self, x):
        """
        Compute boundary line values: t = a_i * x + offset_i
        
        Returns:
            boundary_values: [num_windows + 1, len(x)] array
        """
        # Convert x to numpy array if it isn't already
        x = np.asarray(x)
        
        # Define offsets for each boundary line
        offsets = np.array([
            -self.bin_width,     # Line 0: t = a_0*x - bin_width
            0,                   # Line 1: t = a_1*x  
            self.bin_width,      # Line 2: t = a_2*x + bin_width
            2 * self.bin_width,  # Line 3: t = a_3*x + 2*bin_width
            3 * self.bin_width   # Line 4: t = a_4*x + 3*bin_width
        ])
        
        # Compute boundary values for each x
        if x.ndim == 0:  # Single value
            boundary_values = self.a_params * x + offsets
        else:  # Array of values
            boundary_values = np.zeros((len(self.a_params), len(x)))
            for i in range(len(self.a_params)):
                boundary_values[i] = self.a_params[i] * x + offsets[i]
        
        return boundary_values
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_window_memberships(self, x, t):
        """
        Compute soft membership for each window
        
        Returns:
            memberships: [num_windows, len(x), len(t)] array
        """
        boundary_values = self.get_boundary_lines(x)  # [5, len(x)]
        
        # Create meshgrids for broadcasting
        X, T = np.meshgrid(x, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_windows):
            # Lower boundary (line i)
            lower_bound = boundary_values[i][:, np.newaxis]  # [len(x), 1]
            # Upper boundary (line i+1)  
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len(x), 1]
            
            # Soft membership using sigmoid functions
            # σ((t - lower)/τ) * σ((upper - t)/τ)
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [num_windows, len(x), len(t)]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len(x), len(t)]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, t):
        """
        For each window, interpolate between boundary parameters
        
        Returns:
            interpolated_slopes: [num_windows, len(x), len(t)]
        """
        boundary_values = self.get_boundary_lines(x)  # [5, len(x)]
        
        # Create meshgrids
        X, T = np.meshgrid(x, t, indexing='ij')
        
        interpolated_slopes = []
        
        for i in range(self.num_windows):
            # Window i: between lines i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]      # [len(x), 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len(x), 1]
            
            # Interpolation parameter α ∈ [0,1]
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)  # Avoid division by zero
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to [0,1]
            
            # Interpolate between boundary parameters
            # slope = (1-α) * a_i + α * a_{i+1}
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
        memberships = self.compute_window_memberships(x, t)  # [num_windows, len(x), len(t)]
        
        # Get interpolated slopes for each window
        slopes = self.compute_within_window_interpolation(x, t)  # [num_windows, len(x), len(t)]
        
        # Create meshgrid for x
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Weighted sum of compensations
        # C = Σᵢ wᵢ × slope_i × x
        compensation = np.sum(memberships * slopes * X[np.newaxis, :, :], axis=0)
        
        return compensation

def visualize_simple_2d_problem():
    """
    Visualize the simple 2D multi-window compensation problem
    """
    # Create model
    model = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-150, 450, 200)
    
    print(f"X range: {x_range.min()} to {x_range.max()}")
    print(f"T range: {t_range.min()} to {t_range.max()}")
    
    # Test boundary lines computation
    print(f"Testing boundary computation...")
    test_x = np.array([50, 100, 150])
    test_boundaries = model.get_boundary_lines(test_x)
    print(f"Boundary values shape: {test_boundaries.shape}")
    print(f"Sample boundary values at x=[50,100,150]: \n{test_boundaries}")
    
    # Compute memberships and compensation
    print(f"Computing memberships...")
    memberships = model.compute_window_memberships(x_range, t_range)
    print(f"Memberships shape: {memberships.shape}")
    
    print(f"Computing compensation...")
    compensation = model.compute_compensation(x_range, t_range)
    print(f"Compensation shape: {compensation.shape}")
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Create meshgrids for plotting
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Plot window memberships (2x2 grid for 4 windows)
    for i in range(4):
        ax = plt.subplot(3, 3, i + 1)
        
        # Plot membership
        im = ax.contourf(X, T, memberships[i], levels=20, cmap='viridis')
        ax.set_title(f'Window {i} Membership')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add boundary lines for this window
        boundary_vals = model.get_boundary_lines(x_range)
        if i < len(boundary_vals) - 1:
            ax.plot(x_range, boundary_vals[i], 'r--', linewidth=2, alpha=0.8, label=f'Line {i}')
            ax.plot(x_range, boundary_vals[i + 1], 'r--', linewidth=2, alpha=0.8, label=f'Line {i+1}')
            ax.legend()
    
    # Plot all boundary lines together
    ax5 = plt.subplot(3, 3, 5)
    boundary_vals = model.get_boundary_lines(x_range)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, line_vals in enumerate(boundary_vals):
        ax5.plot(x_range, line_vals, '--', linewidth=2, color=colors[i], 
                label=f'Line {i}: t = {model.a_params[i]:.2f}x + {[-100, 0, 100, 200, 300][i]}')
    
    ax5.set_title('All Boundary Lines')
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Time t')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot total membership (should be 1 everywhere)
    ax6 = plt.subplot(3, 3, 6)
    total_membership = np.sum(memberships, axis=0)
    im6 = ax6.contourf(X, T, total_membership, levels=20, cmap='RdBu_r')
    ax6.set_title('Total Membership (should be ≈1)')
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Time t')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # Plot compensation function
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.contourf(X, T, compensation, levels=30, cmap='coolwarm')
    ax7.set_title('Compensation Function C(x,t)')
    ax7.set_xlabel('X coordinate')
    ax7.set_ylabel('Time t')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # Add boundary lines to compensation plot
    for i, line_vals in enumerate(boundary_vals):
        ax7.plot(x_range, line_vals, 'k--', linewidth=1, alpha=0.6)
    
    # Plot compensation along x=100 (vertical slice)
    ax8 = plt.subplot(3, 3, 8)
    x_idx = np.argmin(np.abs(x_range - 100))  # Find index closest to x=100
    compensation_slice = compensation[x_idx, :]
    ax8.plot(t_range, compensation_slice, 'b-', linewidth=2)
    ax8.set_title(f'Compensation at x = {x_range[x_idx]:.1f}')
    ax8.set_xlabel('Time t')
    ax8.set_ylabel('Compensation C(x,t)')
    ax8.grid(True, alpha=0.3)
    
    # Add vertical lines for boundary intersections at x=100
    boundary_at_x100 = model.get_boundary_lines(x_range[x_idx])  # Single value, returns 1D array
    for i, t_boundary in enumerate(boundary_at_x100):
        if t_range.min() <= t_boundary <= t_range.max():
            ax8.axvline(t_boundary, color=colors[i], linestyle='--', alpha=0.7, label=f'Boundary {i}')
    ax8.legend()
    
    # Plot compensation along t=200 (horizontal slice)
    ax9 = plt.subplot(3, 3, 9)
    t_idx = np.argmin(np.abs(t_range - 200))  # Find index closest to t=200
    compensation_slice = compensation[:, t_idx]
    ax9.plot(x_range, compensation_slice, 'r-', linewidth=2)
    ax9.set_title(f'Compensation at t = {t_range[t_idx]:.1f}')
    ax9.set_xlabel('X coordinate')
    ax9.set_ylabel('Compensation C(x,t)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    print(f"Total membership range: {total_membership.min():.6f} to {total_membership.max():.6f}")
    
    # Check that memberships sum to 1
    membership_error = np.abs(total_membership - 1.0).max()
    print(f"Maximum membership sum error: {membership_error:.6f}")
    
    return model, memberships, compensation

def visualize_parameter_sensitivity():
    """
    Show how changing parameters affects the compensation function
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-150, 450, 200)
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Different parameter settings
    param_sets = [
        {"a_params": [0.1, 0.2, 0.15, 0.25, 0.3], "title": "Original"},
        {"a_params": [0.2, 0.2, 0.2, 0.2, 0.2], "title": "Uniform slopes"},
        {"a_params": [0.0, 0.1, 0.2, 0.3, 0.4], "title": "Increasing slopes"},
        {"a_params": [0.4, 0.3, 0.2, 0.1, 0.0], "title": "Decreasing slopes"},
        {"a_params": [0.1, 0.4, 0.1, 0.4, 0.1], "title": "Alternating slopes"},
        {"a_params": [0.2, 0.15, 0.25, 0.1, 0.35], "title": "Random slopes"}
    ]
    
    for idx, param_set in enumerate(param_sets):
        model = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
        model.a_params = np.array(param_set["a_params"])
        
        compensation = model.compute_compensation(x_range, t_range)
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, T, compensation, levels=20, cmap='coolwarm')
        ax.set_title(param_set["title"])
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        
        # Add boundary lines
        boundary_vals = model.get_boundary_lines(x_range)
        for line_vals in boundary_vals:
            ax.plot(x_range, line_vals, 'k--', linewidth=1, alpha=0.5)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Simple 2D Multi-Window Compensation Visualization")
    print("=" * 60)
    
    # Quick test first
    print("Running quick test...")
    model = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    
    # Test with small arrays first
    x_test = np.array([50, 100, 150])
    t_test = np.array([0, 100, 200])
    
    print(f"Test x values: {x_test}")
    print(f"Test t values: {t_test}")
    
    # Test boundary computation
    boundaries = model.get_boundary_lines(x_test)
    print(f"Boundary shape: {boundaries.shape}")
    print(f"Boundaries:\n{boundaries}")
    
    # Test single value
    single_x = 100.0
    single_boundaries = model.get_boundary_lines(single_x)
    print(f"Single boundary shape: {single_boundaries.shape}")
    print(f"Single boundaries: {single_boundaries}")
    
    # Test membership computation
    memberships_test = model.compute_window_memberships(x_test, t_test)
    print(f"Memberships shape: {memberships_test.shape}")
    print(f"Memberships sum: {np.sum(memberships_test, axis=0)}")
    
    # Test compensation
    compensation_test = model.compute_compensation(x_test, t_test)
    print(f"Compensation shape: {compensation_test.shape}")
    print(f"Compensation values:\n{compensation_test}")
    
    print("\nQuick test passed! Running full visualization...")
    
    # Main visualization
    model, memberships, compensation = visualize_simple_2d_problem()
    
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # Parameter sensitivity
    visualize_parameter_sensitivity()
    
    print("\nVisualization complete!")