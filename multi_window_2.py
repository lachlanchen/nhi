import numpy as np
import matplotlib.pyplot as plt

class Simple2DMultiWindow:
    def __init__(self, num_windows=4, bin_width=100, temperature=1):
        """
        Simple 2D multi-window compensation for visualization
        
        Args:
            num_windows: Number of main windows (default 4)
            bin_width: Width of each time bin  
            temperature: Smoothness parameter for soft assignments (default 1)
        """
        self.num_windows = num_windows
        self.bin_width = bin_width
        self.temperature = temperature
        
        # For n main windows, we need 2 extra edge windows, so n+2 total windows
        # This requires n+3 boundary lines, which needs n+3 parameters
        self.total_windows = num_windows + 2  # Including edge windows
        num_params = num_windows + 3
        
        # Define boundary line parameters [a_0, a_1, ..., a_{n+2}] 
        # Example for 4 main windows: [a_0, a_1, a_2, a_3, a_4, a_5, a_6] (7 parameters)
        self.a_params = np.array([0.05, 0.1, 0.2, 0.15, 0.25, 0.3, 0.35])[:num_params]
        
        print(f"Initialized {num_windows} main windows + 2 edge windows = {self.total_windows} total windows")
        print(f"Using {num_params} parameters: {self.a_params}")
    
    def get_boundary_lines(self, x):
        """
        Compute boundary line values: t = a_i * x + offset_i
        For n main windows, creates n+3 boundary lines using n+3 parameters
        
        Returns:
            boundary_values: [num_windows + 3, len(x)] array
        """
        # Convert x to numpy array if it isn't already
        x = np.asarray(x)
        
        # Define offsets for each boundary line
        # For n main windows + 2 edge windows, we create n+3 boundary lines
        offsets = []
        for i in range(self.num_windows + 3):
            offset = (i - 2) * self.bin_width  # Start from -2*bin_width for edge handling
            offsets.append(offset)
        offsets = np.array(offsets)
        
        # Use all n+3 parameters for the n+3 boundary lines
        boundary_params = self.a_params
        
        # Compute boundary values for each x
        if x.ndim == 0:  # Single value
            boundary_values = boundary_params * x + offsets
        else:  # Array of values
            boundary_values = np.zeros((len(boundary_params), len(x)))
            for i in range(len(boundary_params)):
                boundary_values[i] = boundary_params[i] * x + offsets[i]
        
        return boundary_values
    
    def sigmoid(self, z):
        """Stable sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_window_memberships(self, x, t):
        """
        Compute soft membership for each window (including edge windows)
        
        Returns:
            memberships: [total_windows, len(x), len(t)] array
        """
        boundary_values = self.get_boundary_lines(x)  # [num_windows+3, len(x)]
        
        # Create meshgrids for broadcasting
        X, T = np.meshgrid(x, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.total_windows):
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
        
        memberships = np.array(memberships)  # [total_windows, len(x), len(t)]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len(x), len(t)]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, t):
        """
        For each window, interpolate between boundary parameters
        Uses the full n+3 parameter set for interpolation
        
        Returns:
            interpolated_slopes: [total_windows, len(x), len(t)]
        """
        boundary_values = self.get_boundary_lines(x)  # [num_windows+3, len(x)]
        
        # Create meshgrids
        X, T = np.meshgrid(x, t, indexing='ij')
        
        interpolated_slopes = []
        
        for i in range(self.total_windows):
            # Window i: between lines i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]      # [len(x), 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len(x), 1]
            
            # Interpolation parameter α ∈ [0,1]
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)  # Avoid division by zero
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to [0,1]
            
            # Interpolate between the corresponding parameters
            # For window i, use parameters a_i and a_{i+1}
            param_idx_1 = i
            param_idx_2 = min(i + 1, len(self.a_params) - 1)  # Don't exceed array bounds
            
            slope = (1 - alpha) * self.a_params[param_idx_1] + alpha * self.a_params[param_idx_2]
            
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
            main_memberships: [num_windows, len(x), len(t)] array
        """
        all_memberships = self.compute_window_memberships(x, t)
        # Skip first edge window, take num_windows main windows
        main_memberships = all_memberships[1:1+self.num_windows]
        return main_memberships

def visualize_simple_2d_problem():
    """
    Visualize the simple 2D multi-window compensation problem
    """
    # Create model with 4 main windows (requires 7 parameters, creates 6 total windows)
    model = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-300, 600, 200)  # Extended range to see edge effects
    
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
    all_memberships = model.compute_window_memberships(x_range, t_range)
    main_memberships = model.get_main_window_memberships(x_range, t_range)
    print(f"All memberships shape: {all_memberships.shape}")
    print(f"Main memberships shape: {main_memberships.shape}")
    
    print(f"Computing compensation...")
    compensation = model.compute_compensation(x_range, t_range)
    print(f"Compensation shape: {compensation.shape}")
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Create meshgrids for plotting
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Plot main window memberships (2x2 grid for 4 main windows)
    for i in range(4):
        ax = plt.subplot(3, 3, i + 1)
        
        # Plot membership for main window i
        im = ax.contourf(X, T, main_memberships[i], levels=20, cmap='viridis')
        ax.set_title(f'Main Window {i} Membership')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add boundary lines for this main window (i+1 and i+2 from all boundaries)
        boundary_vals = model.get_boundary_lines(x_range)
        if i+1 < len(boundary_vals) - 1:
            ax.plot(x_range, boundary_vals[i+1], 'r--', linewidth=2, alpha=0.8, label=f'Line {i+1}')
            ax.plot(x_range, boundary_vals[i+2], 'r--', linewidth=2, alpha=0.8, label=f'Line {i+2}')
            ax.legend()
    
    # Plot all boundary lines together
    ax5 = plt.subplot(3, 3, 5)
    boundary_vals = model.get_boundary_lines(x_range)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Calculate offsets for display
    offsets = [(i - 2) * model.bin_width for i in range(len(model.a_params))]
    
    for i, line_vals in enumerate(boundary_vals):
        color = colors[i % len(colors)]
        ax5.plot(x_range, line_vals, '--', linewidth=2, color=color, 
                label=f'Line {i}: t = {model.a_params[i]:.2f}x + {offsets[i]}')
    
    ax5.set_title('All Boundary Lines')
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Time t')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot total membership (should be 1 everywhere)
    ax6 = plt.subplot(3, 3, 6)
    total_membership = np.sum(all_memberships, axis=0)
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
        color = colors[i % len(colors)]
        ax7.plot(x_range, line_vals, '--', linewidth=1, alpha=0.6, color=color)
    
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
            color = colors[i % len(colors)]
            ax8.axvline(t_boundary, color=color, linestyle='--', alpha=0.7, label=f'Boundary {i}')
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
    print(f"Total parameters: {len(model.a_params)}")
    print(f"Number of boundary lines: {len(model.a_params)}")
    print(f"Number of total windows: {model.total_windows}")
    print(f"Number of main windows: {model.num_windows}")
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    print(f"Total membership range: {total_membership.min():.6f} to {total_membership.max():.6f}")
    
    # Check that memberships sum to 1
    membership_error = np.abs(total_membership - 1.0).max()
    print(f"Maximum membership sum error: {membership_error:.6f}")
    
    return model, all_memberships, main_memberships, compensation

def visualize_negative_parameters_case():
    """
    Test with fully negative parameters to see edge case handling
    """
    print("Testing with negative parameters...")
    
    # Create model with negative parameters
    model = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    model.a_params = np.array([-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05])
    
    print(f"Negative parameters: {model.a_params}")
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-300, 600, 200)
    
    # Compute compensation
    compensation = model.compute_compensation(x_range, t_range)
    main_memberships = model.get_main_window_memberships(x_range, t_range)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Plot main window memberships
    for i in range(4):
        ax = plt.subplot(3, 3, i + 1)
        im = ax.contourf(X, T, main_memberships[i], levels=20, cmap='viridis')
        ax.set_title(f'Main Window {i} (Negative Params)')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add boundary lines
        boundary_vals = model.get_boundary_lines(x_range)
        if i+1 < len(boundary_vals) - 1:
            ax.plot(x_range, boundary_vals[i+1], 'r--', linewidth=2, alpha=0.8)
            ax.plot(x_range, boundary_vals[i+2], 'r--', linewidth=2, alpha=0.8)
    
    # Plot all boundary lines
    ax5 = plt.subplot(3, 3, 5)
    boundary_vals = model.get_boundary_lines(x_range)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    offsets = [(i - 2) * model.bin_width for i in range(len(model.a_params))]
    
    for i, line_vals in enumerate(boundary_vals):
        color = colors[i % len(colors)]
        ax5.plot(x_range, line_vals, '--', linewidth=2, color=color, 
                label=f'Line {i}: t = {model.a_params[i]:.2f}x + {offsets[i]}')
    
    ax5.set_title('All Boundary Lines (Negative)')
    ax5.set_xlabel('X coordinate')
    ax5.set_ylabel('Time t')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Plot compensation function
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.contourf(X, T, compensation, levels=30, cmap='coolwarm')
    ax7.set_title('Compensation (Negative Params)')
    ax7.set_xlabel('X coordinate')
    ax7.set_ylabel('Time t')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # Add boundary lines to compensation plot
    for i, line_vals in enumerate(boundary_vals):
        color = colors[i % len(colors)]
        ax7.plot(x_range, line_vals, '--', linewidth=1, alpha=0.6, color=color)
    
    # Compensation slices
    ax8 = plt.subplot(3, 3, 8)
    x_idx = np.argmin(np.abs(x_range - 100))
    compensation_slice = compensation[x_idx, :]
    ax8.plot(t_range, compensation_slice, 'b-', linewidth=2)
    ax8.set_title(f'Compensation at x = {x_range[x_idx]:.1f} (Negative)')
    ax8.set_xlabel('Time t')
    ax8.set_ylabel('Compensation C(x,t)')
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    t_idx = np.argmin(np.abs(t_range - 200))
    compensation_slice = compensation[:, t_idx]
    ax9.plot(x_range, compensation_slice, 'r-', linewidth=2)
    ax9.set_title(f'Compensation at t = {t_range[t_idx]:.1f} (Negative)')
    ax9.set_xlabel('X coordinate')
    ax9.set_ylabel('Compensation C(x,t)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Negative case - Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    
    return model, compensation

def compare_positive_vs_negative():
    """
    Compare positive and negative parameter cases side by side
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-300, 600, 200)
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Positive case
    model_pos = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    model_pos.a_params = np.array([0.05, 0.1, 0.2, 0.15, 0.25, 0.3, 0.35])
    compensation_pos = model_pos.compute_compensation(x_range, t_range)
    boundary_vals_pos = model_pos.get_boundary_lines(x_range)
    
    # Negative case  
    model_neg = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    model_neg.a_params = np.array([-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05])
    compensation_neg = model_neg.compute_compensation(x_range, t_range)
    boundary_vals_neg = model_neg.get_boundary_lines(x_range)
    
    # Mixed case
    model_mix = Simple2DMultiWindow(num_windows=4, bin_width=100, temperature=1)
    model_mix.a_params = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.15, 0.25])
    compensation_mix = model_mix.compute_compensation(x_range, t_range)
    boundary_vals_mix = model_mix.get_boundary_lines(x_range)
    
    cases = [
        (compensation_pos, boundary_vals_pos, "Positive Parameters", model_pos.a_params),
        (compensation_neg, boundary_vals_neg, "Negative Parameters", model_neg.a_params),
        (compensation_mix, boundary_vals_mix, "Mixed Parameters", model_mix.a_params)
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for idx, (compensation, boundary_vals, title, params) in enumerate(cases):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        im = ax.contourf(X, T, compensation, levels=20, cmap='coolwarm')
        ax.set_title(f'{title}\n{params}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        
        # Add boundary lines
        for i, line_vals in enumerate(boundary_vals):
            color = colors[i % len(colors)]
            ax.plot(x_range, line_vals, '--', linewidth=1, alpha=0.7, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add empty subplots with statistics
    for idx in range(3, 6):
        row = idx // 3  
        col = idx % 3
        ax = axes[row, col]
        ax.axis('off')
        
        if idx == 3:
            stats_text = f"""Positive Case Statistics:
Range: {compensation_pos.min():.2f} to {compensation_pos.max():.2f}
Parameters: {model_pos.a_params}"""
        elif idx == 4:
            stats_text = f"""Negative Case Statistics:
Range: {compensation_neg.min():.2f} to {compensation_neg.max():.2f}  
Parameters: {model_neg.a_params}"""
        else:
            stats_text = f"""Mixed Case Statistics:
Range: {compensation_mix.min():.2f} to {compensation_mix.max():.2f}
Parameters: {model_mix.a_params}"""
            
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
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
    
    print("\nQuick test passed! Running full visualization...")
    
    # Main visualization
    model, all_memberships, main_memberships, compensation = visualize_simple_2d_problem()
    
    print("\n" + "=" * 60)
    print("Negative Parameters Test")
    print("=" * 60)
    
    # Negative parameters test
    model_neg, compensation_neg = visualize_negative_parameters_case()
    
    print("\n" + "=" * 60)
    print("Positive vs Negative vs Mixed Comparison")
    print("=" * 60)
    
    # Comparison
    compare_positive_vs_negative()
    
    print("\nVisualization complete!")