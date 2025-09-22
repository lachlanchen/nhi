import numpy as np
import matplotlib.pyplot as plt

class MultiWindowCompensation:
    def __init__(self, num_windows, bin_width=100, temperature=1, a_params=None):
        """
        General multi-window compensation class
        
        Args:
            num_windows: Number of windows (n)
            bin_width: Width of each time bin  
            temperature: Smoothness parameter for soft assignments
            a_params: Array of n+2 parameters [a0, a1, ..., a_{n+1}]. If None, auto-generate
        """
        self.num_windows = num_windows
        self.bin_width = bin_width
        self.temperature = temperature
        
        # For n windows, we need n+2 parameters
        num_params_needed = num_windows + 2
        
        if a_params is None:
            # Auto-generate reasonable default parameters
            self.a_params = np.linspace(0.05, 0.3, num_params_needed)
        else:
            self.a_params = np.array(a_params)
            if len(self.a_params) != num_params_needed:
                raise ValueError(f"For {num_windows} windows, need exactly {num_params_needed} parameters, got {len(self.a_params)}")
        
        print(f"Initialized {num_windows} windows with {num_params_needed} parameters")
        print(f"Parameters: {self.a_params}")
    
    def get_boundary_lines(self, x):
        """
        Compute boundary line values: t = a_i * x + offset_i
        For n windows, we use n+1 boundary lines (first n+1 parameters)
        
        Args:
            x: x coordinates (scalar or array)
            
        Returns:
            boundary_values: [num_windows + 1, len(x)] array or [num_windows + 1] array
        """
        x = np.asarray(x)
        
        # Calculate offsets: [-bin_width, 0, bin_width, 2*bin_width, ...]
        offsets = np.array([(i - 1) * self.bin_width for i in range(self.num_windows + 1)])
        
        # Use first num_windows+1 parameters for boundary lines
        boundary_params = self.a_params[:self.num_windows + 1]
        
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
        Compute soft membership for each window using sigmoid functions
        
        Args:
            x: x coordinates array [len_x]
            t: t coordinates array [len_t]
            
        Returns:
            memberships: [num_windows, len_x, len_t] array
        """
        boundary_values = self.get_boundary_lines(x)  # [num_windows+1, len_x]
        
        # Create meshgrids for broadcasting
        X, T = np.meshgrid(x, t, indexing='ij')
        
        memberships = []
        
        for i in range(self.num_windows):
            # Window i: between boundary line i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]      # [len_x, 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len_x, 1]
            
            # Soft membership: σ((t - lower)/τ) * σ((upper - t)/τ)
            lower_sigmoid = self.sigmoid((T - lower_bound) / self.temperature)
            upper_sigmoid = self.sigmoid((upper_bound - T) / self.temperature)
            
            membership = lower_sigmoid * upper_sigmoid
            memberships.append(membership)
        
        memberships = np.array(memberships)  # [num_windows, len_x, len_t]
        
        # Normalize to ensure sum = 1
        memberships_sum = np.sum(memberships, axis=0, keepdims=True)  # [1, len_x, len_t]
        memberships_sum = np.maximum(memberships_sum, 1e-8)  # Avoid division by zero
        normalized_memberships = memberships / memberships_sum
        
        return normalized_memberships
    
    def compute_within_window_interpolation(self, x, t):
        """
        For each window, interpolate between boundary parameters
        
        Args:
            x: x coordinates array [len_x]
            t: t coordinates array [len_t]
            
        Returns:
            interpolated_slopes: [num_windows, len_x, len_t] array
        """
        boundary_values = self.get_boundary_lines(x)  # [num_windows+1, len_x]
        
        # Create meshgrids
        X, T = np.meshgrid(x, t, indexing='ij')
        
        interpolated_slopes = []
        
        for i in range(self.num_windows):
            # Window i: between boundary lines i and i+1
            lower_bound = boundary_values[i][:, np.newaxis]      # [len_x, 1]
            upper_bound = boundary_values[i + 1][:, np.newaxis]  # [len_x, 1]
            
            # Interpolation parameter α ∈ [0,1]
            window_width = upper_bound - lower_bound
            window_width = np.maximum(window_width, 1e-8)  # Avoid division by zero
            
            alpha = (T - lower_bound) / window_width
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to [0,1]
            
            # Interpolate between boundary parameters a_i and a_{i+1}
            slope = (1 - alpha) * self.a_params[i] + alpha * self.a_params[i + 1]
            
            interpolated_slopes.append(slope)
        
        return np.array(interpolated_slopes)
    
    def compute_compensation(self, x, t):
        """
        Compute the final compensation C(x,t) for given coordinates
        
        Args:
            x: x coordinates array [len_x]
            t: t coordinates array [len_t]
            
        Returns:
            compensation: [len_x, len_t] array
        """
        # Get window memberships
        memberships = self.compute_window_memberships(x, t)  # [num_windows, len_x, len_t]
        
        # Get interpolated slopes for each window
        slopes = self.compute_within_window_interpolation(x, t)  # [num_windows, len_x, len_t]
        
        # Create meshgrid for x
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Weighted sum of compensations: C = Σᵢ wᵢ × slope_i × x
        compensation = np.sum(memberships * slopes * X[np.newaxis, :, :], axis=0)
        
        return compensation
    
    def get_boundary_info(self):
        """Return information about boundary lines"""
        offsets = [(i - 1) * self.bin_width for i in range(self.num_windows + 1)]
        boundary_info = []
        for i in range(self.num_windows + 1):
            boundary_info.append({
                'line_id': i,
                'parameter': self.a_params[i],
                'offset': offsets[i],
                'equation': f't = {self.a_params[i]:.3f}x + {offsets[i]:.0f}'
            })
        return boundary_info

def visualize_multi_window_compensation(num_windows=3, a_params=None, bin_width=100):
    """
    Visualize the multi-window compensation for any number of windows
    
    Args:
        num_windows: Number of windows to create
        a_params: Optional parameter array (if None, auto-generate)
        bin_width: Bin width parameter
    """
    # Create model
    model = MultiWindowCompensation(num_windows=num_windows, bin_width=bin_width, 
                                   temperature=1, a_params=a_params)
    
    # Display boundary information
    print("\nBoundary line information:")
    for info in model.get_boundary_info():
        print(f"  Line {info['line_id']}: {info['equation']}")
    
    # Create coordinate grids
    x_range = np.linspace(0, 200, 100)
    t_range = np.linspace(-150, 450, 200)
    
    # Compute memberships and compensation
    print(f"\nComputing memberships and compensation...")
    memberships = model.compute_window_memberships(x_range, t_range)
    compensation = model.compute_compensation(x_range, t_range)
    
    print(f"Memberships shape: {memberships.shape}")
    print(f"Compensation shape: {compensation.shape}")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Create meshgrids for plotting
    X, T = np.meshgrid(x_range, t_range, indexing='ij')
    
    # Determine subplot layout based on number of windows
    rows = int(np.ceil((num_windows + 4) / 3))  # +4 for additional plots
    cols = 3
    
    # Plot window memberships
    for i in range(num_windows):
        ax = plt.subplot(rows, cols, i + 1)
        
        im = ax.contourf(X, T, memberships[i], levels=20, cmap='viridis')
        ax.set_title(f'Window {i} Membership')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Time t')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add boundary lines for this window
        boundary_vals = model.get_boundary_lines(x_range)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        if i < len(boundary_vals) - 1:
            ax.plot(x_range, boundary_vals[i], '--', linewidth=2, 
                   color=colors[i % len(colors)], alpha=0.8, label=f'Line {i}')
            ax.plot(x_range, boundary_vals[i + 1], '--', linewidth=2, 
                   color=colors[(i+1) % len(colors)], alpha=0.8, label=f'Line {i+1}')
            ax.legend()
    
    # Plot all boundary lines together
    ax_boundaries = plt.subplot(rows, cols, num_windows + 1)
    boundary_vals = model.get_boundary_lines(x_range)
    
    for i, line_vals in enumerate(boundary_vals):
        ax_boundaries.plot(x_range, line_vals, '--', linewidth=2, 
                         color=colors[i % len(colors)], 
                         label=f'Line {i}: {model.get_boundary_info()[i]["equation"]}')
    
    ax_boundaries.set_title('All Boundary Lines')
    ax_boundaries.set_xlabel('X coordinate')
    ax_boundaries.set_ylabel('Time t')
    ax_boundaries.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_boundaries.grid(True, alpha=0.3)
    
    # Plot total membership (should be 1 everywhere)
    ax_total = plt.subplot(rows, cols, num_windows + 2)
    total_membership = np.sum(memberships, axis=0)
    im_total = ax_total.contourf(X, T, total_membership, levels=20, cmap='RdBu_r')
    ax_total.set_title('Total Membership (should be ≈1)')
    ax_total.set_xlabel('X coordinate')
    ax_total.set_ylabel('Time t')
    plt.colorbar(im_total, ax=ax_total, shrink=0.8)
    
    # Plot compensation function
    ax_comp = plt.subplot(rows, cols, num_windows + 3)
    im_comp = ax_comp.contourf(X, T, compensation, levels=30, cmap='coolwarm')
    ax_comp.set_title('Compensation Function C(x,t)')
    ax_comp.set_xlabel('X coordinate')
    ax_comp.set_ylabel('Time t')
    plt.colorbar(im_comp, ax=ax_comp, shrink=0.8)
    
    # Add boundary lines to compensation plot
    for i, line_vals in enumerate(boundary_vals):
        ax_comp.plot(x_range, line_vals, 'k--', linewidth=1, alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Compensation range: {compensation.min():.3f} to {compensation.max():.3f}")
    print(f"Total membership range: {total_membership.min():.6f} to {total_membership.max():.6f}")
    
    membership_error = np.abs(total_membership - 1.0).max()
    print(f"Maximum membership sum error: {membership_error:.6f}")
    
    return model, memberships, compensation

def test_different_configurations():
    """Test different window configurations"""
    
    print("="*80)
    print("TESTING DIFFERENT WINDOW CONFIGURATIONS")
    print("="*80)
    
    # Test 1: 3 windows
    print("\n1. Testing 3 windows with default parameters:")
    model_3w = visualize_multi_window_compensation(num_windows=3)
    
    # Test 2: 3 windows with custom parameters
    print("\n2. Testing 3 windows with custom parameters:")
    custom_params_3w = [0.1, 0.2, 0.15, 0.25, 0.3]  # 5 parameters for 3 windows
    model_3w_custom = visualize_multi_window_compensation(num_windows=3, a_params=custom_params_3w)
    
    # Test 3: 5 windows
    print("\n3. Testing 5 windows:")
    custom_params_5w = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # 7 parameters for 5 windows
    model_5w = visualize_multi_window_compensation(num_windows=5, a_params=custom_params_5w)
    
    return model_3w, model_3w_custom, model_5w

if __name__ == "__main__":
    print("General Multi-Window Compensation Visualization")
    print("=" * 60)
    
    # Quick test
    print("Quick parameter validation test...")
    
    # Test 3 windows (should need 5 parameters)
    try:
        model_test = MultiWindowCompensation(num_windows=3, a_params=[0.1, 0.2, 0.15, 0.25, 0.3])
        print("✓ 3 windows with 5 parameters - SUCCESS")
    except ValueError as e:
        print(f"✗ 3 windows test failed: {e}")
    
    # Test 4 windows (should need 6 parameters)  
    try:
        model_test = MultiWindowCompensation(num_windows=4, a_params=[0.1, 0.2, 0.15, 0.25, 0.3, 0.35])
        print("✓ 4 windows with 6 parameters - SUCCESS")
    except ValueError as e:
        print(f"✗ 4 windows test failed: {e}")
    
    # Test wrong parameter count (should fail)
    try:
        model_test = MultiWindowCompensation(num_windows=3, a_params=[0.1, 0.2, 0.15])
        print("✗ This should have failed!")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\n" + "="*60)
    print("Running visualization tests...")
    
    # Run the main tests
    models = test_different_configurations()
    
    print("\nAll tests completed!")