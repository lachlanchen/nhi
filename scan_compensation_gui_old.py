#!/usr/bin/env python3
"""
Complete GUI application for scan compensation with 3D spectral visualization
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.colors as mcolors

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def wavelength_to_rgb(wavelength):
    """
    Convert wavelength in nm to RGB color
    Based on CIE color matching functions
    """
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)  # Black for out of range
    
    # Simplified wavelength to RGB conversion
    if 380 <= wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength < 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wavelength <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    
    # Intensity falloff near the edges
    factor = 1.0
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 700 < wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    
    return (r * factor, g * factor, b * factor)

def load_npz_events(npz_path):
    """Load events from NPZ file"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32) 
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)
    
    # Convert polarity to [-1, 1] if it's [0, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
    
    return x, y, t, p

class ScanCompensation(nn.Module):
    def __init__(self, initial_params):
        super().__init__()
        self.params = nn.Parameter(initial_params)
    
    def warp(self, x_coords, y_coords, timestamps):
        """Adjust timestamps based on x and y positions"""
        a_x = self.params[0]
        a_y = self.params[1]
        t_warped = timestamps - a_x * x_coords - a_y * y_coords
        return x_coords, y_coords, t_warped
    
    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width):
        """Process events through the model"""
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        # Define time binning parameters
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        t_start = t_warped.min()
        t_end = t_warped.max()
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        # Normalize time to [0, num_bins)
        t_norm = (t_warped - t_start) / time_bin_width

        # Compute floor and ceil indices for time bins
        t0 = torch.floor(t_norm)
        t1 = t0 + 1

        # Compute weights for linear interpolation over time
        wt = (t_norm - t0).float()

        # Clamping indices to valid range
        t0_clamped = t0.clamp(0, num_bins - 1)
        t1_clamped = t1.clamp(0, num_bins - 1)

        # Cast x and y to long for indexing
        x_indices = x_warped.long()
        y_indices = y_warped.long()

        # Ensure spatial indices are within bounds
        valid_mask = (x_indices >= 0) & (x_indices < W) & \
                     (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]

        # Compute linear indices for the event tensor
        spatial_indices = y_indices * W + x_indices
        spatial_indices = spatial_indices.long()

        # For t0
        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t0 = flat_indices_t0.long()
        weights_t0 = ((1 - wt) * polarities).float()

        # For t1
        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        flat_indices_t1 = flat_indices_t1.long()
        weights_t1 = (wt * polarities).float()

        # Combine indices and weights
        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        # Add explicit bounds checking
        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        # Create the flattened event tensor
        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        # Accumulate events
        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        # Reshape back to (num_bins, H, W)
        event_tensor = event_tensor_flat.view(num_bins, H, W)

        # Compute variance loss
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        loss = torch.mean(variances)

        return event_tensor, loss, t_warped, x_indices, y_indices, polarities

class ScanCompensationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scan Compensation with 3D Spectral Visualization")
        self.root.geometry("1400x900")
        
        # Data storage
        self.events_data = None
        self.model = None
        self.compensation_params = None
        self.event_tensor_compensated = None
        self.compensated_events = None  # Individual compensated events
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=40).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Optimization parameters
        opt_frame = ttk.LabelFrame(control_frame, text="Optimization Parameters", padding=10)
        opt_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(opt_frame, text="Optimization Bin Width (μs):").pack(anchor=tk.W)
        self.opt_bin_var = tk.DoubleVar(value=100000)  # 100ms
        ttk.Entry(opt_frame, textvariable=self.opt_bin_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(opt_frame, text="Iterations:").pack(anchor=tk.W)
        self.iterations_var = tk.IntVar(value=1000)
        ttk.Entry(opt_frame, textvariable=self.iterations_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(opt_frame, text="Learning Rate:").pack(anchor=tk.W)
        self.lr_var = tk.DoubleVar(value=1.0)
        ttk.Entry(opt_frame, textvariable=self.lr_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        # Final visualization parameters
        vis_frame = ttk.LabelFrame(control_frame, text="Visualization Parameters", padding=10)
        vis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(vis_frame, text="Final Bin Width (μs):").pack(anchor=tk.W)
        self.final_bin_var = tk.DoubleVar(value=1000)  # 1ms
        ttk.Entry(vis_frame, textvariable=self.final_bin_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(vis_frame, text="Sample Rate (1=all events):").pack(anchor=tk.W)
        self.sample_rate_var = tk.IntVar(value=100)  # Sample every 100th event
        ttk.Entry(vis_frame, textvariable=self.sample_rate_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(vis_frame, text="Time Stretch Factor:").pack(anchor=tk.W)
        self.time_stretch_var = tk.DoubleVar(value=10.0)
        ttk.Entry(vis_frame, textvariable=self.time_stretch_var, width=20).pack(anchor=tk.W, pady=(0, 5))
        
        # Wavelength range controls
        wave_frame = ttk.LabelFrame(control_frame, text="Wavelength Range (nm)", padding=10)
        wave_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Min wavelength
        ttk.Label(wave_frame, text="Min Wavelength:").pack(anchor=tk.W)
        self.min_wave_var = tk.DoubleVar(value=380)
        min_wave_frame = ttk.Frame(wave_frame)
        min_wave_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Entry(min_wave_frame, textvariable=self.min_wave_var, width=10).pack(side=tk.LEFT)
        self.min_wave_scale = ttk.Scale(min_wave_frame, from_=380, to=780, orient=tk.HORIZONTAL, 
                                       variable=self.min_wave_var, length=150)
        self.min_wave_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # Max wavelength
        ttk.Label(wave_frame, text="Max Wavelength:").pack(anchor=tk.W)
        self.max_wave_var = tk.DoubleVar(value=780)
        max_wave_frame = ttk.Frame(wave_frame)
        max_wave_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Entry(max_wave_frame, textvariable=self.max_wave_var, width=10).pack(side=tk.LEFT)
        self.max_wave_scale = ttk.Scale(max_wave_frame, from_=380, to=780, orient=tk.HORIZONTAL, 
                                       variable=self.max_wave_var, length=150)
        self.max_wave_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # 3D View controls
        view_frame = ttk.LabelFrame(control_frame, text="3D View Settings", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(view_frame, text="View Angle:").pack(anchor=tk.W)
        self.view_var = tk.StringVar(value="default")
        view_combo = ttk.Combobox(view_frame, textvariable=self.view_var, width=18)
        view_combo['values'] = ("default", "side", "vertical", "horizontal", "lateral", "normal", "reverse")
        view_combo.pack(anchor=tk.W, pady=(0, 5))
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Run Optimization", command=self.run_optimization).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Generate 3D Visualization", command=self.generate_3d_vis).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Show Compensated Frames", command=self.show_compensated_frames).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Update 3D View", command=self.update_3d_view).pack(fill=tk.X, pady=(0, 5))
        
        # Progress and status
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=8, width=50)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for visualization
        self.vis_frame = ttk.Frame(main_frame)
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select NPZ Event File",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def log_result(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
        
    def run_optimization(self):
        """Run scan compensation optimization in a separate thread"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file first")
            return
            
        def optimization_worker():
            try:
                self.update_status("Loading events...")
                self.log_result("Loading events from: " + self.file_path_var.get())
                
                # Load events
                x, y, t, p = load_npz_events(self.file_path_var.get())
                self.events_data = (x, y, t, p)
                
                self.log_result(f"Loaded {len(x):,} events")
                self.log_result(f"Time range: {t.min():.0f} - {t.max():.0f} μs")
                
                self.update_status("Running optimization...")
                
                # Convert to tensors
                xs = torch.tensor(x, device=device)
                ys = torch.tensor(y, device=device)
                ts = torch.tensor(t, device=device)
                ps = torch.tensor(p, device=device)
                
                # Initialize model
                initial_params = torch.zeros(2, device=device, requires_grad=True)
                self.model = ScanCompensation(initial_params)
                
                # Optimizer
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_var.get())
                
                # Training loop
                losses = []
                num_iterations = self.iterations_var.get()
                bin_width = self.opt_bin_var.get()
                
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    event_tensor, loss, _, _, _, _ = self.model(xs, ys, ts, ps, 720, 1280, bin_width)
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if i % 100 == 0:
                        params = self.model.params.detach().cpu()
                        self.log_result(f"Iteration {i}, Loss: {loss.item():.6f}, "
                                      f"a_x={params[0].item():.3f}, a_y={params[1].item():.3f}")
                        
                        # Update progress
                        progress = (i / num_iterations) * 100
                        self.update_progress(progress)
                    
                    # Learning rate scheduling
                    if i == int(0.5 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.log_result("Reduced learning rate by 50%")
                    elif i == int(0.8 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.1
                        self.log_result("Reduced learning rate by 90%")
                
                # Save final parameters
                final_params = self.model.params.detach().cpu()
                self.compensation_params = [final_params[0].item(), final_params[1].item()]
                
                self.log_result(f"\nOptimization Complete!")
                self.log_result(f"Final parameters: a_x={self.compensation_params[0]:.6f}, a_y={self.compensation_params[1]:.6f}")
                self.log_result(f"Final loss: {losses[-1]:.6f}")
                self.log_result(f"Units: a_x and a_y are in microseconds per pixel")
                
                self.update_status("Optimization complete")
                self.update_progress(100)
                
            except Exception as e:
                self.log_result(f"Error during optimization: {str(e)}")
                self.update_status("Error occurred")
                messagebox.showerror("Error", f"Optimization failed: {str(e)}")
        
        # Run in separate thread
        thread = threading.Thread(target=optimization_worker)
        thread.daemon = True
        thread.start()
        
    def generate_3d_vis(self):
        """Generate 3D visualization of compensated data"""
        if self.events_data is None or self.model is None:
            messagebox.showerror("Error", "Please run optimization first")
            return
            
        try:
            self.update_status("Generating 3D visualization...")
            
            x, y, t, p = self.events_data
            
            # Create compensated events with final bin width
            xs = torch.tensor(x, device=device)
            ys = torch.tensor(y, device=device)
            ts = torch.tensor(t, device=device)
            ps = torch.tensor(p, device=device)
            
            with torch.no_grad():
                event_tensor, _, t_warped, x_valid, y_valid, p_valid = self.model(
                    xs, ys, ts, ps, 720, 1280, self.final_bin_var.get())
                
                # Store compensated individual events
                self.compensated_events = {
                    'x': x_valid.cpu().numpy(),
                    'y': y_valid.cpu().numpy(), 
                    't': t_warped.cpu().numpy(),
                    'p': p_valid.cpu().numpy()
                }
                
                self.event_tensor_compensated = event_tensor.detach().cpu()
                self.event_tensor_shape = event_tensor.shape
            
            # Create 3D visualization
            self.create_3d_plot()
            
            self.log_result(f"Generated 3D visualization with {self.event_tensor_shape[0]} time bins")
            self.log_result(f"Final bin width: {self.final_bin_var.get()} μs")
            self.log_result(f"Compensated events: {len(self.compensated_events['x']):,}")
            self.update_status("3D visualization complete")
            
        except Exception as e:
            self.log_result(f"Error generating 3D visualization: {str(e)}")
            messagebox.showerror("Error", f"3D visualization failed: {str(e)}")
            
    def create_3d_plot(self):
        """Create 3D scatter plot of compensated events"""
        # Clear previous plot
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        if self.compensated_events is None:
            return
            
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get compensated event data
        xs = self.compensated_events['x']
        ys = self.compensated_events['y']
        ts = self.compensated_events['t']
        ps = self.compensated_events['p']
        
        # Sample events for performance
        sample_rate = self.sample_rate_var.get()
        if sample_rate > 1:
            indices = np.arange(0, len(xs), sample_rate)
            xs_sampled = xs[indices]
            ys_sampled = ys[indices]
            ts_sampled = ts[indices]
            ps_sampled = ps[indices]
        else:
            xs_sampled = xs
            ys_sampled = ys  
            ts_sampled = ts
            ps_sampled = ps
            
        self.log_result(f"Displaying {len(xs_sampled):,} sampled events")
        
        # Separate positive and negative events
        pos_mask = ps_sampled > 0
        neg_mask = ps_sampled <= 0
        
        # Time normalization for coloring
        t_min, t_max = ts_sampled.min(), ts_sampled.max()
        time_stretch = self.time_stretch_var.get()
        
        # Map times to wavelengths
        wavelengths = np.linspace(self.min_wave_var.get(), self.max_wave_var.get(), 1000)
        
        # Create colors based on time (representing spectrum)
        if len(ts_sampled) > 0:
            # Normalize time to [0, 1]
            t_norm = (ts_sampled - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(ts_sampled)
            
            # Map to wavelength colors
            colors = []
            for t in t_norm:
                wavelength = self.min_wave_var.get() + t * (self.max_wave_var.get() - self.min_wave_var.get())
                rgb = wavelength_to_rgb(wavelength)
                colors.append(rgb)
            colors = np.array(colors)
            
            # Plot events with swapped axes (x, time, y) like your code
            # Convert time to stretched scale
            t_stretched = time_stretch * (ts_sampled - t_min) / 1e6
            
            if np.any(pos_mask):
                ax.scatter(xs_sampled[pos_mask], t_stretched[pos_mask], ys_sampled[pos_mask], 
                          c=colors[pos_mask], alpha=0.6, s=1, marker='.')
            
            if np.any(neg_mask):
                # Make negative events slightly dimmer
                neg_colors = colors[neg_mask] * 0.7
                ax.scatter(xs_sampled[neg_mask], t_stretched[neg_mask], ys_sampled[neg_mask], 
                          c=neg_colors, alpha=0.4, s=1, marker='.')
        
        # Set limits and labels
        ax.set_xlim3d(0, 1279)
        ax.set_zlim3d(0, 719)
        if len(ts_sampled) > 0:
            ax.set_ylim3d(0, time_stretch * (t_max - t_min) / 1e6)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Time (spectral dimension)')
        ax.set_zlabel('Y (pixels)')
        
        # Apply scaling like your code
        x_scale = 7
        z_scale = 7
        y_scale = time_stretch
        
        scale = np.diag([x_scale, y_scale, z_scale, 1.0])
        scale = scale * (1.0 / scale.max())
        scale[3, 3] = 1.0
        
        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)
        
        ax.get_proj = short_proj
        
        # Set aspect ratio
        ax.set_box_aspect([1280/720, 2, 1.0])
        
        # Set view angle
        self.set_view_angle(ax)
        
        ax.set_title(f'3D Compensated Event Data (Spectral)\n'
                    f'Wavelength: {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm\n'
                    f'Time bins: {self.event_tensor_shape[0]}, Events: {len(xs_sampled):,}')
        
        # Add canvas to GUI
        canvas = FigureCanvasTkAgg(fig, self.vis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
        toolbar.update()
        
        # Store for view updates
        self.current_fig = fig
        self.current_ax = ax
        self.current_canvas = canvas
        
    def set_view_angle(self, ax):
        """Set the 3D view angle based on selection"""
        view_settings = {
            "vertical": (0, 90),
            "horizontal": (90, 0),
            "lateral": (90, 90),
            "side": (30, 18),
            "r-side": (-30, 18),
            "reverse": (-60, 18),
            "normal": (64, 18),
            "normal45": (45, 0),
            "default": (30, -30)
        }
        view = self.view_var.get()
        elev, azim = view_settings.get(view, (30, -30))
        ax.view_init(elev, azim)
        
    def update_3d_view(self):
        """Update the 3D view angle"""
        if hasattr(self, 'current_ax') and hasattr(self, 'current_canvas'):
            self.set_view_angle(self.current_ax)
            self.current_canvas.draw()
            
    def show_compensated_frames(self):
        """Show compensated frames with wavelength color mapping"""
        if self.event_tensor_compensated is None:
            messagebox.showerror("Error", "Please generate 3D visualization first")
            return
            
        try:
            # Create new window for frame display
            frame_window = tk.Toplevel(self.root)
            frame_window.title("Compensated Spectral Frames")
            frame_window.geometry("1200x800")
            
            # Create figure with subplots
            fig = Figure(figsize=(15, 10))
            
            data_tensor = self.event_tensor_compensated
            num_bins = data_tensor.shape[0]
            wavelengths = np.linspace(self.min_wave_var.get(), self.max_wave_var.get(), num_bins)
            
            # Select representative frames
            frame_indices = np.linspace(0, num_bins-1, min(9, num_bins), dtype=int)
            
            rows = 3
            cols = 3
            
            for i, frame_idx in enumerate(frame_indices):
                if i >= rows * cols:
                    break
                    
                try:
                    ax = fig.add_subplot(rows, cols, i+1)
                    
                    # Extract frame data and convert to numpy for plotting
                    frame_data_tensor = data_tensor[frame_idx]
                    frame_data = frame_data_tensor.detach().numpy()
                    wavelength = wavelengths[frame_idx]
                    
                    # Create RGB image
                    rgb_color = wavelength_to_rgb(wavelength)
                    colored_frame = np.zeros((frame_data.shape[0], frame_data.shape[1], 3))
                    
                    # Normalize frame data
                    frame_max = frame_data.max()
                    if frame_max > 0:
                        normalized_frame = frame_data / frame_max
                        
                        # Apply color
                        for c in range(3):
                            colored_frame[:, :, c] = normalized_frame * rgb_color[c]
                    
                    ax.imshow(colored_frame, aspect='auto')
                    ax.set_title(f'λ = {wavelength:.0f} nm\nBin {frame_idx}')
                    ax.set_xlabel('X (pixels)')
                    ax.set_ylabel('Y (pixels)')
                except Exception as e:
                    print(f"Warning: Could not display frame {frame_idx}: {e}")
                    continue
            
            plt.tight_layout()
            
            # Add canvas to window
            canvas = FigureCanvasTkAgg(fig, frame_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame_window)
            toolbar.update()
            
            self.log_result(f"Displayed {len(frame_indices)} spectral frames")
            self.log_result(f"Wavelength range: {wavelengths[0]:.0f} - {wavelengths[-1]:.0f} nm")
            
        except Exception as e:
            self.log_result(f"Error showing compensated frames: {str(e)}")
            messagebox.showerror("Error", f"Frame display failed: {str(e)}")

def main():
    root = tk.Tk()
    app = ScanCompensationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()