#!/usr/bin/env python3
"""
Complete GUI application for scan compensation with 3D spectral visualization
Two-stage compensation: spatial first, then temporal quadratic
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
import matplotlib.colors as mcolors
import gc

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def wavelength_to_rgb(wavelength):
    """Convert wavelength in nm to RGB color"""
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)
    
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
    
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
    
    return x, y, t, p

class SpatialCompensation(nn.Module):
    """Stage 1: Spatial compensation only"""
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    
    def warp(self, x_coords, y_coords, timestamps):
        a_x = self.params[0]
        a_y = self.params[1]
        t_warped = timestamps - a_x * x_coords - a_y * y_coords
        return x_coords, y_coords, t_warped
    
    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width):
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        t_start = t_warped.min()
        t_end = t_warped.max()
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        t_norm = (t_warped - t_start) / time_bin_width
        t0 = torch.floor(t_norm)
        t1 = t0 + 1
        wt = (t_norm - t0).float()

        t0_clamped = t0.clamp(0, num_bins - 1)
        t1_clamped = t1.clamp(0, num_bins - 1)

        x_indices = x_warped.long()
        y_indices = y_warped.long()

        valid_mask = (x_indices >= 0) & (x_indices < W) & \
                     (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]

        spatial_indices = y_indices * W + x_indices
        spatial_indices = spatial_indices.long()

        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t0 = flat_indices_t0.long()
        weights_t0 = ((1 - wt) * polarities).float()

        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        flat_indices_t1 = flat_indices_t1.long()
        weights_t1 = (wt * polarities).float()

        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        event_tensor = event_tensor_flat.view(num_bins, H, W)
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        loss = torch.mean(variances)

        return event_tensor, loss, t_warped, x_indices, y_indices, polarities

class TemporalCompensation(nn.Module):
    """Stage 2: Add quadratic temporal compensation (flat in middle, more at edges)"""
    def __init__(self, spatial_params):
        super().__init__()
        # Fix spatial parameters
        self.a_x = spatial_params[0].item()
        self.a_y = spatial_params[1].item()
        
        # Only optimize temporal quadratic parameter
        self.a_quad = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        self.t_min = None
        self.t_max = None
    
    def warp(self, x_coords, y_coords, timestamps):
        # Apply spatial compensation (fixed)
        t_warped = timestamps - self.a_x * x_coords - self.a_y * y_coords
        
        # Set time normalization on first call
        if self.t_min is None:
            self.t_min = timestamps.min().detach()
            self.t_max = timestamps.max().detach()
        
        # Normalize time to [0, 1]
        t_norm = (timestamps - self.t_min) / (self.t_max - self.t_min + 1e-8)
        
        # Quadratic compensation: (t_norm - 0.5)^2
        # This is 0 at center (t_norm=0.5) and increases toward edges (0 and 1)
        quad_term = (t_norm - 0.5) ** 2
        
        # Apply temporal compensation
        temporal_correction = self.a_quad * quad_term * (self.t_max - self.t_min)
        t_warped = t_warped - temporal_correction
        
        return x_coords, y_coords, t_warped
    
    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width):
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        t_start = t_warped.min()
        t_end = t_warped.max()
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        t_norm = (t_warped - t_start) / time_bin_width
        t0 = torch.floor(t_norm)
        t1 = t0 + 1
        wt = (t_norm - t0).float()

        t0_clamped = t0.clamp(0, num_bins - 1)
        t1_clamped = t1.clamp(0, num_bins - 1)

        x_indices = x_warped.long()
        y_indices = y_warped.long()

        valid_mask = (x_indices >= 0) & (x_indices < W) & \
                     (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]

        spatial_indices = y_indices * W + x_indices
        spatial_indices = spatial_indices.long()

        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t0 = flat_indices_t0.long()
        weights_t0 = ((1 - wt) * polarities).float()

        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        flat_indices_t1 = flat_indices_t1.long()
        weights_t1 = (wt * polarities).float()

        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        num_elements = num_bins * H * W
        valid_flat_mask = (flat_indices >= 0) & (flat_indices < num_elements)
        flat_indices = flat_indices[valid_flat_mask]
        flat_weights = flat_weights[valid_flat_mask]

        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        event_tensor = event_tensor_flat.view(num_bins, H, W)
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        loss = torch.mean(variances)

        return event_tensor, loss, t_warped, x_indices, y_indices, polarities

class ScanCompensationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Two-Stage Scan Compensation")
        self.root.geometry("1600x1000")
        
        # Data storage
        self.events_data = None
        self.spatial_model = None
        self.temporal_model = None
        self.compensation_params = None
        self.compensated_events = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top control panel (compact)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # File selection - Row 1
        file_frame = ttk.LabelFrame(control_frame, text="File", padding=5)
        file_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=2, pady=2)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Parameters - Row 2
        param_frame1 = ttk.LabelFrame(control_frame, text="Optimization", padding=5)
        param_frame1.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        ttk.Label(param_frame1, text="Opt Bin (μs):").grid(row=0, column=0, sticky="w")
        self.opt_bin_var = tk.DoubleVar(value=100000)
        ttk.Entry(param_frame1, textvariable=self.opt_bin_var, width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(param_frame1, text="Spatial Iter:").grid(row=1, column=0, sticky="w")
        self.spatial_iterations_var = tk.IntVar(value=500)
        ttk.Entry(param_frame1, textvariable=self.spatial_iterations_var, width=10).grid(row=1, column=1, padx=2)
        
        ttk.Label(param_frame1, text="Temporal Iter:").grid(row=2, column=0, sticky="w")
        self.temporal_iterations_var = tk.IntVar(value=300)
        ttk.Entry(param_frame1, textvariable=self.temporal_iterations_var, width=10).grid(row=2, column=1, padx=2)
        
        param_frame2 = ttk.LabelFrame(control_frame, text="Visualization", padding=5)
        param_frame2.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        ttk.Label(param_frame2, text="Final Bin (μs):").grid(row=0, column=0, sticky="w")
        self.final_bin_var = tk.DoubleVar(value=1000)
        ttk.Entry(param_frame2, textvariable=self.final_bin_var, width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(param_frame2, text="Sample Rate:").grid(row=1, column=0, sticky="w")
        self.sample_rate_var = tk.IntVar(value=5000)  # Increased default sampling
        ttk.Entry(param_frame2, textvariable=self.sample_rate_var, width=10).grid(row=1, column=1, padx=2)
        
        # Wavelength controls - Row 2 continued
        wave_frame = ttk.LabelFrame(control_frame, text="Wavelength (nm)", padding=5)
        wave_frame.grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        
        ttk.Label(wave_frame, text="Min:").grid(row=0, column=0, sticky="w")
        self.min_wave_var = tk.DoubleVar(value=380)
        ttk.Entry(wave_frame, textvariable=self.min_wave_var, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(wave_frame, text="Max:").grid(row=1, column=0, sticky="w")
        self.max_wave_var = tk.DoubleVar(value=780)
        ttk.Entry(wave_frame, textvariable=self.max_wave_var, width=8).grid(row=1, column=1, padx=2)
        
        # Action buttons - Row 2 continued
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding=5)
        button_frame.grid(row=1, column=3, sticky="ew", padx=2, pady=2)
        
        ttk.Button(button_frame, text="Run Optimization", command=self.run_optimization).grid(row=0, column=0, pady=1, sticky="ew")
        ttk.Button(button_frame, text="Generate 3D", command=self.generate_3d_vis).grid(row=1, column=0, pady=1, sticky="ew")
        
        # Configure grid weights
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        
        # Content area - split between results and visualization
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Results (larger space)
        results_frame = ttk.LabelFrame(content_frame, text="Results & Status", padding=5)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Status and progress
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Results text (larger)
        self.results_text = tk.Text(results_frame, height=25, width=60, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side - Visualization
        self.vis_frame = ttk.LabelFrame(content_frame, text="3D Visualization", padding=5)
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
        """Run two-stage scan compensation optimization"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file first")
            return
            
        def optimization_worker():
            try:
                self.update_status("Loading events...")
                self.log_result("="*60)
                self.log_result("TWO-STAGE SCAN COMPENSATION")
                self.log_result("="*60)
                self.log_result("Loading: " + os.path.basename(self.file_path_var.get()))
                
                # Load events
                x, y, t, p = load_npz_events(self.file_path_var.get())
                self.events_data = (x, y, t, p)
                
                self.log_result(f"Events loaded: {len(x):,}")
                self.log_result(f"Time range: {t.min():.0f} - {t.max():.0f} μs")
                self.log_result(f"Duration: {(t.max()-t.min())/1e6:.3f} seconds")
                
                # Convert to tensors
                xs = torch.tensor(x, device=device)
                ys = torch.tensor(y, device=device)
                ts = torch.tensor(t, device=device)
                ps = torch.tensor(p, device=device)
                
                bin_width = self.opt_bin_var.get()
                
                # STAGE 1: Spatial compensation
                self.log_result("\n" + "="*40)
                self.log_result("STAGE 1: SPATIAL COMPENSATION")
                self.log_result("="*40)
                
                self.spatial_model = SpatialCompensation().to(device)
                spatial_optimizer = torch.optim.Adam(self.spatial_model.parameters(), lr=1.0)
                
                spatial_iterations = self.spatial_iterations_var.get()
                self.log_result(f"Spatial iterations: {spatial_iterations}")
                
                spatial_losses = []
                for i in range(spatial_iterations):
                    spatial_optimizer.zero_grad()
                    _, loss, _, _, _, _ = self.spatial_model(xs, ys, ts, ps, 720, 1280, bin_width)
                    loss.backward()
                    spatial_optimizer.step()
                    spatial_losses.append(loss.item())
                    
                    if i % 50 == 0:
                        params = self.spatial_model.params.detach().cpu()
                        self.log_result(f"  {i:3d}: Loss={loss.item():.6f}, "
                                      f"a_x={params[0].item():.3f}, a_y={params[1].item():.3f}")
                        progress = (i / (spatial_iterations + self.temporal_iterations_var.get())) * 100
                        self.update_progress(progress)
                
                spatial_params = self.spatial_model.params.detach().cpu()
                self.log_result(f"\nSpatial optimization complete:")
                self.log_result(f"  a_x = {spatial_params[0].item():.6f} μs/pixel")
                self.log_result(f"  a_y = {spatial_params[1].item():.6f} μs/pixel")
                self.log_result(f"  Loss reduction: {((spatial_losses[0]-spatial_losses[-1])/spatial_losses[0])*100:.1f}%")
                
                # STAGE 2: Temporal compensation
                self.log_result("\n" + "="*40)
                self.log_result("STAGE 2: TEMPORAL COMPENSATION")
                self.log_result("="*40)
                
                self.temporal_model = TemporalCompensation(spatial_params).to(device)
                temporal_optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr=0.1)
                
                temporal_iterations = self.temporal_iterations_var.get()
                self.log_result(f"Temporal iterations: {temporal_iterations}")
                self.log_result(f"Using quadratic compensation: (t_norm - 0.5)²")
                
                temporal_losses = []
                for i in range(temporal_iterations):
                    temporal_optimizer.zero_grad()
                    _, loss, _, _, _, _ = self.temporal_model(xs, ys, ts, ps, 720, 1280, bin_width)
                    loss.backward()
                    temporal_optimizer.step()
                    temporal_losses.append(loss.item())
                    
                    if i % 30 == 0:
                        a_quad = self.temporal_model.a_quad.item()
                        self.log_result(f"  {i:3d}: Loss={loss.item():.6f}, a_quad={a_quad:.6f}")
                        progress = ((spatial_iterations + i) / (spatial_iterations + temporal_iterations)) * 100
                        self.update_progress(progress)
                
                # Save final parameters
                self.compensation_params = {
                    'a_x': spatial_params[0].item(),
                    'a_y': spatial_params[1].item(),
                    'a_quad': self.temporal_model.a_quad.item(),
                    'spatial_loss_reduction': ((spatial_losses[0]-spatial_losses[-1])/spatial_losses[0])*100,
                    'temporal_loss_reduction': ((temporal_losses[0]-temporal_losses[-1])/temporal_losses[0])*100
                }
                
                self.log_result(f"\n" + "="*60)
                self.log_result("TWO-STAGE OPTIMIZATION COMPLETE!")
                self.log_result("="*60)
                self.log_result(f"Final parameters:")
                self.log_result(f"  Spatial: a_x={self.compensation_params['a_x']:.6f}, a_y={self.compensation_params['a_y']:.6f}")
                self.log_result(f"  Temporal: a_quad={self.compensation_params['a_quad']:.6f}")
                self.log_result(f"Final loss: {temporal_losses[-1]:.6f}")
                self.log_result(f"Spatial loss reduction: {self.compensation_params['spatial_loss_reduction']:.1f}%")
                self.log_result(f"Temporal loss reduction: {self.compensation_params['temporal_loss_reduction']:.1f}%")
                
                self.update_status("Two-stage optimization complete")
                self.update_progress(100)
                
            except Exception as e:
                self.log_result(f"ERROR: {str(e)}")
                self.update_status("Error occurred")
                messagebox.showerror("Error", f"Optimization failed: {str(e)}")
        
        thread = threading.Thread(target=optimization_worker)
        thread.daemon = True
        thread.start()
        
    def generate_3d_vis(self):
        """Generate 3D visualization with memory management"""
        if self.events_data is None or self.temporal_model is None:
            messagebox.showerror("Error", "Please run optimization first")
            return
            
        try:
            self.update_status("Generating 3D visualization...")
            self.log_result("\n" + "="*60)
            self.log_result("3D SPECTRAL VISUALIZATION")
            self.log_result("="*60)
            
            x, y, t, p = self.events_data
            
            # Process events in smaller chunks to avoid memory issues
            chunk_size = 100000
            num_events = len(x)
            
            # Pre-sample events for visualization
            sample_rate = max(1, self.sample_rate_var.get())
            if sample_rate > 1:
                indices = list(range(0, num_events, sample_rate))
                x_sampled = x[indices]
                y_sampled = y[indices]
                t_sampled = t[indices]
                p_sampled = p[indices]
            else:
                x_sampled = x
                y_sampled = y
                t_sampled = t
                p_sampled = p
                
            self.log_result(f"Sampled {len(x_sampled):,} events for visualization")
            
            # Apply compensation to sampled events
            xs = torch.tensor(x_sampled, device=device)
            ys = torch.tensor(y_sampled, device=device)
            ts = torch.tensor(t_sampled, device=device)
            ps = torch.tensor(p_sampled, device=device)
            
            with torch.no_grad():
                # Get compensated times
                _, _, t_warped = self.temporal_model.warp(xs, ys, ts)
                
                # Convert to CPU and clean up GPU memory
                t_comp = t_warped.cpu().numpy()
                x_comp = x_sampled
                y_comp = y_sampled
                p_comp = p_sampled
                
                # Clear GPU tensors
                del xs, ys, ts, ps, t_warped
                torch.cuda.empty_cache()
                gc.collect()
                
                self.compensated_events = {
                    'x': x_comp,
                    'y': y_comp,
                    't': t_comp,
                    'p': p_comp
                }
            
            self.log_result(f"Applied two-stage compensation")
            self.log_result(f"Memory management: GPU cache cleared")
            
            # Create 3D plot
            self.create_3d_plot()
            
            self.update_status("3D visualization complete")
            
        except Exception as e:
            self.log_result(f"ERROR generating 3D: {str(e)}")
            messagebox.showerror("Error", f"3D visualization failed: {str(e)}")
            
    def create_3d_plot(self):
        """Create memory-efficient 3D scatter plot"""
        # Clear previous plot
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        if not self.compensated_events:
            return
            
        try:
            # Create figure
            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get event data
            xs = self.compensated_events['x']
            ys = self.compensated_events['y']
            ts = self.compensated_events['t']
            ps = self.compensated_events['p']
            
            self.log_result(f"Plotting {len(xs):,} events")
            
            if len(ts) > 0:
                # Time normalization and coloring
                t_min = ts.min()
                t_max = ts.max()
                t_range = t_max - t_min if t_max > t_min else 1
                
                # Create colors for wavelength mapping
                wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
                t_norm = (ts - t_min) / t_range
                wavelengths = self.min_wave_var.get() + t_norm * wavelength_range
                colors = [wavelength_to_rgb(w) for w in wavelengths]
                
                # Separate positive and negative events
                pos_mask = ps > 0
                neg_mask = ps <= 0
                
                # Time stretching for visualization
                time_stretch = 10.0
                t_stretched = (ts - t_min) * time_stretch / 1e6
                
                # Plot positive events
                if np.any(pos_mask):
                    pos_colors = [colors[i] for i in range(len(colors)) if pos_mask[i]]
                    ax.scatter(xs[pos_mask], t_stretched[pos_mask], ys[pos_mask], 
                              c=pos_colors, alpha=0.7, s=2, marker='.')
                
                # Plot negative events (dimmer)
                if np.any(neg_mask):
                    neg_colors = [(c[0]*0.5, c[1]*0.5, c[2]*0.5) for i, c in enumerate(colors) if neg_mask[i]]
                    ax.scatter(xs[neg_mask], t_stretched[neg_mask], ys[neg_mask], 
                              c=neg_colors, alpha=0.5, s=1, marker='.')
                
                # Set limits and labels
                ax.set_xlim3d(0, 1279)
                ax.set_zlim3d(0, 719)
                ax.set_ylim3d(0, t_stretched.max())
                
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Time (spectral)')
                ax.set_zlabel('Y (pixels)')
                
                # Set view
                ax.view_init(elev=30, azim=-30)
                
                # Title with compensation info
                ax.set_title(f'3D Two-Stage Compensated Events\n'
                           f'{self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm, '
                           f'a_quad={self.compensation_params["a_quad"]:.4f}')
                
                self.log_result(f"3D plot created successfully")
                self.log_result(f"Two-stage compensation applied")
                
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, self.vis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            
        except Exception as e:
            self.log_result(f"ERROR in 3D plot: {str(e)}")
            error_label = ttk.Label(self.vis_frame, text=f"3D Plot Error: {str(e)}")
            error_label.pack(expand=True)

def main():
    root = tk.Tk()
    app = ScanCompensationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()