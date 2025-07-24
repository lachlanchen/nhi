#!/usr/bin/env python3
"""
Complete GUI application for scan compensation with 3D spectral stack visualization
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
        self.root.title("Scan Compensation with 3D Spectral Stack Visualization")
        self.root.geometry("1800x1000")
        
        # Data storage
        self.events_data = None
        self.model = None
        self.compensation_params = None
        self.event_tensor_compensated = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Controls and small results
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Left side - Controls (compact)
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="File", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        
        # Parameters in compact grid
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding=5)
        param_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Row 1: Optimization
        ttk.Label(param_frame, text="Opt Bin (μs):").grid(row=0, column=0, sticky="w", padx=2)
        self.opt_bin_var = tk.DoubleVar(value=100000)
        ttk.Entry(param_frame, textvariable=self.opt_bin_var, width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(param_frame, text="Iterations:").grid(row=0, column=2, sticky="w", padx=2)
        self.iterations_var = tk.IntVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.iterations_var, width=8).grid(row=0, column=3, padx=2)
        
        # Row 2: Visualization
        ttk.Label(param_frame, text="Final Bin (μs):").grid(row=1, column=0, sticky="w", padx=2)
        self.final_bin_var = tk.DoubleVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.final_bin_var, width=10).grid(row=1, column=1, padx=2)
        
        ttk.Label(param_frame, text="Accumulate (μs):").grid(row=1, column=2, sticky="w", padx=2)
        self.accumulate_var = tk.DoubleVar(value=10000)
        ttk.Entry(param_frame, textvariable=self.accumulate_var, width=10).grid(row=1, column=3, padx=2)
        
        # Row 3: 3D controls
        ttk.Label(param_frame, text="Time Stretch:").grid(row=2, column=0, sticky="w", padx=2)
        self.time_stretch_var = tk.DoubleVar(value=10.0)
        ttk.Entry(param_frame, textvariable=self.time_stretch_var, width=10).grid(row=2, column=1, padx=2)
        
        ttk.Label(param_frame, text="XY Downsample:").grid(row=2, column=2, sticky="w", padx=2)
        self.xy_downsample_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=self.xy_downsample_var, width=8).grid(row=2, column=3, padx=2)
        
        # Row 4: Wavelength
        ttk.Label(param_frame, text="λ Min (nm):").grid(row=3, column=0, sticky="w", padx=2)
        self.min_wave_var = tk.DoubleVar(value=380)
        ttk.Entry(param_frame, textvariable=self.min_wave_var, width=10).grid(row=3, column=1, padx=2)
        
        ttk.Label(param_frame, text="λ Max (nm):").grid(row=3, column=2, sticky="w", padx=2)
        self.max_wave_var = tk.DoubleVar(value=780)
        ttk.Entry(param_frame, textvariable=self.max_wave_var, width=10).grid(row=3, column=3, padx=2)
        
        # Action buttons
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding=5)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        btn_row1 = ttk.Frame(button_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="Run Optimization", command=self.run_optimization).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        btn_row2 = ttk.Frame(button_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="Generate 3D Stack", command=self.generate_3d_vis).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Right side of top - Status and progress
        status_frame = ttk.LabelFrame(top_frame, text="Status", padding=5)
        status_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Key results display (compact)
        results_summary_frame = ttk.Frame(status_frame)
        results_summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(results_summary_frame, height=8, width=40, font=("Courier", 9))
        summary_scroll = ttk.Scrollbar(results_summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom section - Split between detailed results and visualization
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Detailed results (narrower)
        results_frame = ttk.LabelFrame(bottom_frame, text="Detailed Results", padding=5)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=25, width=50, font=("Courier", 8))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side - 3D Visualization (larger)
        self.vis_frame = ttk.LabelFrame(bottom_frame, text="3D Spectral Stack", padding=5)
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
        
    def log_summary(self, message):
        self.summary_text.insert(tk.END, message + "\n")
        self.summary_text.see(tk.END)
        self.root.update_idletasks()
        
    def run_optimization(self):
        """Run scan compensation optimization"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file first")
            return
            
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
            
        def optimization_worker():
            try:
                self.update_status("Loading events...")
                self.log_result("="*60)
                self.log_result("SCAN COMPENSATION OPTIMIZATION")
                self.log_result("="*60)
                self.log_result("Loading: " + os.path.basename(self.file_path_var.get()))
                
                # Load events
                x, y, t, p = load_npz_events(self.file_path_var.get())
                self.events_data = (x, y, t, p)
                
                self.log_result(f"Events loaded: {len(x):,}")
                self.log_result(f"Time range: {t.min():.0f} - {t.max():.0f} μs")
                self.log_result(f"Duration: {(t.max()-t.min())/1e6:.3f} seconds")
                self.log_result(f"X range: {x.min():.0f} - {x.max():.0f}")
                self.log_result(f"Y range: {y.min():.0f} - {y.max():.0f}")
                
                # Summary
                self.log_summary(f"File: {os.path.basename(self.file_path_var.get())}")
                self.log_summary(f"Events: {len(x):,}")
                self.log_summary(f"Duration: {(t.max()-t.min())/1e6:.2f} s")
                
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
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)
                
                # Training
                losses = []
                num_iterations = self.iterations_var.get()
                bin_width = self.opt_bin_var.get()
                
                self.log_result(f"\nOptimization parameters:")
                self.log_result(f"  Bin width: {bin_width/1000:.1f} ms")
                self.log_result(f"  Iterations: {num_iterations}")
                self.log_result(f"  Device: {device}")
                self.log_result(f"\nTraining progress:")
                
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    event_tensor, loss, _, _, _, _ = self.model(xs, ys, ts, ps, 720, 1280, bin_width)
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if i % 100 == 0:
                        params = self.model.params.detach().cpu()
                        self.log_result(f"  {i:4d}: Loss={loss.item():.6f}, "
                                      f"a_x={params[0].item():.3f}, a_y={params[1].item():.3f}")
                        
                        progress = (i / num_iterations) * 100
                        self.update_progress(progress)
                    
                    # Learning rate scheduling
                    if i == int(0.5 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.log_result("  >>> Learning rate reduced by 50%")
                    elif i == int(0.8 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.1
                        self.log_result("  >>> Learning rate reduced by 90%")
                
                # Save final parameters
                final_params = self.model.params.detach().cpu()
                self.compensation_params = [final_params[0].item(), final_params[1].item()]
                
                self.log_result(f"\nOPTIMIZATION COMPLETE!")
                self.log_result(f"Final parameters:")
                self.log_result(f"  a_x = {self.compensation_params[0]:.6f} μs/pixel")
                self.log_result(f"  a_y = {self.compensation_params[1]:.6f} μs/pixel")
                self.log_result(f"Final loss: {losses[-1]:.6f}")
                self.log_result(f"Loss reduction: {((losses[0]-losses[-1])/losses[0])*100:.1f}%")
                
                # Summary update
                self.log_summary(f"")
                self.log_summary(f"OPTIMIZATION RESULTS:")
                self.log_summary(f"a_x = {self.compensation_params[0]:.3f} μs/px")
                self.log_summary(f"a_y = {self.compensation_params[1]:.3f} μs/px")
                self.log_summary(f"Loss: {losses[-1]:.4f}")
                self.log_summary(f"Reduction: {((losses[0]-losses[-1])/losses[0])*100:.1f}%")
                
                self.update_status("Optimization complete")
                self.update_progress(100)
                
            except Exception as e:
                self.log_result(f"ERROR: {str(e)}")
                self.log_summary(f"ERROR: {str(e)}")
                self.update_status("Error occurred")
                messagebox.showerror("Error", f"Optimization failed: {str(e)}")
        
        thread = threading.Thread(target=optimization_worker)
        thread.daemon = True
        thread.start()
        
    def generate_3d_vis(self):
        """Generate 3D stack visualization"""
        if self.events_data is None or self.model is None:
            messagebox.showerror("Error", "Please run optimization first")
            return
            
        try:
            self.update_status("Generating 3D stack...")
            self.log_result("\n" + "="*60)
            self.log_result("3D SPECTRAL STACK VISUALIZATION")
            self.log_result("="*60)
            
            x, y, t, p = self.events_data
            
            # Apply compensation and create accumulated frames
            xs = torch.tensor(x, device=device)
            ys = torch.tensor(y, device=device)
            ts = torch.tensor(t, device=device)
            ps = torch.tensor(p, device=device)
            
            with torch.no_grad():
                event_tensor, _, _, _, _, _ = self.model(
                    xs, ys, ts, ps, 720, 1280, self.accumulate_var.get())
                
                self.event_tensor_compensated = event_tensor.detach().cpu()
            
            self.log_result(f"Accumulation bin width: {self.accumulate_var.get()} μs")
            self.log_result(f"Created {event_tensor.shape[0]} accumulated frames")
            self.log_result(f"Frame size: {event_tensor.shape[1]} x {event_tensor.shape[2]}")
            
            # Summary
            self.log_summary(f"")
            self.log_summary(f"3D STACK:")
            self.log_summary(f"Frames: {event_tensor.shape[0]}")
            self.log_summary(f"Accumulation: {self.accumulate_var.get()/1000:.1f} ms")
            self.log_summary(f"Time stretch: {self.time_stretch_var.get():.1f}x")
            self.log_summary(f"XY downsample: {self.xy_downsample_var.get()}x")
            
            # Create 3D stack plot
            self.create_3d_stack()
            
            self.update_status("3D stack complete")
            
        except Exception as e:
            self.log_result(f"ERROR generating 3D stack: {str(e)}")
            self.log_summary(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"3D stack failed: {str(e)}")
            
    def create_3d_stack(self):
        """Create 3D stack visualization of accumulated frames"""
        # Clear previous plot
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        if self.event_tensor_compensated is None:
            return
            
        try:
            # Create figure
            fig = Figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get frame stack data
            frame_stack = self.event_tensor_compensated
            num_frames, height, width = frame_stack.shape
            
            # Downsampling parameters
            xy_downsample = self.xy_downsample_var.get()
            time_stretch = self.time_stretch_var.get()
            
            # Create downsampled coordinate grids
            x_coords = list(range(0, width, xy_downsample))
            y_coords = list(range(0, height, xy_downsample))
            
            self.log_result(f"Downsampled dimensions: {len(x_coords)} x {len(y_coords)}")
            self.log_result(f"Original: {width} x {height} -> Downsampled: {len(x_coords)} x {len(y_coords)}")
            
            # Create wavelength mapping for coloring
            wavelengths = [self.min_wave_var.get() + i * (self.max_wave_var.get() - self.min_wave_var.get()) / max(1, num_frames - 1) 
                          for i in range(num_frames)]
            
            # Plot each frame as a colored surface in the stack
            for frame_idx in range(num_frames):
                try:
                    # Extract and downsample frame
                    frame = frame_stack[frame_idx]
                    frame_np = frame.detach().numpy() if hasattr(frame, 'detach') else frame
                    
                    # Downsample the frame
                    downsampled_frame = frame_np[::xy_downsample, ::xy_downsample]
                    
                    # Skip if frame is empty
                    if downsampled_frame.max() == 0:
                        continue
                    
                    # Create coordinate meshgrids
                    X, Y = [], []
                    Z = []
                    
                    for i, y_coord in enumerate(y_coords):
                        x_row = []
                        y_row = []
                        z_row = []
                        for j, x_coord in enumerate(x_coords):
                            if i < downsampled_frame.shape[0] and j < downsampled_frame.shape[1]:
                                x_row.append(x_coord)
                                y_row.append(y_coord)
                                z_row.append(frame_idx * time_stretch + downsampled_frame[i, j] * 0.1)
                        if x_row:  # Only add non-empty rows
                            X.append(x_row)
                            Y.append(y_row)
                            Z.append(z_row)
                    
                    if not X:  # Skip if no valid data
                        continue
                    
                    # Convert to numpy arrays
                    X = [[x for x in row] for row in X]
                    Y = [[y for y in row] for row in Y]
                    Z = [[z for z in row] for row in Z]
                    
                    # Get color for this wavelength
                    wavelength = wavelengths[frame_idx]
                    color = wavelength_to_rgb(wavelength)
                    
                    # Plot as wireframe/surface
                    try:
                        ax.plot_wireframe(X, Y, Z, color=color, alpha=0.6, linewidth=0.5)
                    except:
                        # Fallback: plot as scatter points
                        x_flat = [x for row in X for x in row]
                        y_flat = [y for row in Y for y in row]
                        z_flat = [z for row in Z for z in row]
                        ax.scatter(x_flat, y_flat, z_flat, c=[color], alpha=0.6, s=1)
                    
                except Exception as e:
                    self.log_result(f"Warning: Could not plot frame {frame_idx}: {e}")
                    continue
            
            # Set labels and limits
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)') 
            ax.set_zlabel('Time (spectral)')
            
            ax.set_xlim3d(0, width)
            ax.set_ylim3d(0, height)
            ax.set_zlim3d(0, num_frames * time_stretch)
            
            # Set view angle
            ax.view_init(elev=20, azim=45)
            
            # Title
            ax.set_title(f'3D Spectral Stack\n'
                        f'{num_frames} frames, {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm\n'
                        f'Downsampled {xy_downsample}x, Time stretch {time_stretch:.1f}x')
            
            self.log_result(f"3D stack plot created successfully")
            self.log_result(f"Plotted {num_frames} spectral frames")
            self.log_result(f"Wavelength range: {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm")
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, self.vis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            
        except Exception as e:
            self.log_result(f"ERROR in 3D stack plot: {str(e)}")
            # Create error display
            error_label = ttk.Label(self.vis_frame, text=f"3D Stack Error: {str(e)}")
            error_label.pack(expand=True)

def main():
    root = tk.Tk()
    app = ScanCompensationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
