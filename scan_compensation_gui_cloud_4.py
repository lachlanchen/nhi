#!/usr/bin/env python3
"""
Complete GUI application for scan compensation with 3D spectral visualization
Enhanced with file caching, auto-loading, projection views, and improved defaults
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
import json

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
        self.root.title("Enhanced Scan Compensation with 3D Spectral Visualization")
        self.root.geometry("1600x1050")
        
        # Configuration file path
        self.config_file = "scan_compensation_config.json"
        
        # Data storage
        self.events_data = None
        self.model = None
        self.compensation_params = None
        self.compensated_events = None
        
        # Enhanced view perspectives with projections
        self.view_settings = {
            # 3D Perspectives
            "default": (30, -30),
            "vertical": (0, 90),
            "horizontal": (90, 0),
            "lateral": (90, 90),
            "side": (30, 18),
            "r-side": (-30, 18),
            "reverse": (-60, 18),
            "normal": (64, 18),
            "normal45": (45, 0),
            "isometric": (35, 45),
            "top-view": (90, 0),
            "front-view": (0, 0),
            "back-view": (0, 180),
            "left-view": (0, 90),
            "right-view": (0, -90),
            # Projection Views
            "proj-XY": (90, 0),      # Looking down at X-Y plane
            "proj-XT": (0, 0),       # Looking at X-Time plane  
            "proj-YT": (0, 90),      # Looking at Y-Time plane
            "proj-XY-angled": (80, 10), # Slightly angled XY view
            "proj-XT-angled": (10, 10),  # Slightly angled XT view
            "proj-YT-angled": (10, 80),  # Slightly angled YT view
        }
        
        # Load configuration
        self.load_config()
        
        self.create_widgets()
        
        # Auto-load file if cached
        self.auto_load_file()
        
    def load_config(self):
        """Load configuration from file"""
        self.config = {
            "last_file_path": "",
            "ax_value": 0.0,
            "ay_value": 0.0,
            "view_perspective": "default",
            "time_stretch": 10.0,
            "enable_training": False,
            "auto_update": True
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            self.config.update({
                "last_file_path": self.file_path_var.get(),
                "ax_value": self.ax_var.get(),
                "ay_value": self.ay_var.get(),
                "view_perspective": self.view_var.get(),
                "time_stretch": self.time_stretch_var.get(),
                "enable_training": self.enable_training_var.get(),
                "auto_update": self.auto_update_var.get()
            })
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top control panel (expanded)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # File selection - Row 1
        file_frame = ttk.LabelFrame(control_frame, text="File", padding=5)
        file_frame.grid(row=0, column=0, columnspan=5, sticky="ew", padx=2, pady=2)
        
        self.file_path_var = tk.StringVar(value=self.config["last_file_path"])
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).pack(side=tk.LEFT, padx=(0, 5))
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=(0, 5))
        auto_load_btn = ttk.Button(file_frame, text="Auto Load", command=self.auto_load_file)
        auto_load_btn.pack(side=tk.LEFT)
        
        # Manual Parameters - Row 2, Column 1
        manual_frame = ttk.LabelFrame(control_frame, text="Manual Parameters", padding=5)
        manual_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        ttk.Label(manual_frame, text="a_x (μs/px):").grid(row=0, column=0, sticky="w")
        self.ax_var = tk.DoubleVar(value=self.config["ax_value"])
        ttk.Entry(manual_frame, textvariable=self.ax_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(manual_frame, text="a_y (μs/px):").grid(row=1, column=0, sticky="w")
        self.ay_var = tk.DoubleVar(value=self.config["ay_value"])
        ttk.Entry(manual_frame, textvariable=self.ay_var, width=12).grid(row=1, column=1, padx=2)
        
        # Training Control - Row 2, Column 2
        training_frame = ttk.LabelFrame(control_frame, text="Training Control", padding=5)
        training_frame.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        self.enable_training_var = tk.BooleanVar(value=self.config["enable_training"])  # Default: False
        ttk.Checkbutton(training_frame, text="Enable Training", 
                       variable=self.enable_training_var,
                       command=self.on_training_toggle).grid(row=0, column=0, columnspan=2, sticky="w")
        
        ttk.Label(training_frame, text="Iterations:").grid(row=1, column=0, sticky="w")
        self.iterations_var = tk.IntVar(value=1000)
        self.iterations_entry = ttk.Entry(training_frame, textvariable=self.iterations_var, width=10)
        self.iterations_entry.grid(row=1, column=1, padx=2)
        
        # Optimization Parameters - Row 2, Column 3
        param_frame1 = ttk.LabelFrame(control_frame, text="Optimization", padding=5)
        param_frame1.grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        
        ttk.Label(param_frame1, text="Opt Bin (μs):").grid(row=0, column=0, sticky="w")
        self.opt_bin_var = tk.DoubleVar(value=100000)
        self.opt_bin_entry = ttk.Entry(param_frame1, textvariable=self.opt_bin_var, width=10)
        self.opt_bin_entry.grid(row=0, column=1, padx=2)
        
        ttk.Label(param_frame1, text="Final Bin (μs):").grid(row=1, column=0, sticky="w")
        self.final_bin_var = tk.DoubleVar(value=1000)
        ttk.Entry(param_frame1, textvariable=self.final_bin_var, width=10).grid(row=1, column=1, padx=2)
        
        # Visualization Parameters - Row 2, Column 4
        param_frame2 = ttk.LabelFrame(control_frame, text="Visualization", padding=5)
        param_frame2.grid(row=1, column=3, sticky="ew", padx=2, pady=2)
        
        ttk.Label(param_frame2, text="Sample Rate:").grid(row=0, column=0, sticky="w")
        self.sample_rate_var = tk.IntVar(value=1000)
        ttk.Entry(param_frame2, textvariable=self.sample_rate_var, width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(param_frame2, text="Time Stretch:").grid(row=1, column=0, sticky="w")
        self.time_stretch_var = tk.DoubleVar(value=self.config["time_stretch"])
        ttk.Entry(param_frame2, textvariable=self.time_stretch_var, width=10).grid(row=1, column=1, padx=2)
        
        # Actions - Row 2, Column 5
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding=5)
        action_frame.grid(row=1, column=4, sticky="ew", padx=2, pady=2)
        
        # Action buttons
        self.run_button = ttk.Button(action_frame, text="Set Parameters", command=self.run_optimization)
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        
        ttk.Button(action_frame, text="Generate 3D", command=self.generate_3d_vis).grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # Row 3 - Additional visualization controls
        # Wavelength Controls - Row 3, Column 1
        wave_frame = ttk.LabelFrame(control_frame, text="Wavelength (nm)", padding=5)
        wave_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        
        ttk.Label(wave_frame, text="Min λ:").grid(row=0, column=0, sticky="w")
        self.min_wave_var = tk.DoubleVar(value=380)
        ttk.Entry(wave_frame, textvariable=self.min_wave_var, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(wave_frame, text="Max λ:").grid(row=1, column=0, sticky="w")
        self.max_wave_var = tk.DoubleVar(value=780)
        ttk.Entry(wave_frame, textvariable=self.max_wave_var, width=8).grid(row=1, column=1, padx=2)
        
        # View Perspective - Row 3, Column 2-3 (spanning 2 columns for dropdown width)
        view_frame = ttk.LabelFrame(control_frame, text="View Perspective & Projections", padding=5)
        view_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Label(view_frame, text="Perspective:").grid(row=0, column=0, sticky="w")
        self.view_var = tk.StringVar(value=self.config["view_perspective"])
        view_combo = ttk.Combobox(view_frame, textvariable=self.view_var, 
                                 values=list(self.view_settings.keys()), 
                                 state="readonly", width=15)
        view_combo.grid(row=0, column=1, padx=2, columnspan=2)
        
        ttk.Label(view_frame, text="Auto Update:").grid(row=1, column=0, sticky="w")
        self.auto_update_var = tk.BooleanVar(value=self.config["auto_update"])  # Default: True
        ttk.Checkbutton(view_frame, variable=self.auto_update_var).grid(row=1, column=1, sticky="w", padx=2)
        
        # Quick projection buttons
        ttk.Button(view_frame, text="XY", command=lambda: self.set_view("proj-XY")).grid(row=1, column=2, padx=1)
        ttk.Button(view_frame, text="XT", command=lambda: self.set_view("proj-XT")).grid(row=0, column=3, padx=1)
        ttk.Button(view_frame, text="YT", command=lambda: self.set_view("proj-YT")).grid(row=1, column=3, padx=1)
        
        # Bind view change event
        view_combo.bind('<<ComboboxSelected>>', self.on_view_change)
        
        # Auto-save buttons - Row 3, Column 4-5
        save_frame = ttk.LabelFrame(control_frame, text="Configuration", padding=5)
        save_frame.grid(row=2, column=3, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(save_frame, text="Save Config", command=self.save_config).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(save_frame, text="Reset View", command=self.reset_view).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(save_frame, text="Auto Set Params", command=self.auto_set_params).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2)
        
        # Configure grid weights
        for i in range(5):
            control_frame.columnconfigure(i, weight=1)
        
        # Content area - split between results and visualization
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Results (larger space)
        results_frame = ttk.LabelFrame(content_frame, text="Results & Status", padding=5)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Status and progress
        status_frame = ttk.Frame(results_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready - Auto-loading enabled")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Results text (larger)
        self.results_text = tk.Text(results_frame, height=22, width=60, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side - Visualization
        self.vis_frame = ttk.LabelFrame(content_frame, text="3D Visualization", padding=5)
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize UI state
        self.on_training_toggle()
        
        # Log initialization
        self.log_result("Enhanced Scan Compensation GUI Initialized")
        self.log_result(f"Device: {device}")
        self.log_result(f"Configuration loaded from: {self.config_file}")
        self.log_result(f"Available perspectives: {len(self.view_settings)}")
        
    def set_view(self, view_name):
        """Set view perspective and update if auto-update is enabled"""
        self.view_var.set(view_name)
        self.on_view_change()
        
    def reset_view(self):
        """Reset to default view"""
        self.set_view("default")
        
    def auto_set_params(self):
        """Auto-set basic parameters for quick start"""
        if self.file_path_var.get():
            self.ax_var.set(0.0)
            self.ay_var.set(0.0)
            self.log_result("Parameters auto-set to default values")
            self.run_optimization()
        else:
            messagebox.showwarning("Warning", "Please select a file first")
        
    def auto_load_file(self):
        """Auto-load the cached file if it exists"""
        file_path = self.file_path_var.get()
        if file_path and os.path.exists(file_path):
            self.log_result(f"Auto-loading cached file: {os.path.basename(file_path)}")
            self.update_status("Auto-loading cached file...")
            # Auto-set parameters and load
            threading.Thread(target=self.auto_load_worker, daemon=True).start()
        elif file_path:
            self.log_result(f"Cached file not found: {file_path}")
            self.update_status("Cached file not found - please browse for file")
            
    def auto_load_worker(self):
        """Worker thread for auto-loading"""
        try:
            # Load the file data
            x, y, t, p = load_npz_events(self.file_path_var.get())
            self.events_data = (x, y, t, p)
            
            # Auto-set parameters
            manual_params = torch.tensor([self.ax_var.get(), self.ay_var.get()], 
                                       device=device, requires_grad=False)
            self.model = ScanCompensation(manual_params)
            self.compensation_params = [self.ax_var.get(), self.ay_var.get()]
            
            self.log_result(f"Auto-loaded: {len(x):,} events")
            self.log_result(f"Parameters set: ax={self.ax_var.get():.6f}, ay={self.ay_var.get():.6f}")
            self.update_status("Auto-load complete - ready for 3D visualization")
            
        except Exception as e:
            self.log_result(f"Auto-load error: {str(e)}")
            self.update_status("Auto-load failed")
        
    def on_training_toggle(self):
        """Handle training checkbox toggle"""
        training_enabled = self.enable_training_var.get()
        
        # Enable/disable training-related widgets
        state = "normal" if training_enabled else "disabled"
        self.iterations_entry.configure(state=state)
        self.opt_bin_entry.configure(state=state)
        
        # Update button text
        if training_enabled:
            self.run_button.configure(text="Run Optimization")
        else:
            self.run_button.configure(text="Set Parameters")
    
    def on_view_change(self, event=None):
        """Handle view perspective change"""
        if self.auto_update_var.get() and self.compensated_events:
            self.create_3d_plot()
            self.log_result(f"View updated to: {self.view_var.get()}")
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select NPZ Event File",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.save_config()  # Auto-save when file is selected
            self.log_result(f"File selected: {os.path.basename(filename)}")
            
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
        """Run scan compensation optimization or set manual parameters"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file first")
            return
            
        def optimization_worker():
            try:
                self.update_status("Loading events...")
                self.log_result("="*60)
                if self.enable_training_var.get():
                    self.log_result("SCAN COMPENSATION OPTIMIZATION")
                else:
                    self.log_result("MANUAL PARAMETER SETTING")
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
                
                # Convert to tensors
                xs = torch.tensor(x, device=device)
                ys = torch.tensor(y, device=device)
                ts = torch.tensor(t, device=device)
                ps = torch.tensor(p, device=device)
                
                if self.enable_training_var.get():
                    # Run optimization
                    self.update_status("Running optimization...")
                    
                    # Initialize model with manual values as starting point
                    initial_params = torch.tensor([self.ax_var.get(), self.ay_var.get()], 
                                                device=device, requires_grad=True)
                    self.model = ScanCompensation(initial_params)
                    
                    # Optimizer
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)
                    
                    # Training
                    losses = []
                    num_iterations = self.iterations_var.get()
                    bin_width = self.opt_bin_var.get()
                    
                    self.log_result(f"\nOptimization parameters:")
                    self.log_result(f"  Initial a_x: {self.ax_var.get():.6f} μs/pixel")
                    self.log_result(f"  Initial a_y: {self.ay_var.get():.6f} μs/pixel")
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
                                          f"a_x={params[0].item():.6f}, a_y={params[1].item():.6f}")
                            
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
                    
                    # Update the manual parameter fields with optimized values
                    self.ax_var.set(self.compensation_params[0])
                    self.ay_var.set(self.compensation_params[1])
                    
                else:
                    # Use manual parameters directly
                    self.update_status("Setting manual parameters...")
                    
                    # Create model with manual parameters
                    manual_params = torch.tensor([self.ax_var.get(), self.ay_var.get()], 
                                               device=device, requires_grad=False)
                    self.model = ScanCompensation(manual_params)
                    self.compensation_params = [self.ax_var.get(), self.ay_var.get()]
                    
                    self.log_result(f"\nMANUAL PARAMETERS SET:")
                    self.log_result(f"  a_x = {self.compensation_params[0]:.6f} μs/pixel")
                    self.log_result(f"  a_y = {self.compensation_params[1]:.6f} μs/pixel")
                    self.log_result(f"  No optimization performed")
                
                # Auto-save configuration
                self.save_config()
                
                self.update_status("Parameters ready for 3D visualization")
                self.update_progress(100)
                
            except Exception as e:
                self.log_result(f"ERROR: {str(e)}")
                self.update_status("Error occurred")
                messagebox.showerror("Error", f"Operation failed: {str(e)}")
        
        thread = threading.Thread(target=optimization_worker)
        thread.daemon = True
        thread.start()
        
    def generate_3d_vis(self):
        """Generate 3D visualization"""
        if self.events_data is None or self.model is None:
            messagebox.showerror("Error", "Please load events and set/optimize parameters first")
            return
            
        try:
            self.update_status("Generating 3D visualization...")
            self.log_result("\n" + "="*60)
            self.log_result("3D SPECTRAL VISUALIZATION")
            self.log_result("="*60)
            
            x, y, t, p = self.events_data
            
            # Apply compensation and get individual events
            xs = torch.tensor(x, device=device)
            ys = torch.tensor(y, device=device)
            ts = torch.tensor(t, device=device)
            ps = torch.tensor(p, device=device)
            
            with torch.no_grad():
                event_tensor, _, t_warped, x_valid, y_valid, p_valid = self.model(
                    xs, ys, ts, ps, 720, 1280, self.final_bin_var.get())
                
                # Convert compensated events to simple lists (avoid numpy issues)
                x_comp = [float(x) for x in x_valid.cpu().tolist()]
                y_comp = [float(y) for y in y_valid.cpu().tolist()]
                t_comp = [float(t) for t in t_warped.cpu().tolist()]
                p_comp = [float(p) for p in p_valid.cpu().tolist()]
                
                self.compensated_events = {
                    'x': x_comp,
                    'y': y_comp,
                    't': t_comp,
                    'p': p_comp
                }
            
            self.log_result(f"Using parameters:")
            self.log_result(f"  a_x = {self.compensation_params[0]:.6f} μs/pixel")
            self.log_result(f"  a_y = {self.compensation_params[1]:.6f} μs/pixel")
            self.log_result(f"Final bin width: {self.final_bin_var.get()} μs")
            self.log_result(f"Time bins created: {event_tensor.shape[0]}")
            self.log_result(f"Valid compensated events: {len(x_comp):,}")
            self.log_result(f"View perspective: {self.view_var.get()}")
            self.log_result(f"Time stretch factor: {self.time_stretch_var.get():.1f}")
            
            # Create 3D plot
            self.create_3d_plot()
            
            self.update_status("3D visualization complete")
            
        except Exception as e:
            self.log_result(f"ERROR generating 3D: {str(e)}")
            messagebox.showerror("Error", f"3D visualization failed: {str(e)}")
            
    def create_3d_plot(self):
        """Create 3D scatter plot with perspective control and projections"""
        # Clear previous plot
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        if not self.compensated_events:
            return
            
        try:
            # Create figure
            fig = Figure(figsize=(12, 9))
            
            # Determine if this is a projection view
            view_name = self.view_var.get()
            is_projection = view_name.startswith("proj-")
            
            if is_projection:
                ax = fig.add_subplot(111)  # 2D plot for projections
            else:
                ax = fig.add_subplot(111, projection='3d')  # 3D plot
            
            # Get event data
            xs = self.compensated_events['x']
            ys = self.compensated_events['y']
            ts = self.compensated_events['t']
            ps = self.compensated_events['p']
            
            # Sample events for performance
            sample_rate = self.sample_rate_var.get()
            if sample_rate > 1 and len(xs) > sample_rate:
                # Simple sampling by taking every Nth event
                xs_sampled = xs[::sample_rate]
                ys_sampled = ys[::sample_rate]
                ts_sampled = ts[::sample_rate] 
                ps_sampled = ps[::sample_rate]
            else:
                xs_sampled = xs
                ys_sampled = ys
                ts_sampled = ts
                ps_sampled = ps
                
            self.log_result(f"Plotting {len(xs_sampled):,} events")
            
            if len(ts_sampled) > 0:
                # Time normalization and coloring
                t_min = min(ts_sampled)
                t_max = max(ts_sampled)
                t_range = t_max - t_min if t_max > t_min else 1
                
                # Create colors for wavelength mapping
                colors = []
                wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
                
                for t in ts_sampled:
                    # Normalize time to [0, 1]
                    t_norm = (t - t_min) / t_range
                    # Map to wavelength
                    wavelength = self.min_wave_var.get() + t_norm * wavelength_range
                    rgb = wavelength_to_rgb(wavelength)
                    colors.append(rgb)
                
                # Separate positive and negative events
                pos_indices = [i for i, p in enumerate(ps_sampled) if p > 0]
                neg_indices = [i for i, p in enumerate(ps_sampled) if p <= 0]
                
                # Time stretching for visualization (using user input)
                time_stretch = self.time_stretch_var.get()
                t_stretched = [(t - t_min) * time_stretch / 1e6 for t in ts_sampled]
                
                if is_projection:
                    # Handle 2D projections
                    if "XY" in view_name:
                        # X-Y projection
                        if pos_indices:
                            pos_x = [xs_sampled[i] for i in pos_indices]
                            pos_y = [ys_sampled[i] for i in pos_indices]
                            pos_colors = [colors[i] for i in pos_indices]
                            ax.scatter(pos_x, pos_y, c=pos_colors, alpha=0.7, s=2, marker='.')
                        if neg_indices:
                            neg_x = [xs_sampled[i] for i in neg_indices]
                            neg_y = [ys_sampled[i] for i in neg_indices]
                            neg_colors = [(c[0]*0.5, c[1]*0.5, c[2]*0.5) for i, c in enumerate(colors) if i in neg_indices]
                            ax.scatter(neg_x, neg_y, c=neg_colors, alpha=0.5, s=1, marker='.')
                        ax.set_xlim(0, 1279)
                        ax.set_ylim(0, 719)
                        ax.set_xlabel('X (pixels)')
                        ax.set_ylabel('Y (pixels)')
                        ax.set_title(f'X-Y Projection: Spatial Events\n{len(xs_sampled):,} events')
                        
                    elif "XT" in view_name:
                        # X-Time projection
                        if pos_indices:
                            pos_x = [xs_sampled[i] for i in pos_indices]
                            pos_t = [t_stretched[i] for i in pos_indices]
                            pos_colors = [colors[i] for i in pos_indices]
                            ax.scatter(pos_x, pos_t, c=pos_colors, alpha=0.7, s=2, marker='.')
                        if neg_indices:
                            neg_x = [xs_sampled[i] for i in neg_indices]
                            neg_t = [t_stretched[i] for i in neg_indices]
                            neg_colors = [(c[0]*0.5, c[1]*0.5, c[2]*0.5) for i, c in enumerate(colors) if i in neg_indices]
                            ax.scatter(neg_x, neg_t, c=neg_colors, alpha=0.5, s=1, marker='.')
                        ax.set_xlim(0, 1279)
                        ax.set_ylim(0, max(t_stretched) if t_stretched else 1)
                        ax.set_xlabel('X (pixels)')
                        ax.set_ylabel('Time (spectral)')
                        ax.set_title(f'X-Time Projection: Spectral Scan\n{len(xs_sampled):,} events')
                        
                    elif "YT" in view_name:
                        # Y-Time projection
                        if pos_indices:
                            pos_y = [ys_sampled[i] for i in pos_indices]
                            pos_t = [t_stretched[i] for i in pos_indices]
                            pos_colors = [colors[i] for i in pos_indices]
                            ax.scatter(pos_y, pos_t, c=pos_colors, alpha=0.7, s=2, marker='.')
                        if neg_indices:
                            neg_y = [ys_sampled[i] for i in neg_indices]
                            neg_t = [t_stretched[i] for i in neg_indices]
                            neg_colors = [(c[0]*0.5, c[1]*0.5, c[2]*0.5) for i, c in enumerate(colors) if i in neg_indices]
                            ax.scatter(neg_y, neg_t, c=neg_colors, alpha=0.5, s=1, marker='.')
                        ax.set_xlim(0, 719)
                        ax.set_ylim(0, max(t_stretched) if t_stretched else 1)
                        ax.set_xlabel('Y (pixels)')
                        ax.set_ylabel('Time (spectral)')
                        ax.set_title(f'Y-Time Projection: Spectral Scan\n{len(xs_sampled):,} events')
                
                else:
                    # Handle 3D views
                    # Plot using the same axis arrangement as the provided code
                    # (x, time_stretched, y) to match the reference visualization
                    
                    # Plot positive events
                    if pos_indices:
                        pos_x = [xs_sampled[i] for i in pos_indices]
                        pos_y = [ys_sampled[i] for i in pos_indices]
                        pos_t = [t_stretched[i] for i in pos_indices]
                        pos_colors = [colors[i] for i in pos_indices]
                        
                        ax.scatter(pos_x, pos_t, pos_y, c=pos_colors, alpha=0.7, s=2, marker='.')
                    
                    # Plot negative events (dimmer)
                    if neg_indices:
                        neg_x = [xs_sampled[i] for i in neg_indices]
                        neg_y = [ys_sampled[i] for i in neg_indices]
                        neg_t = [t_stretched[i] for i in neg_indices]
                        neg_colors = [(c[0]*0.5, c[1]*0.5, c[2]*0.5) for i, c in enumerate(colors) if i in neg_indices]
                        
                        ax.scatter(neg_x, neg_t, neg_y, c=neg_colors, alpha=0.5, s=1, marker='.')
                    
                    # Set limits and labels (matching the reference code layout)
                    ax.set_xlim3d(0, 1279)
                    ax.set_zlim3d(0, 719)
                    ax.set_ylim3d(0, max(t_stretched) if t_stretched else 1)
                    
                    ax.set_xlabel('X (pixels)')
                    ax.set_ylabel('Time (spectral)')
                    ax.set_zlabel('Y (pixels)')
                    
                    # Apply selected view perspective
                    elev, azim = self.view_settings.get(view_name, (30, -30))
                    ax.view_init(elev=elev, azim=azim)
                    
                    # Title with parameter info
                    mode = "Optimized" if self.enable_training_var.get() else "Manual"
                    ax.set_title(f'3D Compensated Events - {mode} ({view_name})\n'
                               f'ax={self.compensation_params[0]:.3f}, ay={self.compensation_params[1]:.3f}, '
                               f'stretch={time_stretch:.1f}, {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm, '
                               f'{len(xs_sampled):,} events')
                
                self.log_result(f"3D plot created successfully with {view_name} view")
                self.log_result(f"Wavelength range: {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm")
                
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, self.vis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            
        except Exception as e:
            self.log_result(f"ERROR in 3D plot: {str(e)}")
            # Create a simple text display instead
            error_label = ttk.Label(self.vis_frame, text=f"3D Plot Error: {str(e)}")
            error_label.pack(expand=True)

def main():
    root = tk.Tk()
    app = ScanCompensationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()