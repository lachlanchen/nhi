#!/usr/bin/env python3
"""
Complete GUI application for scan compensation with 3D spectral stack visualization
Fixed bounds issues and added manual compensation controls
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
import traceback
from PIL import Image

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def wavelength_to_rgb(wavelength):
    """Convert wavelength in nm to RGB color - Fixed to ensure [0,1] range"""
    try:
        if wavelength < 380 or wavelength > 780:
            return (0.0, 0.0, 0.0)
        
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
        
        # Ensure values are in [0,1] range
        r = max(0.0, min(1.0, r * factor))
        g = max(0.0, min(1.0, g * factor))
        b = max(0.0, min(1.0, b * factor))
        
        return (r, g, b)
    except:
        return (0.5, 0.5, 0.5)

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
    
    def warp(self, x_coords, y_coords, timestamps, manual_params=None):
        """Adjust timestamps based on x and y positions"""
        if manual_params is not None:
            a_x = torch.tensor(manual_params[0], device=device, dtype=torch.float32)
            a_y = torch.tensor(manual_params[1], device=device, dtype=torch.float32)
        else:
            a_x = self.params[0]
            a_y = self.params[1]
        
        t_warped = timestamps - a_x * x_coords - a_y * y_coords
        return x_coords, y_coords, t_warped
    
    def apply_compensation(self, x_coords, y_coords, timestamps, manual_params=None):
        """Apply learned or manual compensation to events"""
        with torch.no_grad():
            return self.warp(x_coords, y_coords, timestamps, manual_params)
    
    def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width):
        """Process events through the model - fixed gradient flow"""
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
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

        # Accumulate events using scatter_add (preserves gradients)
        if len(flat_indices) > 0:
            event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        # Reshape back to (num_bins, H, W)
        event_tensor = event_tensor_flat.view(num_bins, H, W)

        # Compute variance loss
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        loss = torch.mean(variances)

        return event_tensor, loss
    
    def generate_compensated_frames(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width, manual_params=None):
        """Generate frames with manual or learned compensation parameters"""
        with torch.no_grad():
            # Apply compensation (manual or learned)
            x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps, manual_params)
            
            time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
            t_start = t_warped.min()
            t_end = t_warped.max()
            num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

            # Normalize time to bin indices
            t_norm = (t_warped - t_start) / time_bin_width
            t_indices = torch.clamp(torch.floor(t_norm).long(), 0, num_bins - 1)

            # Ensure spatial coordinates are valid
            x_indices = torch.clamp(x_warped.long(), 0, W - 1)
            y_indices = torch.clamp(y_warped.long(), 0, H - 1)

            # Create event tensor
            event_tensor = torch.zeros(num_bins, H, W, device=device, dtype=torch.float32)
            
            # Use index_put for efficient accumulation (no gradients needed)
            indices = torch.stack([t_indices, y_indices, x_indices], dim=0)
            event_tensor = event_tensor.index_put(tuple(indices), polarities, accumulate=True)

            return event_tensor

class ScanCompensationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scan Compensation with Manual Controls")
        self.root.geometry("1900x1100")
        
        # Data storage
        self.events_data = None
        self.model = None
        self.compensation_params = None
        self.event_tensor_compensated = None
        self.event_tensor_original = None
        self.output_folder = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Controls (organized in sections)
        control_frame = ttk.LabelFrame(main_frame, text="CONTROL PANEL", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # File section
        file_section = ttk.LabelFrame(control_frame, text="📁 File Selection", padding=5)
        file_section.grid(row=0, column=0, columnspan=5, sticky="ew", padx=5, pady=2)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_section, textvariable=self.file_path_var, width=80, font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_section, text="Browse NPZ File", command=self.browse_file).pack(side=tk.LEFT)
        
        # Optimization section
        opt_section = ttk.LabelFrame(control_frame, text="⚙️ Optimization Parameters", padding=5)
        opt_section.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        ttk.Label(opt_section, text="Opt Bin (μs):").grid(row=0, column=0, sticky="w", padx=2)
        self.opt_bin_var = tk.DoubleVar(value=100000)
        ttk.Entry(opt_section, textvariable=self.opt_bin_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(opt_section, text="Iterations:").grid(row=1, column=0, sticky="w", padx=2)
        self.iterations_var = tk.IntVar(value=1000)
        ttk.Entry(opt_section, textvariable=self.iterations_var, width=12).grid(row=1, column=1, padx=2)
        
        # Frame accumulation section
        frame_section = ttk.LabelFrame(control_frame, text="📊 Frame Accumulation", padding=5)
        frame_section.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        ttk.Label(frame_section, text="Accumulate (μs):").grid(row=0, column=0, sticky="w", padx=2)
        self.accumulate_var = tk.DoubleVar(value=10000)
        ttk.Entry(frame_section, textvariable=self.accumulate_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(frame_section, text="Output:").grid(row=1, column=0, sticky="w", padx=2)
        self.output_status_label = ttk.Label(frame_section, text="Auto (same as input)", font=("Arial", 8))
        self.output_status_label.grid(row=1, column=1, sticky="w", padx=2)
        
        # Manual compensation section
        manual_section = ttk.LabelFrame(control_frame, text="🎛️ Manual Compensation", padding=5)
        manual_section.grid(row=1, column=2, sticky="ew", padx=5, pady=2)
        
        ttk.Label(manual_section, text="Manual a_x:").grid(row=0, column=0, sticky="w", padx=2)
        self.manual_ax_var = tk.DoubleVar(value=0.0)
        ttk.Entry(manual_section, textvariable=self.manual_ax_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(manual_section, text="Manual a_y:").grid(row=1, column=0, sticky="w", padx=2)
        self.manual_ay_var = tk.DoubleVar(value=0.0)
        ttk.Entry(manual_section, textvariable=self.manual_ay_var, width=12).grid(row=1, column=1, padx=2)
        
        # 3D visualization section
        vis_section = ttk.LabelFrame(control_frame, text="🎨 3D Visualization", padding=5)
        vis_section.grid(row=1, column=3, sticky="ew", padx=5, pady=2)
        
        ttk.Label(vis_section, text="Time Stretch:").grid(row=0, column=0, sticky="w", padx=2)
        self.time_stretch_var = tk.DoubleVar(value=10.0)
        ttk.Entry(vis_section, textvariable=self.time_stretch_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(vis_section, text="XY Downsample:").grid(row=1, column=0, sticky="w", padx=2)
        self.xy_downsample_var = tk.IntVar(value=10)
        ttk.Entry(vis_section, textvariable=self.xy_downsample_var, width=12).grid(row=1, column=1, padx=2)
        
        # Spectral section
        spectral_section = ttk.LabelFrame(control_frame, text="🌈 Spectral Range", padding=5)
        spectral_section.grid(row=1, column=4, sticky="ew", padx=5, pady=2)
        
        ttk.Label(spectral_section, text="λ Min (nm):").grid(row=0, column=0, sticky="w", padx=2)
        self.min_wave_var = tk.DoubleVar(value=380)
        ttk.Entry(spectral_section, textvariable=self.min_wave_var, width=12).grid(row=0, column=1, padx=2)
        
        ttk.Label(spectral_section, text="λ Max (nm):").grid(row=1, column=0, sticky="w", padx=2)
        self.max_wave_var = tk.DoubleVar(value=780)
        ttk.Entry(spectral_section, textvariable=self.max_wave_var, width=12).grid(row=1, column=1, padx=2)
        
        # Action buttons section
        action_section = ttk.LabelFrame(control_frame, text="🚀 Actions", padding=5)
        action_section.grid(row=2, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        
        btn_frame1 = ttk.Frame(action_section)
        btn_frame1.pack(pady=2)
        
        ttk.Button(btn_frame1, text="1. Run Optimization", command=self.run_optimization, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame1, text="2. Use Learned Params", command=self.use_learned_params, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame1, text="3. Apply Manual Params", command=self.apply_manual_params, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame1, text="4. Generate & Save", command=self.generate_frames, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame1, text="Copy Results", command=self.copy_results, width=15).pack(side=tk.LEFT, padx=3)
        
        btn_frame2 = ttk.Frame(action_section)
        btn_frame2.pack(pady=2)
        
        ttk.Button(btn_frame2, text="Show 3D Stack", command=self.show_3d_stack, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame2, text="Show Frame Grid", command=self.show_frame_grid, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame2, text="Compare Before/After", command=self.show_comparison, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame2, text="Show Variance Plot", command=self.show_variance_plot, width=18).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame2, text="Clear Results", command=self.clear_results, width=15).pack(side=tk.LEFT, padx=3)
        
        # Configure grid weights for control frame
        for i in range(5):
            control_frame.columnconfigure(i, weight=1)
        
        # Bottom section - Three panels
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Results (30% width)
        results_frame = ttk.LabelFrame(content_frame, text="📋 DETAILED RESULTS", padding=5)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 3))
        
        self.results_text = tk.Text(results_frame, height=30, width=55, font=("Courier", 9),
                                   selectbackground="blue", selectforeground="white", wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Context menu for results
        self.results_menu = tk.Menu(self.root, tearoff=0)
        self.results_menu.add_command(label="Copy All", command=self.copy_all_results)
        self.results_menu.add_command(label="Copy Selection", command=self.copy_selected_results)
        self.results_menu.add_command(label="Clear", command=self.clear_results)
        
        def show_context_menu(event):
            try:
                self.results_menu.tk_popup(event.x_root, event.y_root)
            except:
                pass
        
        self.results_text.bind("<Button-3>", show_context_menu)
        
        # Middle panel - Status & Summary (25% width)
        status_frame = ttk.LabelFrame(content_frame, text="📊 STATUS & SUMMARY", padding=5)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=3)
        
        # Status section
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12, "bold"))
        status_label.pack(anchor=tk.W, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Summary text
        summary_label = ttk.Label(status_frame, text="Key Results:", font=("Arial", 10, "bold"))
        summary_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.summary_text = tk.Text(status_frame, height=25, width=45, font=("Courier", 10),
                                   selectbackground="blue", selectforeground="white")
        summary_scroll = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Visualization (45% width)
        self.vis_frame = ttk.LabelFrame(content_frame, text="🎨 3D VISUALIZATION", padding=5)
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(3, 0))
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select NPZ Event File",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.output_folder = os.path.dirname(filename)
            if hasattr(self, 'output_status_label'):
                self.output_status_label.config(text=f"✓ {os.path.basename(self.output_folder)}")
            self.log_summary(f"📁 Input & Output: {os.path.basename(os.path.dirname(filename))}")
            
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
        
    def copy_text_to_clipboard(self, text):
        """Copy text to clipboard"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", f"Text copied to clipboard! ({len(text)} characters)")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy: {str(e)}")
        
    def copy_results(self):
        self.copy_all_results()
        
    def copy_all_results(self):
        try:
            content = self.results_text.get(1.0, tk.END)
            self.copy_text_to_clipboard(content)
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy results: {str(e)}")
        
    def copy_selected_results(self):
        try:
            content = self.results_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.copy_text_to_clipboard(content)
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select text first")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy selection: {str(e)}")
            
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        
    def show_error_details(self, error_msg, full_traceback=""):
        """Show error details"""
        # Clear visualization area
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        
        # Create new error frame each time
        error_frame = ttk.LabelFrame(self.vis_frame, text="❌ Error Details", padding=10)
        error_frame.pack(fill=tk.BOTH, expand=True)
        
        error_text = tk.Text(error_frame, height=15, width=70, font=("Courier", 11),
                           selectbackground="red", selectforeground="white", wrap=tk.WORD)
        error_scroll = ttk.Scrollbar(error_frame, orient=tk.VERTICAL, command=error_text.yview)
        error_text.configure(yscrollcommand=error_scroll.set)
        
        error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        error_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert error details
        error_text.delete(1.0, tk.END)
        error_text.insert(tk.END, "ERROR DETAILS:\n")
        error_text.insert(tk.END, "=" * 50 + "\n\n")
        error_text.insert(tk.END, f"Error Message:\n{error_msg}\n\n")
        if full_traceback:
            error_text.insert(tk.END, f"Full Traceback:\n{full_traceback}\n")
        
        # Add copy button
        copy_btn = ttk.Button(error_frame, text="Copy Error Details", 
                             command=lambda: self.copy_text_to_clipboard(error_text.get(1.0, tk.END)))
        copy_btn.pack(pady=5)
        
    def use_learned_params(self):
        """Use the learned compensation parameters from optimization"""
        if self.compensation_params is None:
            messagebox.showerror("Error", "Please run optimization first to get learned parameters")
            return
        
        self.manual_ax_var.set(self.compensation_params[0])
        self.manual_ay_var.set(self.compensation_params[1])
        self.log_result(f"✅ Set manual params to learned values: a_x={self.compensation_params[0]:.6f}, a_y={self.compensation_params[1]:.6f}")
        self.log_summary(f"🎛️ Manual: a_x={self.compensation_params[0]:.3f}, a_y={self.compensation_params[1]:.3f}")
        
    def apply_manual_params(self):
        """Apply manual compensation parameters and generate frames"""
        if self.events_data is None:
            messagebox.showerror("Error", "Please load an NPZ file first")
            return
            
        def manual_worker():
            try:
                self.update_status("Applying manual compensation...")
                
                manual_params = [self.manual_ax_var.get(), self.manual_ay_var.get()]
                
                self.log_result(f"\n🎛️ APPLYING MANUAL COMPENSATION:")
                self.log_result(f"   Manual a_x = {manual_params[0]:.6f}")
                self.log_result(f"   Manual a_y = {manual_params[1]:.6f}")
                
                x, y, t, p = self.events_data
                xs = torch.tensor(x, device=device)
                ys = torch.tensor(y, device=device)
                ts = torch.tensor(t, device=device)
                ps = torch.tensor(p, device=device)
                
                # Initialize model if needed
                if self.model is None:
                    initial_params = torch.zeros(2, device=device, requires_grad=True)
                    self.model = ScanCompensation(initial_params)
                
                # Generate frames with manual parameters
                event_tensor_manual = self.model.generate_compensated_frames(
                    xs, ys, ts, ps, 720, 1280, self.accumulate_var.get(), manual_params=manual_params)
                self.event_tensor_compensated = event_tensor_manual.detach().cpu()
                
                # Also generate original frames for comparison
                event_tensor_original = self.model.generate_compensated_frames(
                    xs, ys, ts, ps, 720, 1280, self.accumulate_var.get(), manual_params=[0.0, 0.0])
                self.event_tensor_original = event_tensor_original.detach().cpu()
                
                num_frames = self.event_tensor_compensated.shape[0]
                
                # Calculate variance improvement
                var_orig = torch.var(self.event_tensor_original.view(num_frames, -1), dim=1).mean()
                var_comp = torch.var(self.event_tensor_compensated.view(num_frames, -1), dim=1).mean()
                improvement = ((var_orig - var_comp) / var_orig * 100).item()
                
                self.log_result(f"✅ Manual compensation applied!")
                self.log_result(f"📊 Generated {num_frames} frames")
                self.log_result(f"📈 Variance improvement: {improvement:.1f}%")
                
                self.log_summary(f"\n🎛️ MANUAL COMPENSATION:")
                self.log_summary(f"📐 a_x = {manual_params[0]:.3f}")
                self.log_summary(f"📐 a_y = {manual_params[1]:.3f}")
                self.log_summary(f"📊 Frames: {num_frames}")
                self.log_summary(f"📈 Improvement: {improvement:.1f}%")
                
                self.update_status("✅ Manual compensation complete")
                self.update_progress(100)
                
            except Exception as e:
                error_msg = str(e)
                full_traceback = traceback.format_exc()
                self.log_result(f"❌ ERROR applying manual compensation: {error_msg}")
                self.log_summary(f"❌ ERROR: {error_msg}")
                self.show_error_details(error_msg, full_traceback)
        
        thread = threading.Thread(target=manual_worker)
        thread.daemon = True
        thread.start()
        
    def run_optimization(self):
        """Run scan compensation optimization"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file first")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        
        def optimization_worker():
            try:
                self.update_status("Loading events...")
                self.log_result("="*70)
                self.log_result("SCAN COMPENSATION OPTIMIZATION")
                self.log_result("="*70)
                
                filename = os.path.basename(self.file_path_var.get())
                self.log_result(f"Loading: {filename}")
                self.log_summary(f"📁 File: {filename}")
                
                # Load events
                x, y, t, p = load_npz_events(self.file_path_var.get())
                self.events_data = (x, y, t, p)
                
                self.log_result(f"Events loaded: {len(x):,}")
                self.log_result(f"Time range: {t.min():.0f} - {t.max():.0f} μs")
                self.log_result(f"Duration: {(t.max()-t.min())/1e6:.3f} seconds")
                self.log_result(f"Sensor size: {x.max()-x.min()+1:.0f} x {y.max()-y.min()+1:.0f}")
                
                self.log_summary(f"📊 Events: {len(x):,}")
                self.log_summary(f"⏱️ Duration: {(t.max()-t.min())/1e6:.2f} s")
                
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
                
                self.log_result(f"\n🔧 Optimization Parameters:")
                self.log_result(f"   Bin width: {bin_width/1000:.1f} ms")
                self.log_result(f"   Iterations: {num_iterations}")
                self.log_result(f"   Device: {device}")
                self.log_result(f"\n🚀 Training Progress:")
                
                for i in range(num_iterations):
                    optimizer.zero_grad()
                    event_tensor, loss = self.model(xs, ys, ts, ps, 720, 1280, bin_width)
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if i % 100 == 0:
                        params = self.model.params.detach().cpu()
                        self.log_result(f"   {i:4d}: Loss={loss.item():.6f}, "
                                      f"a_x={params[0].item():.3f}, a_y={params[1].item():.3f}")
                        
                        progress = (i / num_iterations) * 100
                        self.update_progress(progress)
                    
                    if i == int(0.5 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.log_result("   >>> Learning rate reduced by 50%")
                    elif i == int(0.8 * num_iterations):
                        optimizer.param_groups[0]['lr'] *= 0.1
                        self.log_result("   >>> Learning rate reduced by 90%")
                
                # Save final parameters
                final_params = self.model.params.detach().cpu()
                self.compensation_params = [final_params[0].item(), final_params[1].item()]
                
                self.log_result(f"\n✅ OPTIMIZATION COMPLETE!")
                self.log_result(f"📐 Final Parameters:")
                self.log_result(f"   a_x = {self.compensation_params[0]:.6f} μs/pixel")
                self.log_result(f"   a_y = {self.compensation_params[1]:.6f} μs/pixel")
                self.log_result(f"📊 Final loss: {losses[-1]:.6f}")
                self.log_result(f"📈 Loss reduction: {((losses[0]-losses[-1])/losses[0])*100:.1f}%")
                
                # Show compensation effect
                time_compensation_range_x = self.compensation_params[0] * (x.max() - x.min())
                time_compensation_range_y = self.compensation_params[1] * (y.max() - y.min())
                self.log_result(f"🔧 Compensation Effect:")
                self.log_result(f"   Max time shift from X: {abs(time_compensation_range_x):.1f} μs")
                self.log_result(f"   Max time shift from Y: {abs(time_compensation_range_y):.1f} μs")
                self.log_result(f"   Total max time shift: {abs(time_compensation_range_x) + abs(time_compensation_range_y):.1f} μs")
                
                self.log_summary(f"\n✅ OPTIMIZATION RESULTS:")
                self.log_summary(f"📐 a_x = {self.compensation_params[0]:.3f} μs/px")
                self.log_summary(f"📐 a_y = {self.compensation_params[1]:.3f} μs/px")
                self.log_summary(f"📊 Loss: {losses[-1]:.4f}")
                self.log_summary(f"📈 Reduction: {((losses[0]-losses[-1])/losses[0])*100:.1f}%")
                self.log_summary(f"🔧 Max time shift: {abs(time_compensation_range_x) + abs(time_compensation_range_y):.1f} μs")
                
                self.update_status("✅ Optimization complete")
                self.update_progress(100)
                
            except Exception as e:
                error_msg = str(e)
                full_traceback = traceback.format_exc()
                self.log_result(f"❌ ERROR: {error_msg}")
                self.log_summary(f"❌ ERROR: {error_msg}")
                self.update_status("❌ Error occurred")
                self.show_error_details(error_msg, full_traceback)
        
        thread = threading.Thread(target=optimization_worker)
        thread.daemon = True
        thread.start()
        
    def generate_frames(self):
        """Generate and save frames using current compensation parameters"""
        if self.events_data is None:
            messagebox.showerror("Error", "Please load events and apply compensation first")
            return
            
        if self.event_tensor_compensated is None:
            messagebox.showerror("Error", "Please apply compensation parameters first (learned or manual)")
            return
            
        # Auto-set output folder if not selected
        if not self.output_folder:
            input_dir = os.path.dirname(self.file_path_var.get())
            self.output_folder = input_dir
            self.log_summary(f"📁 Auto-set output: {os.path.basename(input_dir)}")
            
        def frame_worker():
            try:
                self.update_status("Saving compensated frames...")
                self.log_result("\n" + "="*70)
                self.log_result("COMPENSATED FRAME SAVING")
                self.log_result("="*70)
                
                num_frames, height, width = self.event_tensor_compensated.shape
                self.log_result(f"📸 Saving {num_frames} compensated frames")
                self.log_result(f"📏 Frame size: {height} x {width}")
                
                # Create output folder structure
                base_filename = os.path.splitext(os.path.basename(self.file_path_var.get()))[0]
                
                # Use current manual params for naming
                current_ax = self.manual_ax_var.get()
                current_ay = self.manual_ay_var.get()
                
                frame_folder = os.path.join(self.output_folder, f"{base_filename}_compensated_ax{current_ax:.3f}_ay{current_ay:.3f}")
                os.makedirs(frame_folder, exist_ok=True)
                
                # Create comparison folder
                comparison_folder = os.path.join(frame_folder, "comparison_original_vs_compensated")
                os.makedirs(comparison_folder, exist_ok=True)
                
                self.log_result(f"💾 Saving to: {frame_folder}")
                
                # Create wavelength mapping
                wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
                
                for i in range(num_frames):
                    # Bounds checking
                    if i >= self.event_tensor_compensated.shape[0]:
                        self.log_result(f"⚠️ Stopping at frame {i} - out of bounds")
                        break
                        
                    # Extract compensated frame
                    frame_comp_tensor = self.event_tensor_compensated[i]
                    
                    # Get max values
                    frame_comp_max = frame_comp_tensor.max().item()
                    
                    # Convert compensated frame
                    if frame_comp_max > 0:
                        frame_comp_norm_tensor = (frame_comp_tensor / frame_comp_max * 255)
                        frame_comp_norm = frame_comp_norm_tensor.byte().cpu().numpy()
                    else:
                        frame_comp_norm = torch.zeros_like(frame_comp_tensor, dtype=torch.uint8).cpu().numpy()
                    
                    # Calculate wavelength
                    wavelength = self.min_wave_var.get() + (i / max(1, num_frames-1)) * wavelength_range
                    
                    # Save compensated frame
                    filename_comp = f"frame_{i:03d}_compensated_wavelength_{wavelength:.0f}nm.png"
                    filepath_comp = os.path.join(frame_folder, filename_comp)
                    img_comp = Image.fromarray(frame_comp_norm, mode='L')
                    img_comp.save(filepath_comp)
                    
                    # Save comparison frames if original exists
                    if self.event_tensor_original is not None and i < self.event_tensor_original.shape[0]:
                        frame_orig_tensor = self.event_tensor_original[i]
                        frame_orig_max = frame_orig_tensor.max().item()
                        
                        if frame_orig_max > 0:
                            frame_orig_norm_tensor = (frame_orig_tensor / frame_orig_max * 255)
                            frame_orig_norm = frame_orig_norm_tensor.byte().cpu().numpy()
                        else:
                            frame_orig_norm = torch.zeros_like(frame_orig_tensor, dtype=torch.uint8).cpu().numpy()
                        
                        filename_orig = f"frame_{i:03d}_original_wavelength_{wavelength:.0f}nm.png"
                        filepath_orig = os.path.join(comparison_folder, filename_orig)
                        img_orig = Image.fromarray(frame_orig_norm, mode='L')
                        img_orig.save(filepath_orig)
                        
                        filename_comp_cmp = f"frame_{i:03d}_compensated_wavelength_{wavelength:.0f}nm.png"
                        filepath_comp_cmp = os.path.join(comparison_folder, filename_comp_cmp)
                        img_comp.save(filepath_comp_cmp)
                    
                    if i % 10 == 0:
                        progress = (i / num_frames) * 100
                        self.update_progress(progress)
                        self.log_result(f"   Saved frame {i+1}/{num_frames}")
                
                # Save metadata
                metadata_file = os.path.join(frame_folder, "metadata.txt")
                with open(metadata_file, 'w') as f:
                    f.write(f"Compensated Frame Metadata\n")
                    f.write(f"==========================\n")
                    f.write(f"Source file: {os.path.basename(self.file_path_var.get())}\n")
                    f.write(f"Number of frames: {num_frames}\n")
                    f.write(f"Frame size: {height} x {width}\n")
                    f.write(f"Accumulation bin: {self.accumulate_var.get()} μs\n")
                    f.write(f"Wavelength range: {self.min_wave_var.get():.0f} - {self.max_wave_var.get():.0f} nm\n")
                    f.write(f"Compensation parameters: a_x={current_ax:.6f}, a_y={current_ay:.6f}\n")
                    f.write(f"Compensation type: {'Manual' if abs(current_ax - (self.compensation_params[0] if self.compensation_params else 0)) > 1e-6 else 'Learned'}\n")
                
                self.log_result(f"✅ All {num_frames} compensated frames saved successfully!")
                self.log_result(f"📁 Main output: {frame_folder}")
                self.log_result(f"📁 Comparison: {comparison_folder}")
                
                self.log_summary(f"\n💾 FRAMES SAVED:")
                self.log_summary(f"📸 Frames: {num_frames}")
                self.log_summary(f"📁 Folder: {os.path.basename(frame_folder)}")
                self.log_summary(f"🌈 λ: {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm")
                
                self.update_status("✅ Frames saved successfully")
                self.update_progress(100)
                
            except Exception as e:
                error_msg = str(e)
                full_traceback = traceback.format_exc()
                self.log_result(f"❌ ERROR saving frames: {error_msg}")
                self.log_summary(f"❌ ERROR: {error_msg}")
                self.show_error_details(error_msg, full_traceback)
        
        thread = threading.Thread(target=frame_worker)
        thread.daemon = True
        thread.start()
        
    def show_3d_stack(self):
        """Show 3D stack visualization using compensated events"""
        if self.event_tensor_compensated is None:
            messagebox.showerror("Error", "Please apply compensation first")
            return
            
        try:
            self.update_status("Creating 3D stack...")
            
            # Clear visualization area
            for widget in self.vis_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get compensated data
            tensor = self.event_tensor_compensated
            num_frames, height, width = tensor.shape
            
            # Downsampling for visualization
            downsample = self.xy_downsample_var.get()
            time_stretch = self.time_stretch_var.get()
            
            current_ax = self.manual_ax_var.get()
            current_ay = self.manual_ay_var.get()
            
            self.log_result(f"🎨 Creating 3D visualization...")
            self.log_result(f"   Frames: {num_frames} x {height} x {width}")
            self.log_result(f"   XY downsample: {downsample}x")
            self.log_result(f"   Time stretch: {time_stretch:.1f}x")
            self.log_result(f"   Compensation: a_x={current_ax:.3f}, a_y={current_ay:.3f}")
            
            # Wavelength mapping
            wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
            
            points_plotted = 0
            frame_step = max(1, num_frames // 30)  # Show max 30 frames
            
            for frame_idx in range(0, min(num_frames, 30*frame_step), frame_step):
                if frame_idx >= num_frames:
                    break
                    
                frame_tensor = tensor[frame_idx]
                
                # Calculate wavelength and color
                wavelength = self.min_wave_var.get() + (frame_idx / max(1, num_frames-1)) * wavelength_range
                color = wavelength_to_rgb(wavelength)
                
                # Sample points
                x_points = []
                y_points = []
                z_points = []
                
                frame_data = frame_tensor.tolist()
                
                for i in range(0, height, downsample):
                    for j in range(0, width, downsample):
                        if i < len(frame_data) and j < len(frame_data[i]):
                            intensity = frame_data[i][j]
                            if intensity > 0.1:
                                x_points.append(j)
                                y_points.append(i)
                                z_points.append(frame_idx * time_stretch)
                
                if x_points:
                    ax.scatter(x_points, y_points, z_points, c=[color], alpha=0.7, s=3)
                    points_plotted += len(x_points)
            
            # Set proper aspect ratio
            ax.set_xlim3d(0, width)
            ax.set_ylim3d(0, height)
            ax.set_zlim3d(0, num_frames * time_stretch)
            
            ax.set_box_aspect([width/height, 1.0, (num_frames * time_stretch)/(width*0.5)])
            
            # Labels and title
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_zlabel('Time (spectral)')
            
            ax.set_title(f'3D Spectral Stack (COMPENSATED)\n'
                        f'{num_frames} frames, {self.min_wave_var.get():.0f}-{self.max_wave_var.get():.0f} nm\n'
                        f'a_x={current_ax:.3f}, a_y={current_ay:.3f}\n'
                        f'Points: {points_plotted:,}')
            
            ax.view_init(elev=25, azim=45)
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, self.vis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            
            self.log_result(f"✅ 3D stack created with {points_plotted:,} points")
            self.update_status("✅ 3D stack complete")
            
        except Exception as e:
            error_msg = str(e)
            full_traceback = traceback.format_exc()
            self.log_result(f"❌ ERROR in 3D stack: {error_msg}")
            self.show_error_details(error_msg, full_traceback)
            
    def show_frame_grid(self):
        """Show all compensated frames as image grid"""
        if self.event_tensor_compensated is None:
            messagebox.showerror("Error", "Please apply compensation first")
            return
            
        try:
            self.update_status("Creating frame grid...")
            
            # Create new window for frame grid
            grid_window = tk.Toplevel(self.root)
            grid_window.title("Compensated Spectral Frames Grid")
            grid_window.geometry("1400x1000")
            
            # Create figure
            tensor = self.event_tensor_compensated
            num_frames = tensor.shape[0]
            
            # Calculate grid size
            cols = min(10, num_frames)
            rows = (num_frames + cols - 1) // cols
            
            fig = Figure(figsize=(20, rows * 2))
            
            # Wavelength mapping
            wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
            
            current_ax = self.manual_ax_var.get()
            current_ay = self.manual_ay_var.get()
            
            for i in range(min(num_frames, 50)):  # Limit to 50 frames for performance
                ax = fig.add_subplot(rows, cols, i+1)
                
                frame_tensor = tensor[i]
                wavelength = self.min_wave_var.get() + (i / max(1, num_frames-1)) * wavelength_range
                color = wavelength_to_rgb(wavelength)
                
                frame_max = frame_tensor.max().item()
                
                if frame_max > 0:
                    frame_norm_tensor = frame_tensor / frame_max
                    frame_norm = frame_norm_tensor.cpu().numpy()
                    
                    # Create colored frame
                    colored_frame = np.zeros((frame_norm.shape[0], frame_norm.shape[1], 3))
                    for c in range(3):
                        colored_frame[:, :, c] = frame_norm * color[c]
                    
                    ax.imshow(colored_frame, aspect='auto')
                else:
                    frame_display = frame_tensor.cpu().numpy()
                    ax.imshow(frame_display, cmap='gray', aspect='auto')
                
                ax.set_title(f'Frame {i}\nλ={wavelength:.0f}nm', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add overall title
            fig.suptitle(f'Compensated Spectral Frames (a_x={current_ax:.3f}, a_y={current_ay:.3f})', fontsize=14)
            
            plt.tight_layout()
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, grid_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, grid_window)
            toolbar.update()
            
            self.log_result(f"✅ Frame grid created: {min(num_frames, 50)} frames shown")
            self.update_status("✅ Frame grid complete")
            
        except Exception as e:
            error_msg = str(e)
            self.log_result(f"❌ ERROR in frame grid: {error_msg}")
            messagebox.showerror("Error", f"Frame grid failed: {error_msg}")
    
    def show_comparison(self):
        """Show side-by-side comparison of original vs compensated frames"""
        if self.event_tensor_compensated is None or self.event_tensor_original is None:
            messagebox.showerror("Error", "Please apply compensation first to enable comparison")
            return
            
        try:
            self.update_status("Creating comparison...")
            
            # Create new window for comparison
            comp_window = tk.Toplevel(self.root)
            comp_window.title("Before vs After Compensation")
            comp_window.geometry("1600x800")
            
            # Create figure with subplots
            fig = Figure(figsize=(16, 8))
            
            # Select a middle frame for comparison
            num_frames = min(self.event_tensor_compensated.shape[0], self.event_tensor_original.shape[0])
            frame_idx = num_frames // 2
            
            # Get frames
            frame_orig = self.event_tensor_original[frame_idx]
            frame_comp = self.event_tensor_compensated[frame_idx]
            
            # Calculate wavelength for this frame
            wavelength_range = self.max_wave_var.get() - self.min_wave_var.get()
            wavelength = self.min_wave_var.get() + (frame_idx / max(1, num_frames-1)) * wavelength_range
            color = wavelength_to_rgb(wavelength)
            
            current_ax = self.manual_ax_var.get()
            current_ay = self.manual_ay_var.get()
            
            # Original frame
            ax1 = fig.add_subplot(121)
            frame_orig_max = frame_orig.max().item()
            if frame_orig_max > 0:
                frame_orig_norm = (frame_orig / frame_orig_max).cpu().numpy()
                colored_frame_orig = np.zeros((frame_orig_norm.shape[0], frame_orig_norm.shape[1], 3))
                for c in range(3):
                    colored_frame_orig[:, :, c] = frame_orig_norm * color[c]
                ax1.imshow(colored_frame_orig, aspect='auto')
            else:
                ax1.imshow(frame_orig.cpu().numpy(), cmap='gray', aspect='auto')
            ax1.set_title(f'BEFORE Compensation\nFrame {frame_idx}, λ={wavelength:.0f}nm\nOriginal Events')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            
            # Compensated frame
            ax2 = fig.add_subplot(122)
            frame_comp_max = frame_comp.max().item()
            if frame_comp_max > 0:
                frame_comp_norm = (frame_comp / frame_comp_max).cpu().numpy()
                colored_frame_comp = np.zeros((frame_comp_norm.shape[0], frame_comp_norm.shape[1], 3))
                for c in range(3):
                    colored_frame_comp[:, :, c] = frame_comp_norm * color[c]
                ax2.imshow(colored_frame_comp, aspect='auto')
            else:
                ax2.imshow(frame_comp.cpu().numpy(), cmap='gray', aspect='auto')
            ax2.set_title(f'AFTER Compensation\nFrame {frame_idx}, λ={wavelength:.0f}nm\na_x={current_ax:.3f}, a_y={current_ay:.3f}')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            
            # Overall title
            fig.suptitle(f'Scan Compensation Comparison\nShowing effect of compensation parameters', fontsize=14)
            
            plt.tight_layout()
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, comp_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, comp_window)
            toolbar.update()
            
            self.log_result(f"✅ Comparison created for frame {frame_idx}")
            self.update_status("✅ Comparison complete")
            
        except Exception as e:
            error_msg = str(e)
            self.log_result(f"❌ ERROR in comparison: {error_msg}")
            messagebox.showerror("Error", f"Comparison failed: {error_msg}")
    
    def show_variance_plot(self):
        """Show variance comparison plot"""
        if self.event_tensor_compensated is None or self.event_tensor_original is None:
            messagebox.showerror("Error", "Please apply compensation first")
            return
            
        try:
            self.update_status("Creating variance plot...")
            
            # Clear visualization area
            for widget in self.vis_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # Calculate variances
            num_frames = min(self.event_tensor_compensated.shape[0], self.event_tensor_original.shape[0])
            
            var_orig_list = []
            var_comp_list = []
            
            for i in range(num_frames):
                frame_orig = self.event_tensor_original[i]
                frame_comp = self.event_tensor_compensated[i]
                
                var_orig = torch.var(frame_orig).item()
                var_comp = torch.var(frame_comp).item()
                
                var_orig_list.append(var_orig)
                var_comp_list.append(var_comp)
            
            # Plot variances
            frame_indices = list(range(num_frames))
            ax.plot(frame_indices, var_orig_list, label='Original', alpha=0.7, linewidth=2)
            ax.plot(frame_indices, var_comp_list, label='Compensated', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Variance')
            ax.set_title('Variance Comparison: Original vs Compensated')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate improvement
            mean_var_orig = np.mean(var_orig_list)
            mean_var_comp = np.mean(var_comp_list)
            improvement = ((mean_var_orig - mean_var_comp) / mean_var_orig * 100)
            
            current_ax = self.manual_ax_var.get()
            current_ay = self.manual_ay_var.get()
            
            ax.text(0.02, 0.98, f'Original mean variance: {mean_var_orig:.3f}\n'
                              f'Compensated mean variance: {mean_var_comp:.3f}\n'
                              f'Improvement: {improvement:.1f}%\n'
                              f'a_x: {current_ax:.3f}, a_y: {current_ay:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, self.vis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            
            self.log_result(f"✅ Variance plot created")
            self.log_result(f"📊 Variance improvement: {improvement:.1f}%")
            self.update_status("✅ Variance plot complete")
            
        except Exception as e:
            error_msg = str(e)
            self.log_result(f"❌ ERROR in variance plot: {error_msg}")
            self.show_error_details(error_msg, traceback.format_exc())

def main():
    root = tk.Tk()
    app = ScanCompensationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
