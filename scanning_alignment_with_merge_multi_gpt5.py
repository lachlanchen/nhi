#!/usr/bin/env python3
"""
Standalone Multi-Window Scan Compensation (Single-file OR Batched-by-File)

What this does
--------------
• If you pass a *single NPZ file* (no --merge): trains exactly like the
  previous "without_batch" script — processes ALL events with chunking and
  unified variance for proper gradients.

• If you pass a *folder* with --merge: treats each Scan_*_events.npz as
  an independent BATCH. It runs the same per-file pipeline, accumulates
  gradients across batches (no graph kept across files), and applies a single
  optimizer + smoothness regularization across the unified parameters.
  This drastically reduces peak memory for huge multi-file datasets.

All code is self-contained — no imports from other files.

Notes
-----
- Time is assumed in microseconds (μs). bin_width is in μs, too.
- For segmented scans, each NPZ is expected to contain optional metadata:
  start_time, duration_us, direction. If present, the loader will:
    - normalize time to each segment’s [0, duration_us]
    - flip time and polarity for Backward segments to align directions
- For single-file training (no --merge), we do NOT depend on that metadata —
  we just load x,y,t,p directly (like the original "without_batch" script).
- Use environment var PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to
  help reduce CUDA fragmentation (optional, but helpful for very large runs).
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # (not required, but commonly handy)
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Global defaults
# --------------------------------------------------------------------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------------------
# Core model (identical behavior to the "without_batch" advanced multi-window model)
# --------------------------------------------------------------------------------------
class Compensate(nn.Module):
    def __init__(
        self,
        a_params,
        b_params,
        duration,
        num_params=13,
        temperature=5000,
        device='cpu',
        a_fixed=True,
        b_fixed=False,
        boundary_trainable=False,
        debug=False
    ):
        """
        3D Multi-window compensation with gradients.
        - Chunked throughout to avoid OOM
        - Supports fixing a_params and/or b_params
        - Optional trainable boundary offsets (monotonic via cumsum(abs))
        """
        super().__init__()
        self.device = device
        self.duration = float(duration)
        self.temperature = float(temperature)
        self.a_fixed = a_fixed
        self.b_fixed = b_fixed
        self.boundary_trainable = boundary_trainable
        self.debug = debug
        self.num_params = num_params

        if len(a_params) != len(b_params):
            raise ValueError("a_params and b_params must have the same length")
        if len(a_params) != num_params:
            raise ValueError(f"Parameter arrays must have length {num_params}, got {len(a_params)}")

        # a params
        if a_fixed:
            self.a_params = torch.tensor(a_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('a_params_buffer', self.a_params)
            if debug:
                print(f"  a_params: FIXED (example {a_params[0]:.3f})")
        else:
            self.a_params = nn.Parameter(torch.tensor(a_params, dtype=torch.float32, device=device))
            if debug:
                print(f"  a_params: TRAINABLE")

        # b params
        if b_fixed:
            self.b_params = torch.tensor(b_params, dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('b_params_buffer', self.b_params)
            if debug:
                print(f"  b_params: FIXED (example {b_params[0]:.3f})")
        else:
            self.b_params = nn.Parameter(torch.tensor(b_params, dtype=torch.float32, device=device))
            if debug:
                print(f"  b_params: TRAINABLE")

        self.num_boundaries = len(a_params)
        self.num_main_windows = self.num_boundaries - 3
        self.num_total_windows = self.num_main_windows + 2
        if self.num_main_windows <= 0:
            raise ValueError(f"Need at least 4 parameters for 1 main window, got {self.num_boundaries}")

        self.main_window_size = self.duration / self.num_main_windows

        if boundary_trainable:
            initial_raw = torch.tensor([self.main_window_size] * self.num_boundaries,
                                       dtype=torch.float32, device=device)
            initial_raw[0] = -self.main_window_size
            self.raw_boundary_params = nn.Parameter(initial_raw)
            if debug:
                print("  boundary_offsets: TRAINABLE (monotonic via cumsum(abs))")
        else:
            boundary_offsets = torch.tensor([(i - 1) * self.main_window_size for i in range(self.num_boundaries)],
                                            dtype=torch.float32, device=device, requires_grad=False)
            self.register_buffer('fixed_boundary_offsets', boundary_offsets)
            if debug:
                print("  boundary_offsets: FIXED")

        if debug:
            device_str = "GPU" if str(self.device).startswith("cuda") else "CPU"
            trainable_params = (0 if a_fixed else len(a_params)) + (0 if b_fixed else len(b_params)) + (len(a_params) if boundary_trainable else 0)
            total_params = len(a_params) * (3 if boundary_trainable else 2)
            print(f"  Compensate ready: {self.num_main_windows} main windows, {device_str}, temp={self.temperature:.0f} μs")
            print(f"  Trainable parameters: {trainable_params}/{total_params}")

    @property
    def boundary_offsets(self):
        if self.boundary_trainable:
            abs_params = torch.abs(self.raw_boundary_params)
            cumulative_offsets = torch.cumsum(abs_params, dim=0)
            return cumulative_offsets + self.raw_boundary_params[0] - abs_params[0]
        else:
            return self.fixed_boundary_offsets

    def get_a_params(self):
        return self.a_params_buffer if self.a_fixed else self.a_params

    def get_b_params(self):
        return self.b_params_buffer if self.b_fixed else self.b_params

    def get_boundary_surfaces(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        a_params = self.get_a_params()
        b_params = self.get_b_params()

        if x.dim() == 1 and y.dim() == 1:
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return (
                a_params[:, None, None] * X[None, :, :] +
                b_params[:, None, None] * Y[None, :, :] +
                self.boundary_offsets[:, None, None]
            )
        else:
            raise ValueError("x and y must be 1D tensors/arrays")

    def forward(self, x, y, t, chunk_size=500_000, debug=False):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor): y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, dtype=torch.float32, device=self.device)
        t_shifted = t - torch.min(t)
        return self.compute_event_compensation(x, y, t_shifted, chunk_size=chunk_size, debug=debug)

    def compute_event_compensation(self, x, y, t_shifted, chunk_size=500_000, debug=False):
        a_params = self.get_a_params()
        b_params = self.get_b_params()

        num_events = len(x)
        outs = []

        if debug:
            print(f"Computing compensation in chunks of {chunk_size:,} events...")

        for start in range(0, num_events, chunk_size):
            end = min(start + chunk_size, num_events)

            x_ch = x[start:end]
            y_ch = y[start:end]
            t_ch = t_shifted[start:end]

            # [B, Nch]
            aE = a_params[:, None].expand(-1, end - start)
            bE = b_params[:, None].expand(-1, end - start)
            oE = self.boundary_offsets[:, None].expand(-1, end - start)

            xE = x_ch[None, :].expand(self.num_boundaries, -1)
            yE = y_ch[None, :].expand(self.num_boundaries, -1)
            bounds = aE * xE + bE * yE + oE  # [B, Nch]

            tE = t_ch[None, :].expand(self.num_total_windows, -1)  # [W, Nch]
            lo = bounds[:-1, :]
            hi = bounds[1:, :]

            ls = torch.sigmoid((tE - lo) / self.temperature)
            us = torch.sigmoid((hi - tE) / self.temperature)
            m = ls * us
            m_sum = torch.clamp(torch.sum(m, dim=0, keepdim=True), min=1e-8)
            m = m / m_sum

            w = torch.clamp(hi - lo, min=1e-8)
            alpha = (tE - lo) / w

            aL = a_params[:-1, None].expand(-1, end - start)
            aU = a_params[1:, None].expand(-1, end - start)
            bL = b_params[:-1, None].expand(-1, end - start)
            bU = b_params[1:, None].expand(-1, end - start)

            slope_a = (1 - alpha) * aL + alpha * aU
            slope_b = (1 - alpha) * bL + alpha * bU

            comp = torch.sum(m * slope_a * x_ch[None, :], dim=0) + \
                   torch.sum(m * slope_b * y_ch[None, :], dim=0)
            outs.append(comp)

            # cleanup
            del aE, bE, oE, xE, yE, bounds, tE, lo, hi, ls, us, m, m_sum, w, aL, aU, bL, bU, alpha, slope_a, slope_b, comp
            torch.cuda.empty_cache()

            if debug and ((start // chunk_size) % 20 == 0):
                print(f"  Compensation: {end:,}/{num_events:,} ({100.0 * end / num_events:.1f}%)")

        return torch.cat(outs, dim=0)


class ScanCompensation(nn.Module):
    def __init__(
        self,
        duration,
        num_params=13,
        device='cuda',
        a_fixed=True,
        b_fixed=False,
        boundary_trainable=False,
        a_default=0.0,
        b_default=-76.0,
        temperature=5000,
        debug=False
    ):
        super().__init__()
        if debug:
            print(f"Initializing Multi-Window Compensation:")
            print(f"  Duration: {duration/1000:.1f} ms, Parameters: {num_params*3 if boundary_trainable else num_params*2} total")
            print(f"  a_fixed: {a_fixed}, b_fixed: {b_fixed}, boundary_trainable: {boundary_trainable}")
            print(f"  a_default: {a_default}, b_default: {b_default}, temperature: {temperature}")

        a_params = [a_default] * num_params
        b_params = [b_default] * num_params
        self.compensate = Compensate(
            a_params, b_params, duration, num_params=num_params,
            device=device, a_fixed=a_fixed, b_fixed=b_fixed,
            boundary_trainable=boundary_trainable, temperature=temperature, debug=debug
        )

    def warp(self, x_coords, y_coords, timestamps):
        compensation = self.compensate(x_coords, y_coords, timestamps)
        t_warped = timestamps - compensation
        return x_coords, y_coords, t_warped

    def forward(
        self,
        x_coords, y_coords, timestamps, polarities,
        H, W, bin_width,
        original_t_start=None, original_t_end=None,
        chunk_size=500_000, debug=False
    ):
        # Respect a fixed original time range if provided (keeps tensors comparable)
        if original_t_start is not None and original_t_end is not None:
            valid = (timestamps >= original_t_start) & (timestamps <= original_t_end)
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            timestamps = timestamps[valid]
            polarities = polarities[valid]
            t_start = original_t_start
            t_end = original_t_end
        else:
            t_start = timestamps.min()
            t_end = timestamps.max()

        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        complete = torch.zeros(num_bins, H, W, device=device, dtype=torch.float32)
        N = len(x_coords)

        if debug:
            print(f"Processing {N:,} events in chunks of {chunk_size:,} for variance...")

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            x_ch = x_coords[s:e]
            y_ch = y_coords[s:e]
            t_ch = timestamps[s:e]
            p_ch = polarities[s:e]

            comp_ch = self.compensate(x_ch, y_ch, t_ch, chunk_size=chunk_size, debug=debug)
            t_warped = t_ch - comp_ch

            t_norm = (t_warped - t_start) / time_bin_width
            t0 = torch.floor(t_norm)
            t1 = t0 + 1
            wt = (t_norm - t0).float()
            t0c = t0.clamp(0, num_bins - 1)
            t1c = t1.clamp(0, num_bins - 1)

            xi = x_ch.long()
            yi = y_ch.long()
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            xi = xi[valid]; yi = yi[valid]
            t0c = t0c[valid]; t1c = t1c[valid]
            wt = wt[valid]; p_ch = p_ch[valid]

            spatial = (yi * W + xi).long()
            num_elements = num_bins * H * W

            flat_t0 = (t0c * (H * W) + spatial).long()
            flat_t1 = (t1c * (H * W) + spatial).long()
            w0 = ((1 - wt) * p_ch).float()
            w1 = (wt * p_ch).float()

            flat_idx = torch.cat([flat_t0, flat_t1], dim=0)
            flat_wts = torch.cat([w0, w1], dim=0)
            valid_flat = (flat_idx >= 0) & (flat_idx < num_elements)
            flat_idx = flat_idx[valid_flat]; flat_wts = flat_wts[valid_flat]

            flat_tensor = torch.zeros(num_elements, device=device, dtype=torch.float32)
            if len(flat_idx) > 0:
                flat_tensor = flat_tensor.scatter_add(0, flat_idx, flat_wts)

            complete += flat_tensor.view(num_bins, H, W)

            # cleanup
            del x_ch, y_ch, t_ch, p_ch, comp_ch, t_warped, t_norm, t0, t1, wt, t0c, t1c, xi, yi, spatial
            del num_elements, flat_t0, flat_t1, w0, w1, flat_idx, flat_wts, flat_tensor
            torch.cuda.empty_cache()

            if debug and ((s // chunk_size) % 10 == 0):
                print(f"  Chunk: {e:,}/{N:,} ({100.0 * e / N:.1f}%)")

        variances = torch.var(complete.view(num_bins, -1), dim=1)
        total_loss = torch.sum(variances)
        return complete, total_loss

# --------------------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------------------
def load_npz_events(npz_path, debug=False):
    """Single-file loader: x,y,t,p only — like the original 'without_batch' script."""
    if debug:
        print(f"Loading events from: {npz_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)

    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2
        if debug:
            print("  Converted polarity [0,1] -> [-1,1]")

    if debug:
        print(f"✓ Loaded {len(x):,} events")
        print(f"  Time: {t.min():.0f} - {t.max():.0f} μs ({(t.max()-t.min())/1e6:.3f}s)")
        print(f"  Spatial: X=[{x.min():.0f}, {x.max():.0f}], Y=[{y.min():.0f}, {y.max():.0f}]")
    return x, y, t, p


def load_single_segment_with_metadata(npz_path, debug=False):
    """
    Segment loader for batched-by-file training.
    Handles optional metadata: start_time, duration_us, direction.
    Returns x,y,t_norm,p where t_norm is in [0, duration] (Forward-only),
    and polarity is in [-1,1] with Backward scans flipped.
    """
    if debug:
        print(f"  Loading segment: {npz_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    t = data['t'].astype(np.float32)
    p = data['p'].astype(np.float32)

    start_time = float(data['start_time']) if 'start_time' in data else float(t.min())
    duration = float(data['duration_us']) if 'duration_us' in data else float(t.max() - t.min())
    direction = str(data['direction']) if 'direction' in data else 'Forward'

    # Normalize to [0, duration]
    t_norm = t - start_time

    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2

    if 'Backward' in direction:
        t_norm = duration - t_norm
        p = -p

    order = np.argsort(t_norm)
    x = x[order]; y = y[order]; t_norm = t_norm[order]; p = p[order]

    if debug:
        print(f"    → {len(x):,} events, duration≈{duration/1000:.1f} ms, dir={direction}")
    return x, y, t_norm, p


def list_batches(input_folder, debug=False):
    """Return sorted list of Scan_*_events.npz under a folder."""
    pattern = os.path.join(input_folder, "Scan_*_events.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No segment files found under folder: {input_folder}")
    if debug:
        print(f"Found {len(files)} NPZ segments (batched): {input_folder}")
    return files

# --------------------------------------------------------------------------------------
# Single-file training (unified – same as "without_batch")
# --------------------------------------------------------------------------------------
def train_scan_compensation(
    x, y, t, p,
    sensor_width=1280,
    sensor_height=720,
    bin_width=1e5,
    num_iterations=1000,
    learning_rate=1.0,
    debug=False,
    smoothness_weight=0.001,
    a_fixed=True,
    b_fixed=False,
    boundary_trainable=False,
    a_default=0.0,
    b_default=-76.0,
    num_params=13,
    temperature=5000,
    chunk_size=250_000,
):
    if debug:
        print(f"Training multi-window scan compensation (SINGLE FILE)...")
        print(f"  Sensor: {sensor_width} x {sensor_height}")
        print(f"  Bin width: {bin_width/1000:.1f} ms")
        print(f"  Iterations: {num_iterations}")
        print(f"  LR: {learning_rate}, Smoothness: {smoothness_weight}")
        print(f"  a_fixed={a_fixed}, b_fixed={b_fixed}, boundary_trainable={boundary_trainable}")
        print(f"  Defaults: a={a_default}, b={b_default}, num_params={num_params}, temp={temperature}, chunk={chunk_size:,}")

    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    original_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
    original_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
    duration = (original_t_end - original_t_start).item()

    model = ScanCompensation(
        duration, num_params=num_params, device=device,
        a_fixed=a_fixed, b_fixed=b_fixed, boundary_trainable=boundary_trainable,
        a_default=a_default, b_default=b_default, temperature=temperature, debug=debug
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        print("\n⚠️  NO TRAINABLE PARAMETERS — SKIPPING OPTIMIZATION, doing evaluation only.")
        with torch.no_grad():
            _, loss_eval = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                 original_t_start, original_t_end, chunk_size=chunk_size, debug=debug)
        losses = [loss_eval.item()] * num_iterations
        variance_losses = [loss_eval.item()] * num_iterations
        smoothness_losses = [0.0] * num_iterations
        a_hist = []
        b_hist = []
        final_a = model.compensate.get_a_params().detach().cpu().numpy()
        final_b = model.compensate.get_b_params().detach().cpu().numpy()
        for _ in range(num_iterations):
            a_hist.append(final_a.copy())
            b_hist.append(final_b.copy())
        return model, losses, variance_losses, smoothness_losses, a_hist, b_hist, original_t_start, original_t_end

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    print(f"Training Progress (ALL {len(x):,} events, chunked processing, unified variance):")
    print(f"{'Iter':<6} {'Total Loss':<12} {'Var Loss':<12} {'Smooth Loss':<12} {'a_range':<18} {'b_range':<18}")
    print("-" * 90)

    losses, variance_losses, smoothness_losses = [], [], []
    a_hist, b_hist = [], []

    for it in range(num_iterations):
        optimizer.zero_grad()

        _, var_loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                            original_t_start, original_t_end, chunk_size=chunk_size, debug=debug)

        smoothness = torch.tensor(0.0, device=device)
        if not a_fixed:
            a_params = model.compensate.get_a_params()
            smoothness += torch.mean((a_params[1:] - a_params[:-1]) ** 2)
        if not b_fixed:
            b_params = model.compensate.get_b_params()
            smoothness += torch.mean((b_params[1:] - b_params[:-1]) ** 2)
        if boundary_trainable:
            bo = model.compensate.boundary_offsets
            smoothness += torch.mean((bo[1:] - bo[:-1]) ** 2)

        total = var_loss + smoothness_weight * smoothness
        total.backward()
        optimizer.step()

        cur_a = model.compensate.get_a_params().detach().cpu().numpy().copy()
        cur_b = model.compensate.get_b_params().detach().cpu().numpy().copy()

        losses.append(total.item())
        variance_losses.append(var_loss.item())
        smoothness_losses.append(smoothness.item())
        a_hist.append(cur_a)
        b_hist.append(cur_b)

        if it % 100 == 0 or it == num_iterations - 1:
            a_range = f"[{cur_a.min():.3f}, {cur_a.max():.3f}]"
            b_range = f"[{cur_b.min():.3f}, {cur_b.max():.3f}]"
            print(f"{it:<6} {losses[-1]:<12.6f} {variance_losses[-1]:<12.6f} {smoothness_losses[-1]:<12.6f} {a_range:<18} {b_range:<18}")

        # simple schedule
        if it == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            if debug:
                print(f"  → LR reduced to {optimizer.param_groups[0]['lr']:.4f}")
        elif it == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            if debug:
                print(f"  → LR reduced to {optimizer.param_groups[0]['lr']:.4f}")

    print("-" * 90)
    print("Training completed!")
    print(f"  Final total loss: {losses[-1]:.6f}")
    print(f"  Final variance loss: {variance_losses[-1]:.6f}")
    print(f"  Final smoothness loss: {smoothness_losses[-1]:.6f}")

    return model, losses, variance_losses, smoothness_losses, a_hist, b_hist, original_t_start, original_t_end

# --------------------------------------------------------------------------------------
# Batched-by-file training (gradient accumulation over NPZ files)
# --------------------------------------------------------------------------------------
def train_batched_by_file(
    batch_files,
    sensor_width=1280,
    sensor_height=720,
    bin_width=1e5,
    num_iterations=1000,
    learning_rate=1.0,
    debug=False,
    smoothness_weight=0.001,
    a_fixed=True,
    b_fixed=False,
    boundary_trainable=False,
    a_default=0.0,
    b_default=-76.0,
    num_params=13,
    temperature=5000,
    chunk_size=250_000,
):
    """
    Treat each NPZ segment as a separate batch.
    For each iteration:
      loop over files -> compute variance loss -> backward (accumulate)
      then add smoothness reg once -> backward -> optimizer.step()
    """
    # Use first batch duration to initialize the model
    x0, y0, t0, p0 = load_single_segment_with_metadata(batch_files[0], debug=debug)
    orig_t_start0 = float(t0.min())
    orig_t_end0 = float(t0.max())
    duration0 = orig_t_end0 - orig_t_start0

    if debug:
        print(f"Initial duration from first batch: {duration0/1000:.1f} ms")

    model = ScanCompensation(
        duration0, num_params=num_params, device=device,
        a_fixed=a_fixed, b_fixed=b_fixed, boundary_trainable=boundary_trainable,
        a_default=a_default, b_default=b_default, temperature=temperature, debug=debug
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        print("\n⚠️  NO TRAINABLE PARAMETERS — SKIPPING OPTIMIZATION, doing evaluation only (batched).")
        # One pass evaluation to get a variance value
        total_variance_value = 0.0
        for npz_path in batch_files:
            x, y, t, p = load_single_segment_with_metadata(npz_path, debug=False)
            xs = torch.tensor(x, device=device, dtype=torch.float32)
            ys = torch.tensor(y, device=device, dtype=torch.float32)
            ts = torch.tensor(t, device=device, dtype=torch.float32)
            ps = torch.tensor(p, device=device, dtype=torch.float32)
            orig_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
            orig_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)
            with torch.no_grad():
                _, var_loss = model(xs, ys, ts, ps, sensor_height, sensor_width, bin_width,
                                    orig_t_start, orig_t_end, chunk_size=chunk_size, debug=False)
            total_variance_value += float(var_loss.detach().cpu())
            del xs, ys, ts, ps, orig_t_start, orig_t_end
            torch.cuda.empty_cache()

        losses = [total_variance_value] * num_iterations
        variance_losses = [total_variance_value] * num_iterations
        smoothness_losses = [0.0] * num_iterations
        a_hist, b_hist = [], []
        final_a = model.compensate.get_a_params().detach().cpu().numpy()
        final_b = model.compensate.get_b_params().detach().cpu().numpy()
        for _ in range(num_iterations):
            a_hist.append(final_a.copy())
            b_hist.append(final_b.copy())

        original_t_start0 = torch.tensor(orig_t_start0, device=device, dtype=torch.float32)
        original_t_end0 = torch.tensor(orig_t_end0, device=device, dtype=torch.float32)
        return model, losses, variance_losses, smoothness_losses, a_hist, b_hist, original_t_start0, original_t_end0

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    print("Training Progress (Batched-by-File, chunked, unified across batches):")
    print(f"Batches: {len(batch_files)} | Sensor: {sensor_width}x{sensor_height} | bin={bin_width/1000:.1f} ms")
    print(f"{'Iter':<6} {'Total Loss':<12} {'Var Loss':<12} {'Smooth Loss':<12} {'a_range':<18} {'b_range':<18}")
    print("-" * 90)

    losses = []
    variance_losses = []
    smoothness_losses = []
    a_hist = []
    b_hist = []

    for it in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        total_variance_value = 0.0

        # Accumulate variance gradients file-by-file
        for bi, npz_path in enumerate(batch_files):
            x, y, t, p = load_single_segment_with_metadata(npz_path, debug=(debug and it == 0 and bi < 2))
            xs = torch.tensor(x, device=device, dtype=torch.float32)
            ys = torch.tensor(y, device=device, dtype=torch.float32)
            ts = torch.tensor(t, device=device, dtype=torch.float32)
            ps = torch.tensor(p, device=device, dtype=torch.float32)

            orig_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
            orig_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)

            _, var_loss = model(xs, ys, ts, ps,
                                sensor_height, sensor_width, bin_width,
                                orig_t_start, orig_t_end,
                                chunk_size=chunk_size, debug=False)

            var_loss.backward()
            total_variance_value += float(var_loss.detach().cpu())

            del xs, ys, ts, ps, orig_t_start, orig_t_end, var_loss
            torch.cuda.empty_cache()

        # Smoothness reg applied once per iteration
        smoothness_loss = torch.tensor(0.0, device=device)
        if not a_fixed:
            a_params = model.compensate.get_a_params()
            smoothness_loss = smoothness_loss + torch.mean((a_params[1:] - a_params[:-1]) ** 2)
        if not b_fixed:
            b_params = model.compensate.get_b_params()
            smoothness_loss = smoothness_loss + torch.mean((b_params[1:] - b_params[:-1]) ** 2)
        if boundary_trainable:
            boundary_offsets = model.compensate.boundary_offsets
            smoothness_loss = smoothness_loss + torch.mean((boundary_offsets[1:] - boundary_offsets[:-1]) ** 2)

        (smoothness_weight * smoothness_loss).backward()
        optimizer.step()

        current_total_loss = total_variance_value + float((smoothness_weight * smoothness_loss).detach().cpu())
        losses.append(current_total_loss)
        variance_losses.append(total_variance_value)
        smoothness_losses.append(float(smoothness_loss.detach().cpu()))

        cur_a = model.compensate.get_a_params().detach().cpu().numpy().copy()
        cur_b = model.compensate.get_b_params().detach().cpu().numpy().copy()
        a_hist.append(cur_a)
        b_hist.append(cur_b)

        if it % 100 == 0 or it == num_iterations - 1:
            a_range = f"[{cur_a.min():.3f}, {cur_a.max():.3f}]"
            b_range = f"[{cur_b.min():.3f}, {cur_b.max():.3f}]"
            print(f"{it:<6} {current_total_loss:<12.6f} {variance_losses[-1]:<12.6f} {smoothness_losses[-1]:<12.6f} {a_range:<18} {b_range:<18}")

        # Simple LR schedule
        if it == int(0.5 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.5
            if debug:
                print(f"  → LR reduced to {optimizer.param_groups[0]['lr']:.4f}")
        elif it == int(0.8 * num_iterations):
            optimizer.param_groups[0]['lr'] *= 0.1
            if debug:
                print(f"  → LR reduced to {optimizer.param_groups[0]['lr']:.4f}")

    print("-" * 90)
    print("Training completed (batched-by-file)!")
    print(f"  Final total loss: {losses[-1]:.6f}")
    print(f"  Final variance loss: {variance_losses[-1]:.6f}")
    print(f"  Final smoothness loss: {smoothness_losses[-1]:.6f}")

    original_t_start0 = torch.tensor(orig_t_start0, device=device, dtype=torch.float32)
    original_t_end0 = torch.tensor(orig_t_end0, device=device, dtype=torch.float32)
    return model, losses, variance_losses, smoothness_losses, a_hist, b_hist, original_t_start0, original_t_end0

# --------------------------------------------------------------------------------------
# Visualization + saving (brought over from the "without_batch" script)
# --------------------------------------------------------------------------------------
def create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end,
                        compensated=True, chunk_size=250_000, debug=False):
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)

    with torch.no_grad():
        if compensated:
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end,
                                    chunk_size=chunk_size, debug=debug)
        else:
            # Approximate "no compensation": zero trainable params
            if not model.compensate.a_fixed:
                save_a = model.compensate.a_params.clone(); model.compensate.a_params.data.zero_()
            if not model.compensate.b_fixed:
                save_b = model.compensate.b_params.clone(); model.compensate.b_params.data.zero_()
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end,
                                    chunk_size=chunk_size, debug=debug)
            if not model.compensate.a_fixed:
                model.compensate.a_params.data = save_a
            if not model.compensate.b_fixed:
                model.compensate.b_params.data = save_b
    return event_tensor


def get_param_string(model):
    a_params = model.compensate.get_a_params().detach().cpu().numpy()
    b_params = model.compensate.get_b_params().detach().cpu().numpy()
    a_status = "FIXED" if model.compensate.a_fixed else "TRAIN"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAIN"
    return f"a({a_status})=[{a_params.min():.4f},{a_params.max():.4f}], b({b_status})=[{b_params.min():.4f},{b_params.max():.4f}]"


def get_param_suffix(model):
    a_params = model.compensate.get_a_params().detach().cpu().numpy()
    b_params = model.compensate.get_b_params().detach().cpu().numpy()
    a_status = "fixed" if model.compensate.a_fixed else "train"
    b_status = "fixed" if model.compensate.b_fixed else "train"
    return f"_multiwindow_chunked_processing_a{a_status}_{a_params.min():.4f}_{a_params.max():.4f}_b{b_status}_{b_params.min():.4f}_{b_params.max():.4f}"


def plot_learned_parameters_with_data(model, x, y, t, sensor_width, sensor_height, output_dir=None, filename_prefix=""):
    print("Creating learned parameters visualization with data projections...")

    final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
    final_b_params = model.compensate.get_b_params().detach().cpu().numpy()

    x_range = np.linspace(0, sensor_width, 100)
    y_range = np.linspace(0, sensor_height, 100)
    duration = model.compensate.duration

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']

    def compute_boundary_lines_simple(a_params, b_params, coord_range, coord_type):
        boundary_offsets = model.compensate.boundary_offsets.detach().cpu().numpy()
        lines = []
        for i in range(len(a_params)):
            if coord_type == 'x':
                y_center = sensor_height / 2
                line_values = a_params[i] * coord_range + b_params[i] * y_center + boundary_offsets[i]
            else:
                x_center = sensor_width / 2
                line_values = a_params[i] * x_center + b_params[i] * coord_range + boundary_offsets[i]
            lines.append(line_values)
        return lines

    # Subsample for plot (increase cap if you want more)
    max_plot_events = 1_000_000
    if len(x) > max_plot_events:
        idx = np.random.choice(len(x), max_plot_events, replace=False)
        x_plot = x[idx]; y_plot = y[idx]; t_plot = t[idx]
    else:
        x_plot, y_plot, t_plot = x, y, t

    # Plot 1: X-T with data
    ax = axes[0, 0]
    ax.scatter(x_plot, t_plot/1000, c='lightblue', alpha=0.1, s=0.1, rasterized=True, label='Events')
    x_lines = compute_boundary_lines_simple(final_a_params, final_b_params, x_range, 'x')
    for i, vals in enumerate(x_lines):
        ax.plot(x_range, vals/1000, '--', alpha=0.8, linewidth=2.0, color=colors[i % len(colors)], label=f'Boundary {i}')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Time (ms)')
    ax.set_title('Learned Compensation - X-T View')
    ax.grid(True, alpha=0.3); ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, duration/1000); ax.set_xlim(0, sensor_width)

    # Plot 2: Y-T with data
    ax = axes[0, 1]
    ax.scatter(y_plot, t_plot/1000, c='lightgreen', alpha=0.1, s=0.1, rasterized=True, label='Events')
    y_lines = compute_boundary_lines_simple(final_a_params, final_b_params, y_range, 'y')
    for i, vals in enumerate(y_lines):
        ax.plot(y_range, vals/1000, '--', alpha=0.8, linewidth=2.0, color=colors[i % len(colors)], label=f'Boundary {i}')
    ax.set_xlabel('Y (px)'); ax.set_ylabel('Time (ms)')
    ax.set_title('Learned Compensation - Y-T View')
    ax.grid(True, alpha=0.3); ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, duration/1000); ax.set_xlim(0, sensor_height)

    # Plot 3: X-T boundaries only
    ax = axes[1, 0]
    for i, vals in enumerate(x_lines):
        ax.plot(x_range, vals/1000, '--', alpha=0.9, linewidth=3, color=colors[i % len(colors)], label=f'Boundary {i}')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Time (ms)')
    ax.set_title('Boundaries Only - X-T'); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, duration/1000); ax.set_xlim(0, sensor_width)

    # Plot 4: Y-T boundaries only
    ax = axes[1, 1]
    for i, vals in enumerate(y_lines):
        ax.plot(y_range, vals/1000, '--', alpha=0.9, linewidth=3, color=colors[i % len(colors)], label=f'Boundary {i}')
    ax.set_xlabel('Y (px)'); ax.set_ylabel('Time (ms)')
    ax.set_title('Boundaries Only - Y-T'); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, duration/1000); ax.set_xlim(0, sensor_height)

    a_status = "FIXED" if model.compensate.a_fixed else "TRAINABLE"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINABLE"
    fig.text(
        0.02, 0.02,
        f'Multi-Window Compensation Summary:\n'
        f'• Total events: {len(x):,} (plotted: {len(x_plot):,})\n'
        f'• Duration: {duration/1000:.1f} ms\n'
        f'• Main windows: {model.compensate.num_main_windows}, Total windows: {model.compensate.num_total_windows}\n'
        f'• a_params ({a_status}): [{final_a_params.min():.4f}, {final_a_params.max():.4f}]\n'
        f'• b_params ({b_status}): [{final_b_params.min():.4f}, {final_b_params.max():.4f}]',
        fontsize=11, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
    )

    plt.suptitle('Multi-Window Scan Compensation: Learned Boundaries with Event Data (CHUNKED)', fontsize=16, y=0.98)
    plt.tight_layout()

    if output_dir:
        plot_path = os.path.join(output_dir, f"{filename_prefix}_learned_parameters_with_data_chunked_processing.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Parameters with data plot saved to: {plot_path}")
    plt.show()


def visualize_results(
    model, x, y, t, p,
    losses, variance_losses, smoothness_losses,
    a_params_history, b_params_history,
    bin_width, sensor_width, sensor_height,
    original_t_start, original_t_end,
    output_dir=None, filename_prefix=""
):
    param_str = get_param_string(model)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Loss plots
    axes[0, 0].plot(losses, label='Total Loss', alpha=0.8)
    axes[0, 0].plot(variance_losses, label='Variance Loss', alpha=0.8)
    axes[0, 0].plot(smoothness_losses, label='Smoothness Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Iteration'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses (Chunked Processing)'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')

    # Parameter evolution (min/max)
    if len(a_params_history) > 0 and len(b_params_history) > 0:
        a_arr = np.array(a_params_history); b_arr = np.array(b_params_history)
        if a_arr.ndim >= 2 and b_arr.ndim >= 2:
            a_min, a_max = np.min(a_arr, axis=1), np.max(a_arr, axis=1)
            b_min, b_max = np.min(b_arr, axis=1), np.max(b_arr, axis=1)
            axes[0, 1].plot(a_min, label='a min', alpha=0.7)
            axes[0, 1].plot(a_max, label='a max', alpha=0.7)
            axes[0, 1].plot(b_min, label='b min', alpha=0.7)
            axes[0, 1].plot(b_max, label='b max', alpha=0.7)
            axes[0, 1].set_xlabel('Iteration'); axes[0, 1].set_ylabel('Param')
            axes[0, 1].set_title('Param Evolution (Min/Max)'); axes[0, 1].legend(); axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, 'Parameter history\nnot available',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Param Evolution (Min/Max)')
    else:
        axes[0, 1].text(0.5, 0.5, 'Parameter history\nnot available',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Param Evolution (Min/Max)')

    # Event frames (one middle slice)
    event_tensor_orig = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width,
                                            original_t_start, original_t_end, compensated=False)
    event_tensor_comp = create_event_frames(model, x, y, t, p, sensor_height, sensor_width, bin_width,
                                            original_t_start, original_t_end, compensated=True)

    Nbins = event_tensor_orig.shape[0]
    bin_idx = Nbins // 2

    im1 = axes[0, 2].imshow(event_tensor_orig[bin_idx].detach().cpu().numpy(), cmap='inferno', aspect='auto')
    axes[0, 2].set_title(f'Original - Bin {bin_idx}'); plt.colorbar(im1, ax=axes[0, 2])

    im2 = axes[1, 2].imshow(event_tensor_comp[bin_idx].detach().cpu().numpy(), cmap='inferno', aspect='auto')
    axes[1, 2].set_title(f'Compensated - Bin {bin_idx}'); plt.colorbar(im2, ax=axes[1, 2])

    # Variance comparison
    with torch.no_grad():
        if event_tensor_orig.shape != event_tensor_comp.shape:
            min_bins = min(event_tensor_orig.shape[0], event_tensor_comp.shape[0])
            event_tensor_orig = event_tensor_orig[:min_bins]
            event_tensor_comp = event_tensor_comp[:min_bins]
            Nbins = min_bins

        H, W = event_tensor_orig.shape[1], event_tensor_orig.shape[2]
        var_orig = torch.var(event_tensor_orig.reshape(Nbins, H * W), dim=1)
        var_comp = torch.var(event_tensor_comp.reshape(Nbins, H * W), dim=1)
        var_orig_mean = var_orig.mean().item()
        var_comp_mean = var_comp.mean().item()

    axes[1, 0].plot(var_orig.cpu().tolist(), label='Original', alpha=0.7)
    axes[1, 0].plot(var_comp.cpu().tolist(), label='Compensated', alpha=0.7)
    axes[1, 0].set_xlabel('Time Bin'); axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Variance Comparison'); axes[1, 0].legend(); axes[1, 0].grid(True)

    final_a = model.compensate.get_a_params().detach().cpu().numpy()
    final_b = model.compensate.get_b_params().detach().cpu().numpy()
    a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
    b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
    improvement_pct = (var_comp_mean / var_orig_mean - 1) * 100.0
    a_smooth = np.mean((final_a[1:] - final_a[:-1]) ** 2) if len(final_a) > 1 else 0.0
    b_smooth = np.mean((final_b[1:] - final_b[:-1]) ** 2) if len(final_b) > 1 else 0.0

    ax = axes[1, 1]
    ax.text(0.1, 0.95, f'Original mean variance: {var_orig_mean:.2f}', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.90, f'Compensated mean variance: {var_comp_mean:.2f}', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.85, f'Improvement: {improvement_pct:.1f}%', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.80, f'a_params ({a_status}): [{final_a.min():.2f}, {final_a.max():.2f}]', transform=ax.transAxes, fontsize=9)
    ax.text(0.1, 0.75, f'b_params ({b_status}): [{final_b.min():.2f}, {final_b.max():.2f}]', transform=ax.transAxes, fontsize=9)
    ax.text(0.1, 0.70, f'a_smoothness: {a_smooth:.4f}', transform=ax.transAxes, fontsize=9)
    ax.text(0.1, 0.65, f'b_smoothness: {b_smooth:.4f}', transform=ax.transAxes, fontsize=9)
    ax.text(0.1, 0.55, f'Final total loss: {losses[-1]:.6f}', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.50, f'Final variance loss: {variance_losses[-1]:.6f}', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.45, f'Final smoothness loss: {smoothness_losses[-1]:.6f}', transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Summary')
    fig.suptitle(f'Multi-Window Results (CHUNKED & UNIFIED)\n{param_str}', fontsize=16, y=0.98)

    plt.tight_layout()
    if output_dir:
        suffix = get_param_suffix(model)
        out_path = os.path.join(output_dir, f"{filename_prefix}_multiwindow_compensation_results{suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to: {out_path}")
    plt.show()


def save_results(model, losses, a_params_history, b_params_history, output_dir, filename_prefix):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        suffix = get_param_suffix(model)

        final_a = model.compensate.get_a_params().detach().cpu().numpy()
        final_b = model.compensate.get_b_params().detach().cpu().numpy()

        txt_path = os.path.join(output_dir, f"{filename_prefix}_multiwindow_compensation_results{suffix}.txt")
        with open(txt_path, 'w') as f:
            a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
            b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
            f.write("MULTI-WINDOW SCAN COMPENSATION RESULTS (CHUNKED PROCESSING - UNIFIED VARIANCE)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"a_params: {a_status}\n")
            f.write(f"b_params: {b_status}\n\n")
            f.write(f"Final a_params: {final_a.tolist()}\n")
            f.write(f"Final b_params: {final_b.tolist()}\n")
            f.write(f"a range: [{final_a.min():.6f}, {final_a.max():.6f}]\n")
            f.write(f"b range: [{final_b.min():.6f}, {final_b.max():.6f}]\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Training iterations: {len(losses)}\n")
            f.write(f"Main windows: {model.compensate.num_main_windows}\n")
            f.write(f"Total windows: {model.compensate.num_total_windows}\n")
            f.write(f"Duration: {model.compensate.duration:.0f} μs\n")
        print(f"Results saved to: {txt_path}")

        np.save(os.path.join(output_dir, f"{filename_prefix}_final_a_params{suffix}.npy"), final_a)
        np.save(os.path.join(output_dir, f"{filename_prefix}_final_b_params{suffix}.npy"), final_b)
        np.save(os.path.join(output_dir, f"{filename_prefix}_loss_history{suffix}.npy"), np.array(losses))
        np.save(os.path.join(output_dir, f"{filename_prefix}_a_params_history{suffix}.npy"), np.array(a_params_history))
        np.save(os.path.join(output_dir, f"{filename_prefix}_b_params_history{suffix}.npy"), np.array(b_params_history))

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Multi-Window Scan Compensation (Standalone) – Single-file or Batched-by-File (--merge)')
    p.add_argument('input_path', help='Path to NPZ event file (single-file) OR segments folder (when using --merge)')
    p.add_argument('--merge', action='store_true', help='Treat a folder of segments as multiple batches (no concatenation)')
    p.add_argument('--output_dir', default=None, help='Directory to save results')
    p.add_argument('--sensor_width', type=int, default=1280, help='Sensor width (W)')
    p.add_argument('--sensor_height', type=int, default=720, help='Sensor height (H)')
    p.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds (μs)')
    p.add_argument('--iterations', type=int, default=1000, help='Training iterations')
    p.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    p.add_argument('--smoothness_weight', type=float, default=0.001, help='Smoothness regularization weight')

    # Parameter config
    p.add_argument('--num_params', type=int, default=13)
    p.add_argument('--temperature', type=float, default=5000)
    p.add_argument('--chunk_size', type=int, default=250_000)

    # Train/fix flags (defaults mirror the "without_batch" behavior)
    p.add_argument('--a_fixed', action='store_true', default=True, help='Fix a_params during training (default: True)')
    p.add_argument('--a_trainable', dest='a_fixed', action='store_false', help='Make a_params trainable')
    p.add_argument('--b_fixed', dest='b_fixed', action='store_true', default=False, help='Fix b_params during training (default: False)')
    p.add_argument('--b_trainable', dest='b_fixed', action='store_false', help='Make b_params trainable (default)')
    p.add_argument('--boundary_trainable', action='store_true', help='Make boundary offsets trainable (default: False)')
    p.add_argument('--a_default', type=float, default=0.0)
    p.add_argument('--b_default', type=float, default=-76.0)

    p.add_argument('--visualize', action='store_true', help='Show visualization plots')
    p.add_argument('--plot_params', action='store_true', help='Plot learned parameters with event data')
    p.add_argument('--debug', action='store_true', help='Verbose debug prints')

    args = p.parse_args()

    print("=" * 80)
    print("MULTI-WINDOW SCAN COMPENSATION – STANDALONE")
    print("• Single-file (no --merge): unified, chunked processing (like without_batch)")
    print("• Folder + --merge       : batched-by-file (avoid OOM, same objective)")
    print("=" * 80)

    if args.merge:
        # Expect a folder of Scan_*_events.npz
        if not os.path.isdir(args.input_path):
            raise ValueError(f"When using --merge, input_path must be a directory: {args.input_path}")
        batch_files = list_batches(args.input_path, debug=args.debug)

        if args.output_dir is None:
            args.output_dir = args.input_path
        os.makedirs(args.output_dir, exist_ok=True)

        model, losses, variance_losses, smoothness_losses, a_hist, b_hist, t_start0, t_end0 = train_batched_by_file(
            batch_files,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            bin_width=args.bin_width,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            debug=args.debug,
            smoothness_weight=args.smoothness_weight,
            a_fixed=args.a_fixed,
            b_fixed=args.b_fixed,
            boundary_trainable=args.boundary_trainable,
            a_default=args.a_default,
            b_default=args.b_default,
            num_params=args.num_params,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
        )

        # Report
        a_final = model.compensate.get_a_params().detach().cpu().numpy()
        b_final = model.compensate.get_b_params().detach().cpu().numpy()
        a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
        b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
        boundary_status = "TRAINED" if getattr(model.compensate, 'boundary_trainable', False) else "FIXED"

        print(f"\n🎯 Final Results (Batched-by-File):")
        print(f"  Batches processed: {len(batch_files)} (chunk size: {args.chunk_size:,})")
        print(f"  a_params ({a_status}): [{a_final.min():.6f}, {a_final.max():.6f}]")
        print(f"  b_params ({b_status}): [{b_final.min():.6f}, {b_final.max():.6f}]")
        print(f"  boundaries ({boundary_status})")
        print(f"  Final loss: {losses[-1]:.6f}")

        trainable_params = 0
        if not model.compensate.a_fixed: trainable_params += len(a_final)
        if not model.compensate.b_fixed: trainable_params += len(b_final)
        if getattr(model.compensate, 'boundary_trainable', False): trainable_params += len(a_final)
        total_params = len(a_final) * (3 if getattr(model.compensate, 'boundary_trainable', False) else 2)
        print(f"  Trainable parameters: {trainable_params}/{total_params}")
        print(f"  Config: {args.num_params} params, temp={args.temperature:.0f}, chunk={args.chunk_size:,}")

        # Save
        base_name = os.path.basename(args.input_path.rstrip('/'))
        save_results(model, losses, a_hist, b_hist, args.output_dir, base_name + "_batched")

        # Optional viz using FIRST batch only (memory-friendly)
        if args.plot_params or args.visualize:
            x_vis, y_vis, t_vis, p_vis = load_single_segment_with_metadata(batch_files[0], debug=False)
            if args.plot_params:
                try:
                    plot_learned_parameters_with_data(model, x_vis, y_vis, t_vis,
                                                      args.sensor_width, args.sensor_height,
                                                      args.output_dir, base_name + "_batched")
                except Exception as e:
                    print(f"plot_learned_parameters_with_data failed: {e}")
            if args.visualize:
                try:
                    visualize_results(model, x_vis, y_vis, t_vis, p_vis,
                                      losses, variance_losses, smoothness_losses,
                                      a_hist, b_hist,
                                      args.bin_width, args.sensor_width, args.sensor_height,
                                      t_start0, t_end0,
                                      args.output_dir, base_name + "_batched")
                except Exception as e:
                    print(f"visualize_results failed: {e}")

        print("\n✅ Batched-by-file training complete (chunked + gradient accumulation). OOM be gone.")

    else:
        # Expect a single NPZ file path
        if not os.path.isfile(args.input_path):
            raise ValueError(f"Expected a single NPZ file when not using --merge: {args.input_path}")
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.input_path) or "."

        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        print(f"Analyzing single file: {args.input_path}")

        x, y, t, p = load_npz_events(args.input_path, debug=args.debug)

        print(f"\n🔥 PROCESSING {len(x):,} EVENTS - CHUNKED PROCESSING, UNIFIED VARIANCE 🔥")
        model, losses, variance_losses, smoothness_losses, a_hist, b_hist, original_t_start, original_t_end = train_scan_compensation(
            x, y, t, p,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            bin_width=args.bin_width,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            debug=args.debug,
            smoothness_weight=args.smoothness_weight,
            a_fixed=args.a_fixed,
            b_fixed=args.b_fixed,
            boundary_trainable=args.boundary_trainable,
            a_default=args.a_default,
            b_default=args.b_default,
            num_params=args.num_params,
            temperature=args.temperature,
            chunk_size=args.chunk_size
        )

        # Final report
        final_a_params = model.compensate.get_a_params().detach().cpu().numpy()
        final_b_params = model.compensate.get_b_params().detach().cpu().numpy()
        a_status = "FIXED" if model.compensate.a_fixed else "TRAINED"
        b_status = "FIXED" if model.compensate.b_fixed else "TRAINED"
        boundary_status = "TRAINED" if model.compensate.boundary_trainable else "FIXED"

        print(f"\n🎯 Final Results (Single-file):")
        print(f"  Processed: {len(x):,} events (chunk size: {args.chunk_size:,})")
        print(f"  a_params ({a_status}): [{final_a_params.min():.6f}, {final_a_params.max():.6f}]")
        print(f"  b_params ({b_status}): [{final_b_params.min():.6f}, {final_b_params.max():.6f}]")
        print(f"  boundaries ({boundary_status})")
        print(f"  Final loss: {losses[-1]:.6f}")

        trainable_params = 0
        if not model.compensate.a_fixed: trainable_params += len(final_a_params)
        if not model.compensate.b_fixed: trainable_params += len(final_b_params)
        if model.compensate.boundary_trainable: trainable_params += len(final_a_params)
        total_params = len(final_a_params) * (3 if model.compensate.boundary_trainable else 2)
        print(f"  Trainable parameters: {trainable_params}/{total_params}")
        print(f"  Config: {args.num_params} params, temp={args.temperature:.0f}, chunk={args.chunk_size:,}")

        # Save results
        save_results(model, losses, a_hist, b_hist, args.output_dir, f"{base_name}_chunked_processing")

        # Optional plots
        if args.plot_params:
            plot_learned_parameters_with_data(model, x, y, t, args.sensor_width, args.sensor_height,
                                              args.output_dir, f"{base_name}_chunked_processing")
        if args.visualize:
            visualize_results(model, x, y, t, p,
                              losses, variance_losses, smoothness_losses,
                              a_hist, b_hist,
                              args.bin_width, args.sensor_width, args.sensor_height,
                              original_t_start, original_t_end,
                              args.output_dir, f"{base_name}_chunked_processing")

        print("\n✅ Multi-Window Scan compensation complete (CHUNKED PROCESSING - UNIFIED VARIANCE)!")
        print("🚀 All events processed with chunking for memory efficiency and unified variance for proper learning!")

if __name__ == "__main__":
    main()
