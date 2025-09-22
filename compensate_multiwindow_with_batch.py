#!/usr/bin/env python3
"""
Multi-Window Scan Compensation (Batched-by-File version)

Goal
----
Handle very large datasets (tens of millions of events across many NPZ files)
without CUDA OOM by:
  • Treating EACH NPZ file as a BATCH
  • Running the SAME per-file pipeline as the single-file case
  • Accumulating gradients across batches (no graph kept in memory across files)
  • Keeping unified optimization + smoothness regularization

This preserves correctness of the objective (sum of per-batch variance losses)
while drastically cutting peak memory usage.

Notes
-----
- Uses chunked accumulation internally (like the no_batch_2 script),
  but now *also* accumulates across many NPZ files.
- If you set --merge, we DO NOT concatenate events into one giant array.
  We simply build a list of segment files and iterate them.
- Per batch we fully free intermediate tensors before moving to the next file.

Tip: For large models, also consider setting the env var
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
as suggested by PyTorch to reduce fragmentation.

This file is self-contained but will try to import reusable classes from
`scanning_alignment_with_merge_multi_no_batch_2.py` if available.
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Try to import reusable parts from the user's previous script to avoid duplication
# --------------------------------------------------------------------------------------
_Imported = False
try:
    from scanning_alignment_with_merge_multi_without_batch import (
        Compensate,
        ScanCompensation as _OrigScanCompensation,
        create_event_frames as _orig_create_event_frames,
        visualize_results as _orig_visualize_results,
        plot_learned_parameters_with_data as _orig_plot_learned_parameters_with_data,
        save_results as _orig_save_results,
    )
    _Imported = True
except Exception:
    _Imported = False

# --------------------------------------------------------------------------------------
# If import failed, provide minimal self-contained implementations (copied/adapted)
# --------------------------------------------------------------------------------------

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not _Imported:
    class Compensate(nn.Module):
        def __init__(self, a_params, b_params, duration, num_params=13, temperature=5000,
                     device='cpu', a_fixed=True, b_fixed=False, boundary_trainable=False, debug=False):
            super().__init__()
            self.device = device
            self.duration = float(duration)
            self.temperature = float(temperature)
            self.a_fixed = a_fixed
            self.b_fixed = b_fixed
            self.boundary_trainable = boundary_trainable
            self.debug = debug

            if len(a_params) != len(b_params):
                raise ValueError("a_params and b_params must have the same length")
            if len(a_params) != num_params:
                raise ValueError(f"Parameter arrays must have length {num_params}, got {len(a_params)}")

            if a_fixed:
                self.a_params = torch.tensor(a_params, dtype=torch.float32, device=device, requires_grad=False)
                self.register_buffer('a_params_buffer', self.a_params)
            else:
                self.a_params = nn.Parameter(torch.tensor(a_params, dtype=torch.float32, device=device))

            if b_fixed:
                self.b_params = torch.tensor(b_params, dtype=torch.float32, device=device, requires_grad=False)
                self.register_buffer('b_params_buffer', self.b_params)
            else:
                self.b_params = nn.Parameter(torch.tensor(b_params, dtype=torch.float32, device=device))

            self.num_boundaries = len(a_params)
            self.num_main_windows = self.num_boundaries - 3
            self.num_total_windows = self.num_main_windows + 2
            if self.num_main_windows <= 0:
                raise ValueError("Need at least 4 parameters for 1 main window")

            self.main_window_size = self.duration / self.num_main_windows

            if boundary_trainable:
                initial_raw = torch.tensor([self.main_window_size] * self.num_boundaries, dtype=torch.float32, device=device)
                initial_raw[0] = -self.main_window_size
                self.raw_boundary_params = nn.Parameter(initial_raw)
            else:
                boundary_offsets = torch.tensor([
                    (i - 1) * self.main_window_size for i in range(self.num_boundaries)
                ], dtype=torch.float32, device=device, requires_grad=False)
                self.register_buffer('fixed_boundary_offsets', boundary_offsets)

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
            for start in range(0, num_events, chunk_size):
                end = min(start + chunk_size, num_events)
                x_ch = x[start:end]; y_ch = y[start:end]; t_ch = t_shifted[start:end]

                aE = a_params[:, None].expand(-1, end-start)
                bE = b_params[:, None].expand(-1, end-start)
                oE = self.boundary_offsets[:, None].expand(-1, end-start)

                xE = x_ch[None, :].expand(self.num_boundaries, -1)
                yE = y_ch[None, :].expand(self.num_boundaries, -1)

                bounds = aE * xE + bE * yE + oE

                tE = t_ch[None, :].expand(self.num_total_windows, -1)
                lo = bounds[:-1, :]
                hi = bounds[1:, :]
                ls = torch.sigmoid((tE - lo) / self.temperature)
                us = torch.sigmoid((hi - tE) / self.temperature)
                m = ls * us
                m_sum = torch.clamp(torch.sum(m, dim=0, keepdim=True), min=1e-8)
                m = m / m_sum

                w = torch.clamp(hi - lo, min=1e-8)
                aL = a_params[:-1, None].expand(-1, end-start)
                aU = a_params[1:,  None].expand(-1, end-start)
                bL = b_params[:-1, None].expand(-1, end-start)
                bU = b_params[1:,  None].expand(-1, end-start)
                alpha = (tE - lo) / w
                slope_a = (1 - alpha) * aL + alpha * aU
                slope_b = (1 - alpha) * bL + alpha * bU

                comp = torch.sum(m * slope_a * x_ch[None, :], dim=0) + \
                       torch.sum(m * slope_b * y_ch[None, :], dim=0)
                outs.append(comp)

                del aE,bE,oE,xE,yE,bounds,tE,lo,hi,ls,us,m,m_sum,w,aL,aU,bL,bU,alpha,slope_a,slope_b,comp
                torch.cuda.empty_cache()
            return torch.cat(outs, dim=0)

    class ScanCompensation(nn.Module):
        def __init__(self, duration, num_params=13, device='cuda', a_fixed=True, b_fixed=False,
                     boundary_trainable=False, a_default=0.0, b_default=-76.0, temperature=5000, debug=False):
            super().__init__()
            a_params = [a_default] * num_params
            b_params = [b_default] * num_params
            self.compensate = Compensate(a_params, b_params, duration, num_params=num_params,
                                         device=device, a_fixed=a_fixed, b_fixed=b_fixed,
                                         boundary_trainable=boundary_trainable, temperature=temperature, debug=debug)

        def forward(self, x_coords, y_coords, timestamps, polarities, H, W, bin_width,
                    original_t_start=None, original_t_end=None, chunk_size=500_000, debug=False):
            if original_t_start is not None and original_t_end is not None:
                mask = (timestamps >= original_t_start) & (timestamps <= original_t_end)
                x_coords = x_coords[mask]; y_coords = y_coords[mask]
                timestamps = timestamps[mask]; polarities = polarities[mask]
                t_start = original_t_start; t_end = original_t_end
            else:
                t_start = timestamps.min(); t_end = timestamps.max()

            time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)
            num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

            N = len(x_coords)
            if debug:
                print(f"  Batch events: {N:,} -> bins={num_bins}")

            complete = torch.zeros(num_bins, H, W, device=device, dtype=torch.float32)
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                x_ch = x_coords[s:e]; y_ch = y_coords[s:e]
                t_ch = timestamps[s:e]; p_ch = polarities[s:e]

                comp_ch = self.compensate(x_ch, y_ch, t_ch, chunk_size=chunk_size, debug=debug)
                t_warped = t_ch - comp_ch

                t_norm = (t_warped - t_start) / time_bin_width
                t0 = torch.floor(t_norm); t1 = t0 + 1
                wt = (t_norm - t0).float()
                t0c = t0.clamp(0, num_bins-1)
                t1c = t1.clamp(0, num_bins-1)

                xi = x_ch.long(); yi = y_ch.long()
                valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                xi = xi[valid]; yi = yi[valid]
                t0c = t0c[valid]; t1c = t1c[valid]
                wt = wt[valid]; p_ch = p_ch[valid]

                spatial = (yi * W + xi).long()
                num_elements = num_bins * H * W

                flat_t0 = (t0c * (H*W) + spatial).long()
                flat_t1 = (t1c * (H*W) + spatial).long()

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

                del x_ch,y_ch,t_ch,p_ch,comp_ch,t_warped,t_norm,t0,t1,wt,t0c,t1c,xi,yi,valid,spatial
                del num_elements,flat_t0,flat_t1,w0,w1,flat_idx,flat_wts,flat_tensor
                torch.cuda.empty_cache()

            variances = torch.var(complete.view(num_bins, -1), dim=1)
            total_loss = torch.sum(variances)
            return complete, total_loss

# --------------------------------------------------------------------------------------
# Helper loaders specific to batched-by-file training
# --------------------------------------------------------------------------------------

def load_single_segment_with_metadata(npz_path, debug=False):
    """Load ONE segment NPZ and normalize time/polarity like the merge code did.
    Expected keys: x,y,t,p,start_time,duration_us,direction.
    Returns numpy arrays x,y,t,p (float32), where t is normalized to [0, duration].
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

    # Defaults if missing (single-file recordings may not have metadata)
    start_time = float(data['start_time']) if 'start_time' in data else float(t.min())
    duration = float(data['duration_us']) if 'duration_us' in data else float(t.max() - t.min())
    direction = str(data['direction']) if 'direction' in data else 'Forward'

    # Normalize time
    t_norm = t - start_time

    # Normalize polarity to [-1, 1]
    if p.min() >= 0 and p.max() <= 1:
        p = (p - 0.5) * 2

    # Backward scan handling
    if 'Backward' in direction:
        t_norm = duration - t_norm
        p = -p

    # Sort by time
    order = np.argsort(t_norm)
    x = x[order]; y = y[order]; t_norm = t_norm[order]; p = p[order]

    if debug:
        print(f"    → {len(x):,} events, duration≈{duration/1000:.1f}ms, dir={direction}")
    return x, y, t_norm, p


def list_batches(input_path, merge, debug=False):
    """Return a list of NPZ file paths to treat as separate batches.
    - If merge=False and input_path is a file: [input_path]
    - If merge=True and input_path is a folder: sorted list of Scan_*_events.npz
    - If merge=False but input_path is a folder: also treat each Scan_* file as a batch
    """
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "Scan_*_events.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No segment files found under folder: {input_path}")
        if debug:
            print(f"Found {len(files)} NPZ segments under folder (batched): {input_path}")
        return files
    else:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)
        return [input_path]

# --------------------------------------------------------------------------------------
# Training (batched-by-file) — gradient accumulation across batches
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
    """Train the model by iterating each NPZ file as a batch and accumulating gradients.
    This avoids constructing a giant event tensor spanning all files at once.
    """

    # Inspect first batch to set the model's duration (use that file's full span)
    # This mirrors the single-file case and keeps the same behavior per batch.
    x0, y0, t0, p0 = load_single_segment_with_metadata(batch_files[0], debug=debug)
    orig_t_start0 = float(t0.min())
    orig_t_end0 = float(t0.max())
    duration0 = orig_t_end0 - orig_t_start0

    if debug:
        print(f"Initial duration from first batch: {duration0/1000:.1f} ms")

    model = (_OrigScanCompensation(duration0, num_params=num_params, device=device,
                                   a_fixed=a_fixed, b_fixed=b_fixed, boundary_trainable=boundary_trainable,
                                   a_default=a_default, b_default=b_default, temperature=temperature, debug=debug)
             if _Imported else
             ScanCompensation(duration0, num_params=num_params, device=device,
                              a_fixed=a_fixed, b_fixed=b_fixed, boundary_trainable=boundary_trainable,
                              a_default=a_default, b_default=b_default, temperature=temperature, debug=debug))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training Progress (Batched-by-File, chunked, unified across batches):")
    print(f"Batches: {len(batch_files)} | Sensor: {sensor_width}x{sensor_height} | bin={bin_width/1000:.1f}ms")
    print(f"{'Iter':<6} {'Total Loss':<12} {'Var Loss':<12} {'Smooth Loss':<12} {'a_range':<18} {'b_range':<18}")
    print("-" * 90)

    losses = []
    variance_losses = []
    smoothness_losses = []
    a_params_history = []
    b_params_history = []

    for it in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)

        total_variance_value = 0.0

        # --- Accumulate gradients across batches ---
        for bi, npz_path in enumerate(batch_files):
            x, y, t, p = load_single_segment_with_metadata(npz_path, debug=(debug and it == 0 and bi < 2))

            # Convert to tensors on device
            xs = torch.tensor(x, device=device, dtype=torch.float32)
            ys = torch.tensor(y, device=device, dtype=torch.float32)
            ts = torch.tensor(t, device=device, dtype=torch.float32)
            ps = torch.tensor(p, device=device, dtype=torch.float32)

            orig_t_start = torch.tensor(float(t.min()), device=device, dtype=torch.float32)
            orig_t_end = torch.tensor(float(t.max()), device=device, dtype=torch.float32)

            # Forward on this batch
            _, var_loss = model(xs, ys, ts, ps,
                                sensor_height, sensor_width, bin_width,
                                orig_t_start, orig_t_end,
                                chunk_size=chunk_size, debug=False)

            # Accumulate gradients immediately to avoid keeping graph for all batches
            var_loss.backward()
            total_variance_value += float(var_loss.detach().cpu())

            # Clear per-batch tensors
            del xs, ys, ts, ps, orig_t_start, orig_t_end, var_loss
            torch.cuda.empty_cache()

        # Smoothness reg (only depends on parameters; tiny cost once per iter)
        smoothness_loss = torch.tensor(0.0, device=device)
        if not a_fixed:
            a_params = model.compensate.get_a_params()
            smoothness_loss = smoothness_loss + torch.mean((a_params[1:] - a_params[:-1])**2)
        if not b_fixed:
            b_params = model.compensate.get_b_params()
            smoothness_loss = smoothness_loss + torch.mean((b_params[1:] - b_params[:-1])**2)
        if boundary_trainable:
            boundary_offsets = model.compensate.boundary_offsets
            smoothness_loss = smoothness_loss + torch.mean((boundary_offsets[1:] - boundary_offsets[:-1])**2)

        # Backprop smoothness (single small graph)
        (smoothness_weight * smoothness_loss).backward()

        # Now step
        optimizer.step()

        # Book-keeping
        current_total_loss = total_variance_value + float((smoothness_weight * smoothness_loss).detach().cpu())
        losses.append(current_total_loss)
        variance_losses.append(total_variance_value)
        smoothness_losses.append(float(smoothness_loss.detach().cpu()))

        cur_a = model.compensate.get_a_params().detach().cpu().numpy().copy()
        cur_b = model.compensate.get_b_params().detach().cpu().numpy().copy()
        a_params_history.append(cur_a)
        b_params_history.append(cur_b)

        if it % 100 == 0 or it == num_iterations - 1:
            a_range = f"[{cur_a.min():.3f}, {cur_a.max():.3f}]"
            b_range = f"[{cur_b.min():.3f}, {cur_b.max():.3f}]"
            print(f"{it:<6} {current_total_loss:<12.6f} {variance_losses[-1]:<12.6f} {smoothness_losses[-1]:<12.6f} {a_range:<18} {b_range:<18}")

        # Simple LR schedule (same milestones as before)
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

    # Provide first-batch times for downstream visualization convenience
    original_t_start0 = torch.tensor(orig_t_start0, device=device, dtype=torch.float32)
    original_t_end0 = torch.tensor(orig_t_end0, device=device, dtype=torch.float32)
    return model, losses, variance_losses, smoothness_losses, a_params_history, b_params_history, original_t_start0, original_t_end0

# --------------------------------------------------------------------------------------
# Visualization wrappers – re-use originals if imported, else provide light fallbacks
# --------------------------------------------------------------------------------------

def create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end,
                        compensated=True, chunk_size=250_000, debug=False):
    if _Imported:
        return _orig_create_event_frames(model, x, y, t, p, H, W, bin_width, original_t_start, original_t_end,
                                         compensated=compensated, chunk_size=chunk_size, debug=debug)
    # Fallback: run model forward either with current parameters or with zeroed params (approximate)
    xs = torch.tensor(x, device=device, dtype=torch.float32)
    ys = torch.tensor(y, device=device, dtype=torch.float32)
    ts = torch.tensor(t, device=device, dtype=torch.float32)
    ps = torch.tensor(p, device=device, dtype=torch.float32)
    with torch.no_grad():
        if compensated:
            event_tensor, _ = model(xs, ys, ts, ps, H, W, bin_width, original_t_start, original_t_end,
                                    chunk_size=chunk_size, debug=debug)
        else:
            # temporarily zero params if trainable to simulate "no comp"
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


def plot_learned_parameters_with_data(*args, **kwargs):
    if _Imported:
        return _orig_plot_learned_parameters_with_data(*args, **kwargs)
    # If originals aren’t available, silently skip (keeps script functioning)
    print("(plot_learned_parameters_with_data) skipped – original helper not found.")


def visualize_results(*args, **kwargs):
    if _Imported:
        return _orig_visualize_results(*args, **kwargs)
    print("(visualize_results) skipped – original helper not found.")


def save_results(*args, **kwargs):
    if _Imported:
        return _orig_save_results(*args, **kwargs)
    # Fallback: write minimal results
    model, losses, a_hist, b_hist, out_dir, prefix = args[0], args[1], args[2], args[3], args[4], args[5]
    os.makedirs(out_dir, exist_ok=True)
    a = model.compensate.get_a_params().detach().cpu().numpy()
    b = model.compensate.get_b_params().detach().cpu().numpy()
    path = os.path.join(out_dir, f"{prefix}_batched_results.txt")
    with open(path, 'w') as f:
        f.write("Batched-by-file results (minimal fallback)\n")
        f.write(f"Final a: {a.tolist()}\nFinal b: {b.tolist()}\nFinal loss: {losses[-1]:.6f}\n")
    print(f"Results saved to: {path}")

# --------------------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Multi-Window Scan compensation – Batched-by-File (OOM-safe)')
    p.add_argument('input_path', help='Path to NPZ event file OR segments folder')
    p.add_argument('--merge', action='store_true', help='Treat folder of segments as multiple batches (no concatenation)')
    p.add_argument('--output_dir', default=None, help='Output directory for results')
    p.add_argument('--sensor_width', type=int, default=1280, help='Sensor width')
    p.add_argument('--sensor_height', type=int, default=720, help='Sensor height')
    p.add_argument('--bin_width', type=float, default=1e5, help='Time bin width in microseconds')
    p.add_argument('--iterations', type=int, default=1000, help='Training iterations')
    p.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    p.add_argument('--smoothness_weight', type=float, default=0.001, help='Smoothness reg weight')

    # Parameter config
    p.add_argument('--num_params', type=int, default=13)
    p.add_argument('--temperature', type=float, default=5000)
    p.add_argument('--chunk_size', type=int, default=250_000)

    # Fix/train flags (same defaults as your previous run)
    p.add_argument('--a_fixed', action='store_true', default=True, help='Fix a_params during training (default: True)')
    p.add_argument('--a_trainable', dest='a_fixed', action='store_false', help='Make a_params trainable')
    p.add_argument('--b_fixed', action='store_true', help='Fix b_params during training (default: False)')
    p.add_argument('--b_trainable', dest='b_fixed', action='store_false', default=True, help='Make b_params trainable (default)')
    p.add_argument('--boundary_trainable', action='store_true', help='Train boundary offsets')
    p.add_argument('--a_default', type=float, default=0.0)
    p.add_argument('--b_default', type=float, default=-76.0)

    p.add_argument('--visualize', action='store_true', help='Show visualization plots (requires helpers)')
    p.add_argument('--plot_params', action='store_true', help='Plot learned parameters (requires helpers)')
    p.add_argument('--debug', action='store_true')

    args = p.parse_args()

    print("="*80)
    print("MULTI-WINDOW SCAN COMPENSATION – BATCHED-BY-FILE VERSION (CHUNKED + GRAD ACCUM)")
    print("="*80)

    # Resolve batches
    batch_files = list_batches(args.input_path, merge=args.merge, debug=args.debug)

    # Output dir
    if args.output_dir is None:
        args.output_dir = args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Train across batches
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

    # Final report
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
    base_name = (os.path.basename(args.input_path.rstrip('/')) if os.path.isdir(args.input_path) else
                 os.path.splitext(os.path.basename(args.input_path))[0])
    save_results(model, losses, a_hist, b_hist, args.output_dir, base_name + "_batched")

    # Optional viz using FIRST batch for frames/params (keeps memory modest)
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


if __name__ == "__main__":
    main()
