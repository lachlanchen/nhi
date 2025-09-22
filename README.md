Spectral Reconstruction with Event Cameras: Segment, Compensate, Visualize

Overview

This repository reconstructs spectra from an event camera while the scene is illuminated by a dispersed light (e.g., diffraction grating). The camera records intensity change events e = (x, y, t, p) where p ∈ {−1, +1} indicates the polarity of log-intensity change. Because illumination sweeps across wavelengths over time, the event stream encodes a temporal derivative of the underlying spectrum along the dispersion axis. The pipeline has three stages:

- Segment: find scan timing and split the recording into forward/backward passes.
- Compensate: estimate a piecewise-linear time-warp that removes scan-induced temporal tilt in the x–t and y–t planes.
- Visualize: overlay learned boundaries and compare original vs compensated time-binned frames.

Below is a concise mathematical description of the three main scripts and how they fit together.

1) Segment: segment_robust_fixed.py

Goal: From raw events, recover the scan timing (start/end, period) and slice the recording into 6 one-way scans (F, B, F, B, F, B).

- Activity signal (1D):
  Given timestamps t (μs), bin them with Δt = 1000 μs to produce a[n] = number of events in bin n. Let t_min = min(t) and N be the number of bins.

    a[n] = #{ i | t_min + nΔt ≤ t_i < t_min + (n+1)Δt }.

- Active window detection (80% events):
  Find the smallest contiguous index window [s, e) such that Σ_{n=s}^{e−1} a[n] ≥ 0.8 Σ_{n=0}^{N−1} a[n]. This yields a compact region that concentrates the scan.

- Period estimation (two modes):
  • Automatic: use autocorrelation of the normalized activity signal

      r[k] = Σ_n (a[n]−μ)(a[n+k]−μ), with r[0] used to normalize,

    then detect symmetric side peaks near ±P to estimate the round-trip period P (in bins). Alternatively,
  • Manual: pass a fixed round-trip period P (default 1688 bins). In the fixed variant, peak search is skipped and P is used directly.

- Reverse-correlation and timing structure:
  With the activity reversed a_rev[n] = a[N−1−n], compute

      R[k] = Σ_n a[n] a_rev[n+k].

  The maximal peak location gives a constraint between the prelude and aftermath durations; together with the total active length and 3P main span (F+B+F), this determines (prelude, aftermath, main-length).

- Convert to absolute time and segment:
  With one-way period τ = P/2 bins and scan start index s_start, absolute times are

      T_start = t_min + s_start Δt,
      segment i: [ T_start + (i−1)τΔt, T_start + i τΔt ), i = 1..6,

  alternating Forward/Backward directions. Each segment is saved as NPZ with x, y, t, p plus metadata (start_time, duration_us, direction, scan_id).

2) Compensate: compensate_multiwindow_train_saved_params.py

Goal: Learn a time-warp that removes scan-induced temporal shear so that events for the same wavelength align across the sensor. The model is multi-window, piecewise linear in x and y with soft transitions across temporal boundaries.

- Boundary surfaces (piecewise windows):
  Let there be M = num_params parameters for a and b. Define boundary times as planes in (x, y, t):

      T_i(x, y) = a_i x + b_i y + c_i,  i = 0..M−1,

  with fixed offsets c_i spaced uniformly over the scan duration (or learned when boundary_trainable=True). These M boundaries define M−1 windows.

- Soft window memberships:
  For an event (x, y, t), define the membership of window i (between T_i and T_{i+1}) via sigmoids (temperature τ > 0):

      m_i(x,y,t) = σ((t − T_i)/τ) · σ((T_{i+1} − t)/τ),
      w_i = m_i / (Σ_j m_j + ε).

- Interpolated slopes within each window:
  Define α_i = (t − T_i) / (T_{i+1} − T_i) ∈ [0, 1] and

      ã_i = (1 − α_i) a_i + α_i a_{i+1},
      b̃_i = (1 − α_i) b_i + α_i b_{i+1}.

- Time warp (compensation):

      Δt(x,y,t) = Σ_i w_i [ ã_i x + b̃_i y ],
      t′ = t − Δt(x,y,t).

  In the “FAST” visualization path, a constant linear compensation is used with a_avg = mean_i a_i and b_avg = mean_i b_i:

      t′_FAST = t − a_avg x − b_avg y.

- Training objective (variance minimization):
  Accumulate events into a time-binned tensor E ∈ R^{B×H×W} using linear interpolation in time:

      E[b, y, x] = Σ_{events} p · κ_b(t′),

  where κ_b distributes an event between adjacent time bins based on t′. The loss encourages spatial concentration per time bin:

      L_var = Σ_{b=1}^B Var_{y,x}(E[b, y, x]).

  Add smoothness penalties across adjacent parameters:

      L_smooth = Σ_i (a_{i+1} − a_i)^2 + Σ_i (b_{i+1} − b_i)^2 (+ boundary smoothness if enabled),

  and optimize L = L_var + λ L_smooth over trainable parameters (a, b, and optionally boundaries).

3) Visualize: visualize_boundaries_and_frames.py

Goal: Summarize learned parameters and show qualitative improvements.

- Parameter overlays:
  Plot events (x vs t and y vs t) with normalized time t − min(t), and overlay boundary lines T_i(x, y_center) and T_i(x_center, y) computed from the learned parameters and the actual event duration (ensuring proper overlap with normalized time).

- Time-binned frame comparisons:
  Build two sets of frames using the same temporal windows for both original t and compensated t′_FAST:

  • 50 ms bin width, 2 ms shift (sliding) – highlights coarse evolution.
  • 2 ms bin width, 2 ms shift (sliding) – a fine “derivative-like” view.

  For each set, compare original vs compensated per-bin frames and summary statistics (means, counts, standard deviation).

Relation between sliding (2 ms) and cumulative views

If F(T) is the per-pixel average of the cumulative frame formed by all events up to time T, then the finite difference

    ΔF(T) ≈ [F(T) − F(T − Δ)] / Δ

approximates the derivative of the cumulative signal. The 2 ms sliding frames (bin_width = shift = 2 ms) act like this finite difference: they isolate what changed within [T − 2 ms, T), which is comparable to the temporal derivative of the keep-accumulating view.

Quick Start

1) Segment a RAW file into 6 scans (forward/backward):

    python segment_robust_fixed.py <path/to/file.raw> \
      --segment_events --output_dir <output_dir>

   Optionally pass a manual round-trip period (bins @ 1 ms):

    python segment_robust_fixed.py <file.raw> \
      --segment_events --output_dir <out> --round_trip_period 1688

2) Train compensation on a segment (e.g., Scan_1_Forward):

    python compensate_multiwindow_train_saved_params.py <segment.npz> \
      --bin_width 50000 --plot_params --visualize --a_trainable \
      --iterations 1000 --b_default 0 --smoothness_weight 0.001

3) Visualize results with overlays and time-binned comparisons:

    python visualize_boundaries_and_frames.py <segment.npz>

The visualization automatically loads the most recent learned parameter NPZ co-located with the segment file.
