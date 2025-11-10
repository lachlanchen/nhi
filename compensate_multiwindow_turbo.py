#!/usr/bin/env python3
"""
Turbo wrapper for multi‑window compensation that can merge one or more
segmented scans (Forward/Backward) and run the existing trainer on the
combined event stream.

Key features
- Accept a single segment, a directory of segments, or an explicit list.
- For Backward scans, flip polarity and reverse time before merging.
  - If p ∈ {0,1} → map p := 1-p
  - If p ∈ {−1,1} → map p := −p
- Concatenate per‑scan relative time axes into a single continuous timeline.
- Forward all extra flags to compensate_multiwindow_train_saved_params.py.

This script intentionally reuses the proven trainer to avoid divergence, while
reducing overhead from manual pre‑merging and offering predictable behavior for
multi‑scan inputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge scan segments and run multi-window compensation on the combined stream.")
    ap.add_argument("--segment", type=Path, default=None, help="Single segment NPZ to use (if not using --segments-dir or --segments)")
    ap.add_argument("--segments", type=Path, nargs="*", default=None, help="Explicit list of segment NPZ files (Forward/Backward)")
    ap.add_argument("--segments-dir", type=Path, default=None, help="Directory containing multiple Scan_*_{Forward,Backward}_events.npz")
    ap.add_argument("--include", choices=("all", "forward", "backward"), default="all", help="Which directions to include")
    ap.add_argument("--sort", choices=("name", "time"), default="name", help="Order segments before merge (filename natural order or NPZ start_time)")
    ap.add_argument("--bin-width", type=float, default=50000.0, help="Bin width (μs) forwarded to trainer")
    ap.add_argument("--trainer", type=Path, default=Path("compensate_multiwindow_train_saved_params.py"), help="Path to base trainer script")
    ap.add_argument("--output-dir", type=Path, default=None, help="Optional destination directory for combined NPZ and trainer outputs")
    ap.add_argument("--load-params", type=Path, default=None, help="Optional learned params file to reuse; skips training")
    # Two ways to pass through extra args to the trainer:
    # 1) Use positional remainder after a '--' separator (preferred in docs)
    # 2) Or via --extra <args...>
    ap.add_argument('extras', nargs=argparse.REMAINDER)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Any extra flags to pass through to the trainer")
    return ap.parse_args()


def list_segments(segment: Path | None, segments: List[Path] | None, segments_dir: Path | None) -> List[Path]:
    if segment and segment.exists():
        return [segment.resolve()]
    if segments:
        return [p.resolve() for p in segments if p.exists()]
    if segments_dir and segments_dir.exists():
        paths = sorted(segments_dir.glob("Scan_*_*_events.npz"))
        if not paths:
            # also consider any *.npz for flexibility
            paths = sorted(segments_dir.glob("*.npz"))
        return [p.resolve() for p in paths]
    raise FileNotFoundError("No segments found. Provide --segment, --segments, or --segments-dir")


def is_backward(path: Path, meta_direction: str | None) -> bool:
    name = path.name.lower()
    if "backward" in name:
        return True
    if "forward" in name:
        return False
    if meta_direction:
        return meta_direction.lower().startswith("back")
    return False


def normalize_scan(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, backward: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return x,y,dt,p for one scan, with dt starting at 0 and increasing,
    flipping polarity and reversing time if backward.
    Returns also the duration in μs (int)."""
    t0 = int(np.min(t))
    t1 = int(np.max(t))
    dt = t.astype(np.int64) - t0
    duration = int(t1 - t0)
    pp = p.astype(np.int16, copy=True)
    if backward:
        # flip polarity first
        pmin, pmax = int(pp.min()), int(pp.max())
        if pmin >= 0 and pmax <= 1:
            # 0/1 encoding → swap 0↔1: p' = 1 - p
            pp = (1 - pp).astype(np.int16, copy=False)
        else:
            # assume −1/1 → multiply by −1
            pp = (-pp).astype(np.int16, copy=False)
        # reverse time within the scan window
        dt = duration - dt
    return x, y, dt.astype(np.int64), pp, duration


def merge_segments(paths: Iterable[Path], include: str, sort_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    items = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            direction = None
            if "direction" in d:
                try:
                    direction = str(d["direction"])  # may be numpy string
                except Exception:
                    direction = None
            back = is_backward(p, direction)
            if include == "forward" and back:
                continue
            if include == "backward" and (not back):
                continue
            start_time = int(d["start_time"]) if "start_time" in d else int(np.min(d["t"]))
            items.append((p, back, start_time))

    if not items:
        raise FileNotFoundError("No usable segment files after filtering")

    if sort_key == "time":
        items.sort(key=lambda tup: tup[2])
    else:
        items.sort(key=lambda tup: tup[0].name)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    offset = 0
    for p, back, _ in items:
        with np.load(p, allow_pickle=False) as d:
            x, y, t, p_arr = d["x"], d["y"], d["t"], d["p"]
        x, y, dt, pp, dur = normalize_scan(x, y, t, p_arr, back)
        ts.append(dt + offset)
        xs.append(x)
        ys.append(y)
        ps.append(pp)
        offset += dur + 1  # keep scans separated by 1μs to preserve ordering

    x_all = np.concatenate(xs).astype(np.uint16)
    y_all = np.concatenate(ys).astype(np.uint16)
    t_all = np.concatenate(ts).astype(np.int64)
    p_all = np.concatenate(ps).astype(np.int16)
    duration = int(t_all.max() - t_all.min()) if t_all.size else 0
    return x_all, y_all, t_all, p_all, duration


def save_combined_npz(out_path: Path, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        x=x, y=y, t=t, p=p,
        start_time=int(t.min()) if t.size else 0,
        end_time=int(t.max()) if t.size else 0,
        duration_us=int((int(t.max()) - int(t.min())) if t.size else 0),
        event_count=int(x.size),
        direction="Combined",
        scan_id="combined",
    )
    np.savez(out_path, **payload)


def main() -> None:
    args = parse_args()
    segs = list_segments(args.segment, args.segments, args.segments_dir)
    x, y, t, p, duration = merge_segments(segs, args.include, args.sort)

    # Destination for the combined file
    base_dir = (args.output_dir or segs[0].parent).resolve()
    combined_npz = base_dir / "combined_events.npz"
    save_combined_npz(combined_npz, x, y, t, p)

    # Build trainer command
    cmd = [sys.executable, str(args.trainer.resolve()), str(combined_npz), "--bin_width", str(float(args.bin_width))]
    if args.load_params:
        cmd += ["--load_params", str(args.load_params.resolve())]
    passthrough: list[str] = []
    if args.extra:
        passthrough.extend([e for e in args.extra if e != "--"])
    if args.extras:
        passthrough.extend([e for e in args.extras if e != "--"])
    if passthrough:
        cmd += passthrough

    # Use same environment; let the trainer handle GPU/plots
    print("Running trainer:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
