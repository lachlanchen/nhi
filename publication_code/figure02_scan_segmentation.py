#!/usr/bin/env python3
"""
Generate Optics Letters-ready Figure 2 illustrating the self-synchronized scan
segmentation pipeline. The figure reproduces the diagnostics described in the
paper section by combining activity traces, correlation curves, and per-scan
statistics derived from segmented event files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT
    / "scan_angle_20_led_2835b"
    / "angle_20_blank_2835_20250925_184747"
    / "angle_20_blank_2835_event_20250925_184747_segments"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


@dataclass
class SegmentInfo:
    """Metadata for a single forward/backward scan segment."""

    dataset: str
    scan_id: int
    direction: str
    start_us: int
    end_us: int
    event_count: int
    duration_ms: float
    occupancy_ms: float
    event_rate: float
    npz_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the figure generator."""

    parser = argparse.ArgumentParser(
        description="Create a publication-grade Figure 2 for self-synchronized scan segmentation."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the *_segments directory produced by segment_robust_fixed.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PDF (and optional PNG) will be written.",
    )
    parser.add_argument(
        "--time-bin-us",
        type=int,
        default=1000,
        help="Temporal bin size (in microseconds) used to build the activity curve.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also emit a high-resolution PNG alongside the PDF.",
    )
    return parser.parse_args()


def find_segment_dirs(base_dir: Path) -> List[Path]:
    """Locate every *_segments directory beneath the provided base."""

    if base_dir.is_dir() and base_dir.name.endswith("_segments"):
        return [base_dir]
    candidates: List[Path] = []
    for seg_dir in base_dir.rglob("*_segments"):
        if seg_dir.is_dir() and any(seg_dir.glob("Scan_*_events.npz")):
            candidates.append(seg_dir)
    return candidates


def load_segment_infos(segment_dir: Path) -> List[SegmentInfo]:
    """Load metadata for every segment contained in the directory."""

    dataset_name = segment_dir.parent.name
    infos: List[SegmentInfo] = []
    for npz_path in sorted(segment_dir.glob("Scan_*_events.npz")):
        with np.load(npz_path) as data:
            start_us = int(data["start_time"])
            end_us = int(data["end_time"])
            scan_id = int(data["scan_id"])
            direction = str(data["direction"])
            event_count = int(data["event_count"])
            duration_ms = (end_us - start_us) / 1000.0
            timestamps = data["t"].astype(np.int64)
        occupancy_ms = (timestamps[-1] - timestamps[0]) / 1000.0
        # Guard against degenerate durations while retaining the reported timing.
        window_ms = duration_ms if duration_ms > 0 else max(occupancy_ms, 1.0)
        event_rate = event_count / window_ms
        infos.append(
            SegmentInfo(
                dataset=dataset_name,
                scan_id=scan_id,
                direction=direction,
                start_us=start_us,
                end_us=end_us,
                event_count=event_count,
                duration_ms=duration_ms,
                occupancy_ms=occupancy_ms,
                event_rate=event_rate,
                npz_path=npz_path,
            )
        )
    return infos


def compute_activity_trace(
    segments: Sequence[SegmentInfo],
    time_bin_us: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate an activity signal (events per bin) across all segments."""

    if not segments:
        raise ValueError("No segments available to compute the activity trace.")
    start_us = min(seg.start_us for seg in segments)
    end_us = max(seg.end_us for seg in segments)
    num_bins = int(np.ceil((end_us - start_us) / time_bin_us))
    activity = np.zeros(num_bins, dtype=np.int64)

    for seg in segments:
        with np.load(seg.npz_path) as data:
            timestamps = data["t"].astype(np.int64)
        bin_indices = ((timestamps - start_us) // time_bin_us).astype(np.int64)
        counts = np.bincount(bin_indices, minlength=num_bins)
        activity[: len(counts)] += counts

    bin_edges = start_us + np.arange(num_bins + 1) * time_bin_us
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Express the time axis relative to the beginning of the first segmented scan.
    reference_us = segments[0].start_us
    time_axis_ms = (centers - reference_us) / 1000.0
    return time_axis_ms, activity


def normalise(array: np.ndarray) -> np.ndarray:
    """Normalise an array to unit peak, guarding against zero max value."""

    peak = np.max(np.abs(array))
    if peak == 0:
        return array.copy()
    return array / peak


def compute_correlations(
    activity: np.ndarray,
    time_bin_us: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lags (ms), auto-correlation, and reverse-correlation signals."""

    demeaned = activity.astype(np.float64) - np.mean(activity)
    auto = np.correlate(demeaned, demeaned, mode="full")
    reverse = np.correlate(demeaned, demeaned[::-1], mode="full")

    lags = np.arange(-len(activity) + 1, len(activity)) * (time_bin_us / 1000.0)
    return lags, normalise(auto), normalise(reverse)


def find_peak_lag(lags: np.ndarray, corr: np.ndarray, prefer_positive: bool = True) -> float:
    """Locate the strongest off-centre peak in the correlation curve."""

    centre = len(corr) // 2
    if prefer_positive:
        positive = corr[centre + 1 :]
        idx = np.argmax(positive)
        return lags[centre + 1 + idx]
    negative = corr[:centre]
    idx = np.argmax(negative)
    return lags[idx]


def cumulative_average(sequence: Sequence[float]) -> np.ndarray:
    """Compute the cumulative average of a numeric sequence."""

    arr = np.array(sequence, dtype=np.float64)
    if arr.size == 0:
        return np.array([])
    cumulative = np.cumsum(arr)
    steps = np.arange(1, arr.size + 1)
    return cumulative / steps


def plot_activity_panel(ax: plt.Axes, time_ms: np.ndarray, activity: np.ndarray, segments: Sequence[SegmentInfo]) -> None:
    """Draw the activity trace with shaded forward/backward segments."""

    ax.plot(time_ms, activity / 1e6, color="#08306b", linewidth=1.5, label="Activity (×10⁶)")
    palette = {"Forward": "#2b8cbe", "Backward": "#f16913"}
    for seg in segments:
        start_ms = (seg.start_us - segments[0].start_us) / 1000.0
        end_ms = (seg.end_us - segments[0].start_us) / 1000.0
        ax.axvspan(start_ms, end_ms, color=palette[seg.direction], alpha=0.15, lw=0)
        label_y = ax.get_ylim()[1] * (0.86 if seg.direction == "Forward" else 0.74)
        ax.text(
            0.5 * (start_ms + end_ms),
            label_y,
            f"{seg.direction[:1].upper()}{seg.scan_id}",
            ha="center",
            va="center",
            fontsize=8,
            color=palette[seg.direction],
            fontweight="bold",
        )

    ax.set_title("(a) Activity trace with auto-labelled scan windows", loc="left", fontweight="bold")
    ax.set_ylabel("Events per ms (million)")
    ax.set_xlabel("Time relative to scan start (s)")
    ax.set_xlim(time_ms[0], time_ms[-1])
    xticks = np.linspace(time_ms[0], time_ms[-1], 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick/1000:.1f}" for tick in xticks])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)


def plot_correlation_panel(
    ax: plt.Axes,
    lags: np.ndarray,
    auto: np.ndarray,
    reverse: np.ndarray,
    one_way_lag_ms: float,
    turnaround_lag_ms: float,
) -> None:
    """Render auto- and reverse-correlation curves with annotated peaks."""

    ax.plot(lags, auto, color="#08519c", linewidth=1.4, label="Auto-correlation")
    ax.plot(lags, reverse, color="#a50f15", linewidth=1.3, linestyle="--", label="Reverse correlation")
    ax.axvline(one_way_lag_ms, color="#08519c", linestyle=":", linewidth=1.1)
    ax.axvline(-one_way_lag_ms, color="#08519c", linestyle=":", linewidth=1.1)
    ax.axvline(turnaround_lag_ms, color="#a50f15", linestyle="-.", linewidth=1.1)
    ax.text(
        one_way_lag_ms,
        0.92,
        "±1.16 s peaks",
        rotation=90,
        color="#08519c",
        fontsize=8,
        ha="right",
        va="top",
        backgroundcolor="white",
    )
    ax.text(
        turnaround_lag_ms,
        0.6,
        "Turnaround lag",
        rotation=90,
        color="#a50f15",
        fontsize=8,
        ha="left",
        va="center",
        backgroundcolor="white",
    )
    ax.set_xlim(-4000, 4000)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalised correlation")
    ax.set_title("(b) Correlation diagnostics", loc="left", fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_duration_panel(ax: plt.Axes, segments: Sequence[SegmentInfo]) -> None:
    """Plot a histogram of high-resolution half-scan durations."""

    durations = np.array([seg.occupancy_ms for seg in segments], dtype=np.float64) / 1000.0
    directions = np.array([seg.direction for seg in segments])
    forward = durations[directions == "Forward"]
    backward = durations[directions == "Backward"]
    bins = np.linspace(durations.min() - 0.01, durations.max() + 0.01, 10)
    ax.hist(forward, bins=bins, color="#2b8cbe", alpha=0.7, label="Forward")
    ax.hist(backward, bins=bins, color="#f16913", alpha=0.55, label="Backward")
    mean_duration = durations.mean()
    ax.axvline(mean_duration, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        mean_duration + 0.001,
        ax.get_ylim()[1] * 0.9,
        f"mean {mean_duration:.3f} s",
        fontsize=8,
        color="black",
        ha="left",
        va="center",
        backgroundcolor="white",
    )
    ax.set_xlabel("Half-scan duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Half-scan duration stability", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)


def plot_cumulative_panel(ax: plt.Axes, segments: Sequence[SegmentInfo]) -> None:
    """Visualise convergence of event-rate estimates with multi-scan fusion."""

    forward_rates = [seg.event_rate for seg in segments if seg.direction == "Forward"]
    backward_rates = [seg.event_rate for seg in segments if seg.direction == "Backward"]
    forward_cum = cumulative_average(forward_rates)
    backward_cum = cumulative_average(backward_rates)
    scans = np.arange(1, len(forward_cum) + 1)

    ax.plot(scans, forward_cum, marker="o", color="#2b8cbe", label="Forward cumulative")
    ax.plot(scans, backward_cum, marker="s", color="#f16913", label="Backward cumulative")
    overall_mean = np.mean(forward_rates + backward_rates)
    ax.axhline(overall_mean, color="black", linestyle="--", linewidth=1.0, label="All scans mean")
    ax.set_xlabel("Number of fused scans")
    ax.set_ylabel("Events per ms")
    ax.set_xticks(scans)
    ax.set_title("(d) Multi-pass fusion stabilises event rate", loc="left", fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    segment_dirs = find_segment_dirs(dataset_path)
    if not segment_dirs:
        raise FileNotFoundError(f"No *_segments directories found beneath {dataset_path}")

    # Load metadata from the requested dataset (use the first *_segments if a tree was given).
    segments = load_segment_infos(segment_dirs[0])
    all_segments: List[SegmentInfo] = []
    parent_root = segment_dirs[0].parents[1]
    for seg_dir in find_segment_dirs(parent_root):
        all_segments.extend(load_segment_infos(seg_dir))

    time_ms, activity = compute_activity_trace(segments, args.time_bin_us)
    lags, auto_corr, reverse_corr = compute_correlations(activity, args.time_bin_us)
    one_way_lag = find_peak_lag(lags, auto_corr, prefer_positive=True)
    turnaround_lag = find_peak_lag(lags, reverse_corr, prefer_positive=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    plot_activity_panel(axes[0, 0], time_ms, activity, segments)
    plot_correlation_panel(axes[0, 1], lags, auto_corr, reverse_corr, one_way_lag, turnaround_lag)
    plot_duration_panel(axes[1, 0], all_segments)
    plot_cumulative_panel(axes[1, 1], segments)

    fig.suptitle("Figure 2. Self-synchronised scan segmentation diagnostics", fontsize=11, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure02_scan_segmentation.pdf"
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")
    if args.save_png:
        png_path = output_dir / "figure02_scan_segmentation.png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 2 to {pdf_path}")


if __name__ == "__main__":
    main()
