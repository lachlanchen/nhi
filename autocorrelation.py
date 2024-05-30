import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate and plot autocorrelation of tensor data along time dimension.')
    parser.add_argument('tensor_path', help='Path to the tensor file.')
    parser.add_argument('--sample_rate', type=int, default=1, help='Downsample rate (default: 1, no downsampling).')
    parser.add_argument('--output_dir', default=None, help='Directory to save the plot and period information. Defaults to the tensor file\'s directory.')
    return parser.parse_args()

def load_tensor(tensor_path, sample_rate, device):
    tensor = torch.load(tensor_path, map_location=device)
    if sample_rate > 1:
        tensor = tensor[:, ::sample_rate, ::sample_rate]
    return tensor

def calculate_autocorrelation(tensor):
    time_dim = tensor.shape[0]
    autocorrelation = []

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating autocorrelation"):
        if lag == 0:
            autocorr = torch.mean(tensor * tensor)
        elif lag > 0:
            autocorr = torch.mean(tensor[:-lag] * tensor[lag:])
        else:
            autocorr = torch.mean(tensor[-lag:] * tensor[:lag])
        autocorrelation.append(autocorr.item())
    return autocorrelation

def calculate_reverse_original_correlation(tensor):
    reverse_tensor = torch.flip(tensor, [0])  # Reverse along the time dimension
    time_dim = tensor.shape[0]
    reverse_original_correlation = []

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating reverse correlation"):
        if lag == 0:
            corr = torch.mean(reverse_tensor * tensor)
        elif lag > 0:
            corr = torch.mean(reverse_tensor[:-lag] * tensor[lag:])
        else:
            corr = torch.mean(reverse_tensor[-lag:] * tensor[:lag])
        reverse_original_correlation.append(corr.item())
    return reverse_original_correlation

def calculate_original_reverse_correlation(tensor):
    reverse_tensor = torch.flip(tensor, [0])  # Reverse along the time dimension
    time_dim = tensor.shape[0]
    original_reverse_correlation = []

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating ending region correlation"):
        if lag == 0:
            corr = torch.mean(tensor * reverse_tensor)
        elif lag > 0:
            corr = torch.mean(tensor[:-lag] * reverse_tensor[lag:])
        else:
            corr = torch.mean(tensor[-lag:] * reverse_tensor[:lag])
        original_reverse_correlation.append(corr.item())
    return original_reverse_correlation

def find_peaks_with_threshold(correlation, threshold=0.004):
    peaks = [i for i in range(1, len(correlation) - 1) if correlation[i] > threshold and correlation[i] > correlation[i - 1] and correlation[i] > correlation[i + 1]]
    return peaks

def find_peaks_and_mins_with_threshold(correlation, threshold_max=0.0015, threshold_min=-0.0015, period=0):
    peaks = []
    mins = []
    period = period // 2
    for i in range(period, len(correlation), period):
        segment = correlation[i-period:i+period]
        peak = max(segment, default=0)
        min_val = min(segment, default=0)
        peak_idx = i - period + np.argmax(segment)
        min_idx = i - period + np.argmin(segment)
        if peak > threshold_max:
            peaks.append(peak_idx)
        if min_val < threshold_min:
            mins.append(min_idx)
    return peaks, mins

def plot_correlations(lags, autocorrelation, reverse_original_correlation, original_reverse_correlation, peaks_auto, peaks_reverse, mins_reverse, peaks_original_reverse, mins_original_reverse, output_path):
    plt.figure(figsize=(10, 6))
    plt.xlim(-2600, 2600)
    plt.ylim(-0.0025, 0.01)
    plt.plot(lags, autocorrelation, label='Autocorrelation', alpha=0.5)
    plt.plot(lags, reverse_original_correlation, label='Reverse Original Correlation', alpha=0.5)
    plt.plot(lags, original_reverse_correlation, label='Original Reverse Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_auto], [autocorrelation[p] for p in peaks_auto], color='red', marker='o', label='Peaks Autocorrelation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_reverse], [reverse_original_correlation[p] for p in peaks_reverse], color='blue', marker='x', label='Peaks Reverse Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in mins_reverse], [reverse_original_correlation[m] for m in mins_reverse], color='blue', marker='x', label='Mins Reverse Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_original_reverse], [original_reverse_correlation[p] for p in peaks_original_reverse], color='green', marker='s', label='Peaks ending Region Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in mins_original_reverse], [original_reverse_correlation[m] for m in mins_original_reverse], color='green', marker='s', label='Mins ending Region Correlation', alpha=0.5)

    plt.title('Autocorrelation and Reverse Correlation along time dimension')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved autocorrelation and reverse correlation plot to {output_path}")

def save_periods_info(peaks_auto, peaks_reverse, mins_reverse, peaks_original_reverse, mins_original_reverse, lags, autocorrelation, reverse_original_correlation, original_reverse_correlation, output_path, start, end):
    with open(output_path, 'w') as f:
        f.write("Peak indices for Autocorrelation:\n")
        f.write(", ".join(map(str, peaks_auto)) + "\n\n")
        f.write("Autocorrelation values at peaks:\n")
        f.write(", ".join(map(str, [autocorrelation[p] for p in peaks_auto])) + "\n\n")
        
        f.write("Peak indices for Reverse Correlation:\n")
        f.write(", ".join(map(str, peaks_reverse)) + "\n\n")
        f.write("Reverse Correlation values at peaks:\n")
        f.write(", ".join(map(str, [reverse_original_correlation[p] for p in peaks_reverse])) + "\n\n")
        
        f.write("Min indices for Reverse Correlation:\n")
        f.write(", ".join(map(str, mins_reverse)) + "\n\n")
        f.write("Reverse Correlation values at mins:\n")
        f.write(", ".join(map(str, [reverse_original_correlation[m] for m in mins_reverse])) + "\n\n")

        f.write("Peak indices for ending Region Correlation:\n")
        f.write(", ".join(map(str, peaks_original_reverse)) + "\n\n")
        f.write("ending Region Correlation values at peaks:\n")
        f.write(", ".join(map(str, [original_reverse_correlation[p] for p in peaks_original_reverse])) + "\n\n")

        f.write("Min indices for ending Region Correlation:\n")
        f.write(", ".join(map(str, mins_original_reverse)) + "\n\n")
        f.write("ending Region Correlation values at mins:\n")
        f.write(", ".join(map(str, [original_reverse_correlation[m] for m in mins_original_reverse])) + "\n\n")

        if len(peaks_auto) > 1:
            periods_auto = np.diff([lags[p] for p in peaks_auto])
            half_periods_auto = periods_auto / 2
            f.write("Full periods (Autocorrelation):\n")
            f.write(", ".join(map(str, periods_auto)) + "\n\n")
            f.write("Half periods (Autocorrelation):\n")
            f.write(", ".join(map(str, half_periods_auto)) + "\n\n")

        if len(peaks_reverse) > 1:
            periods_reverse = np.diff([lags[p] for p in peaks_reverse])
            half_periods_reverse = periods_reverse / 2
            f.write("Full periods (Reverse Correlation):\n")
            f.write(", ".join(map(str, periods_reverse)) + "\n\n")
            f.write("Half periods (Reverse Correlation):\n")
            f.write(", ".join(map(str, half_periods_reverse)) + "\n\n")

        if len(peaks_original_reverse) > 1:
            periods_original_reverse = np.diff([lags[p] for p in peaks_original_reverse])
            half_periods_original_reverse = periods_original_reverse / 2
            f.write("Full periods (ending Region Correlation):\n")
            f.write(", ".join(map(str, periods_original_reverse)) + "\n\n")
            f.write("Half periods (ending Region Correlation):\n")
            f.write(", ".join(map(str, half_periods_original_reverse)) + "\n\n")

        if len(peaks_auto) <= 1:
            f.write("Not enough peaks to determine periods (Autocorrelation).\n")
        if len(peaks_reverse) <= 1:
            f.write("Not enough peaks to determine periods (Reverse Correlation).\n")
        if len(peaks_original_reverse) <= 1:
            f.write("Not enough peaks to determine periods (ending Region Correlation).\n")
        
        f.write(f"\nstart: {start}\nend:{end}")
    print(f"Saved periods information to {output_path}")

def calculate_value(first_peak_auto, first_peak_reverse, period):
    return first_peak_auto - first_peak_reverse, first_peak_reverse

def generate_video(tensor, autocorrelation, reverse_original_correlation, original_reverse_correlation, output_dir, tensor_filename):
    tensor = tensor.mean(dim=1)
    time_dim = tensor.shape[0]
    height_dim = tensor.shape[1]
    max_abs_val = torch.max(torch.abs(tensor)).item()

    video_folder = os.path.join(output_dir, f"{tensor_filename}_video_frames")
    os.makedirs(video_folder, exist_ok=True)

    for lag in tqdm(range(-time_dim + 1, time_dim, 10), desc="Generating video frames"):
        fig, axs = plt.subplots(2, 1, figsize=(16, 8))

        # Tensor projection plot
        combined_tensor = np.zeros((height_dim, time_dim * 3))
        combined_tensor[:, time_dim:time_dim*2] += tensor.cpu().numpy().T
        combined_tensor[:, time_dim+lag:2*time_dim+lag] += tensor.cpu().numpy().T
        # if lag >= 0:
        #     combined_tensor[:, time_dim+lag:2*time_dim+lag] = tensor.cpu().numpy().T
        # else:
        #     combined_tensor[:, time_dim+lag:2*time_dim+lag] = tensor.cpu().numpy().T
        #     # combined_tensor[:, time_dim:time_dim-lag] = tensor[lag:].cpu().numpy().T

        axs[0].imshow(combined_tensor, cmap='bwr', vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
        axs[0].set_title(f'Tensor Projection with Lag {lag}')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Height')
        axs[0].axvline(x=time_dim, color='k', linestyle='--')
        axs[0].axvline(x=2*time_dim, color='k', linestyle='--')
        axs[0].axvline(x=2*time_dim+lag, color='r', linestyle='--')


        # Correlation plot
        lags = np.arange(-time_dim + 1, time_dim)
        axs[1].plot(lags, autocorrelation, label='Autocorrelation', alpha=0.5)
        axs[1].plot(lags, reverse_original_correlation, label='Reverse Original Correlation', alpha=0.5)
        axs[1].plot(lags, original_reverse_correlation, label='Original Reverse Correlation', alpha=0.5)
        axs[1].axvline(x=lag, color='r', linestyle='--', label='Current Lag')
        axs[1].set_title('Correlation with Current Lag')
        axs[1].set_xlabel('Lag')
        axs[1].set_ylabel('Correlation')
        axs[1].legend()
        axs[1].grid(True)

        frame_filename = os.path.join(video_folder, f'frame_{lag + time_dim:04d}.png')
        plt.savefig(frame_filename, bbox_inches='tight')
        plt.close()

    # Compile video using ffmpeg
    video_path = os.path.join(output_dir, f"{tensor_filename}_correlation_video.mp4")
    command = [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', os.path.join(video_folder, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ]
    subprocess.run(command, check=True)
    print(f"Compiled video saved to {video_path}")

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = load_tensor(args.tensor_path, args.sample_rate, device)
    autocorrelation = calculate_autocorrelation(tensor)
    reverse_original_correlation = calculate_reverse_original_correlation(tensor)
    original_reverse_correlation = calculate_original_reverse_correlation(tensor)

    lags = np.arange(-len(autocorrelation) // 2 + 1, len(autocorrelation) // 2 + 1)
    peaks_auto = find_peaks_with_threshold(autocorrelation, threshold=0.004)

    if len(peaks_auto) > 1:
        period = np.diff(peaks_auto).mean().astype(int)
    else:
        period = len(autocorrelation) // 10  # Fallback period estimate

    peaks_reverse, mins_reverse = find_peaks_and_mins_with_threshold(reverse_original_correlation, threshold_max=0.0015, threshold_min=-0.0015, period=period)
    peaks_original_reverse, mins_original_reverse = find_peaks_and_mins_with_threshold(original_reverse_correlation, threshold_max=0.0015, threshold_min=-0.0015, period=period)

    output_dir = args.output_dir or os.path.dirname(args.tensor_path)
    os.makedirs(output_dir, exist_ok=True)

    tensor_filename = os.path.basename(args.tensor_path).replace('.pt', '')
    sample_rate_suffix = f"_sample_rate_{args.sample_rate}" if args.sample_rate > 1 else ""
    output_filename = f"{tensor_filename}_correlations{sample_rate_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)

    plot_correlations(lags, autocorrelation, reverse_original_correlation, original_reverse_correlation, peaks_auto, peaks_reverse, mins_reverse, peaks_original_reverse, mins_original_reverse, output_path)

    periods_info_filename = f"{tensor_filename}_periods_info{sample_rate_suffix}.txt"
    periods_info_path = os.path.join(output_dir, periods_info_filename)

    # Calculate the desired value
    first_peak_auto = peaks_auto[0]
    first_peak_reverse = peaks_reverse[0]
    last_peak_original_reverse = 5000 - peaks_original_reverse[-1]

    start, end = calculate_value(first_peak_auto, first_peak_reverse, period)
    print(f"start: {start}\nend: {end}")
    print(f"start: {last_peak_original_reverse/2 - period/2}")

    save_periods_info(peaks_auto, peaks_reverse, mins_reverse, peaks_original_reverse, mins_original_reverse, lags, autocorrelation, reverse_original_correlation, original_reverse_correlation, periods_info_path, start, end)

    # Generate and save the video
    generate_video(tensor, autocorrelation, reverse_original_correlation, original_reverse_correlation, output_dir, tensor_filename)
