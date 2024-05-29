import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

def calculate_reverse_correlation(tensor):
    reverse_tensor = torch.flip(tensor, [0])  # Reverse along the time dimension
    time_dim = tensor.shape[0]
    reverse_correlation = []

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating reverse correlation"):
        if lag == 0:
            corr = torch.mean(reverse_tensor * tensor)
        elif lag > 0:
            corr = torch.mean(reverse_tensor[:-lag] * tensor[lag:])
        else:
            corr = torch.mean(reverse_tensor[-lag:] * tensor[:lag])
        reverse_correlation.append(corr.item())
    return reverse_correlation

def calculate_leading_region_correlation(tensor):
    reverse_tensor = torch.flip(tensor, [0])  # Reverse along the time dimension
    time_dim = tensor.shape[0]
    leading_region_correlation = []

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating leading region correlation"):
        if lag == 0:
            corr = torch.mean(tensor * reverse_tensor)
        elif lag > 0:
            corr = torch.mean(tensor[:-lag] * reverse_tensor[lag:])
        else:
            corr = torch.mean(tensor[-lag:] * reverse_tensor[:lag])
        leading_region_correlation.append(corr.item())
    return leading_region_correlation

def find_peaks_with_threshold(correlation, threshold=0.004):
    peaks = [i for i in range(1, len(correlation) - 1) if correlation[i] > threshold and correlation[i] > correlation[i - 1] and correlation[i] > correlation[i + 1]]
    return peaks

def find_peaks_and_mins_with_threshold(correlation, threshold_max=0.0015, threshold_min=-0.0015, period=0):
# def find_peaks_and_mins_with_threshold(correlation, threshold_max=0.00, threshold_min=-0.00, period=0):
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

# def find_peaks_and_mins_with_threshold(correlation, threshold_max=0.0015, threshold_min=-0.0015, period=0):
#     peaks = []
#     mins = []
#     half_period = period // 2
#     for i in range(half_period, len(correlation), half_period):
#         segment = correlation[i-half_period:i+half_period]
#         peak = max(segment, default=0)
#         min_val = min(segment, default=0)
#         peak_idx = i - half_period + np.argmax(segment)
#         min_idx = i - half_period + np.argmin(segment)
#         peaks.append(peak_idx)
#         mins.append(min_idx)
#     return peaks, mins


def plot_correlations(lags, autocorrelation, reverse_correlation, leading_region_correlation, peaks_auto, peaks_reverse, mins_reverse, peaks_leading, mins_leading, output_path):
    plt.figure(figsize=(10, 6))
    plt.xlim(-2600, 2600)
    plt.ylim(-0.0025, 0.01)
    plt.plot(lags, autocorrelation, label='Autocorrelation', alpha=0.5)
    plt.plot(lags, reverse_correlation, label='Reverse Correlation', alpha=0.5)
    plt.plot(lags, leading_region_correlation, label='Leading Region Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_auto], [autocorrelation[p] for p in peaks_auto], color='red', marker='o', label='Peaks Autocorrelation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_reverse], [reverse_correlation[p] for p in peaks_reverse], color='blue', marker='x', label='Peaks Reverse Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in mins_reverse], [reverse_correlation[m] for m in mins_reverse], color='blue', marker='x', label='Mins Reverse Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_leading], [leading_region_correlation[p] for p in peaks_leading], color='green', marker='s', label='Peaks Leading Region Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in mins_leading], [leading_region_correlation[m] for m in mins_leading], color='green', marker='s', label='Mins Leading Region Correlation', alpha=0.5)

    plt.title('Autocorrelation and Reverse Correlation along time dimension')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved autocorrelation and reverse correlation plot to {output_path}")

def save_periods_info(peaks_auto, peaks_reverse, mins_reverse, peaks_leading, mins_leading, lags, autocorrelation, reverse_correlation, leading_region_correlation, output_path):
    with open(output_path, 'w') as f:
        f.write("Peak indices for Autocorrelation:\n")
        f.write(", ".join(map(str, peaks_auto)) + "\n\n")
        f.write("Autocorrelation values at peaks:\n")
        f.write(", ".join(map(str, [autocorrelation[p] for p in peaks_auto])) + "\n\n")
        
        f.write("Peak indices for Reverse Correlation:\n")
        f.write(", ".join(map(str, peaks_reverse)) + "\n\n")
        f.write("Reverse Correlation values at peaks:\n")
        f.write(", ".join(map(str, [reverse_correlation[p] for p in peaks_reverse])) + "\n\n")
        
        f.write("Min indices for Reverse Correlation:\n")
        f.write(", ".join(map(str, mins_reverse)) + "\n\n")
        f.write("Reverse Correlation values at mins:\n")
        f.write(", ".join(map(str, [reverse_correlation[m] for m in mins_reverse])) + "\n\n")

        f.write("Peak indices for Leading Region Correlation:\n")
        f.write(", ".join(map(str, peaks_leading)) + "\n\n")
        f.write("Leading Region Correlation values at peaks:\n")
        f.write(", ".join(map(str, [leading_region_correlation[p] for p in peaks_leading])) + "\n\n")

        f.write("Min indices for Leading Region Correlation:\n")
        f.write(", ".join(map(str, mins_leading)) + "\n\n")
        f.write("Leading Region Correlation values at mins:\n")
        f.write(", ".join(map(str, [leading_region_correlation[m] for m in mins_leading])) + "\n\n")

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

        if len(peaks_leading) > 1:
            periods_leading = np.diff([lags[p] for p in peaks_leading])
            half_periods_leading = periods_leading / 2
            f.write("Full periods (Leading Region Correlation):\n")
            f.write(", ".join(map(str, periods_leading)) + "\n\n")
            f.write("Half periods (Leading Region Correlation):\n")
            f.write(", ".join(map(str, half_periods_leading)) + "\n\n")

        if len(peaks_auto) <= 1:
            f.write("Not enough peaks to determine periods (Autocorrelation).\n")
        if len(peaks_reverse) <= 1:
            f.write("Not enough peaks to determine periods (Reverse Correlation).\n")
        if len(peaks_leading) <= 1:
            f.write("Not enough peaks to determine periods (Leading Region Correlation).\n")
    print(f"Saved periods information to {output_path}")

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = load_tensor(args.tensor_path, args.sample_rate, device)
    autocorrelation = calculate_autocorrelation(tensor)
    reverse_correlation = calculate_reverse_correlation(tensor)
    leading_region_correlation = calculate_leading_region_correlation(tensor)

    lags = np.arange(-len(autocorrelation) // 2 + 1, len(autocorrelation) // 2 + 1)
    peaks_auto = find_peaks_with_threshold(autocorrelation, threshold=0.004)

    if len(peaks_auto) > 1:
        period = np.diff(peaks_auto).mean().astype(int)
    else:
        period = len(autocorrelation) // 10  # Fallback period estimate

    peaks_reverse, mins_reverse = find_peaks_and_mins_with_threshold(reverse_correlation, threshold_max=0.0015, threshold_min=-0.0015, period=period)
    peaks_leading, mins_leading = find_peaks_and_mins_with_threshold(leading_region_correlation, threshold_max=0.0015, threshold_min=-0.0015, period=period)

    output_dir = args.output_dir or os.path.dirname(args.tensor_path)
    os.makedirs(output_dir, exist_ok=True)

    tensor_filename = os.path.basename(args.tensor_path).replace('.pt', '')
    sample_rate_suffix = f"_sample_rate_{args.sample_rate}" if args.sample_rate > 1 else ""
    output_filename = f"{tensor_filename}_correlations{sample_rate_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)

    plot_correlations(lags, autocorrelation, reverse_correlation, leading_region_correlation, peaks_auto, peaks_reverse, mins_reverse, peaks_leading, mins_leading, output_path)

    periods_info_filename = f"{tensor_filename}_periods_info{sample_rate_suffix}.txt"
    periods_info_path = os.path.join(output_dir, periods_info_filename)

    save_periods_info(peaks_auto, peaks_reverse, mins_reverse, peaks_leading, mins_leading, lags, autocorrelation, reverse_correlation, leading_region_correlation, periods_info_path)
