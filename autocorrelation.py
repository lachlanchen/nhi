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

    for lag in tqdm(range(-time_dim + 1, time_dim), desc="Calculating Original-Reverse correlation"):
        if lag == 0:
            corr = torch.mean(tensor * reverse_tensor)
        elif lag > 0:
            corr = torch.mean(tensor[:-lag] * reverse_tensor[lag:])
        else:
            corr = torch.mean(tensor[-lag:] * reverse_tensor[:lag])
        original_reverse_correlation.append(corr.item())
    return original_reverse_correlation

# def find_top_peaks(correlation, num_peaks=5):
#     indices = np.argpartition(correlation, -num_peaks)[-num_peaks:]
#     # print(indices)
#     # indices = np.array(indices, dtype=int)
#     # sorted_indices = indices[np.argsort(correlation[indices])[::-1]]
#     # return sorted_indices
#     return indices
# def find_top_peaks(correlation, num_peaks=5, sort=True):
#     print("Initial correlation:", correlation)
    
#     # Step 1: Find the indices of the largest `num_peaks` values using np.argpartition
#     partitioned_indices = np.argpartition(correlation, -num_peaks)
#     print("Indices after argpartition (unsorted):", partitioned_indices)
    
#     # Step 2: Select the top `num_peaks` indices from the partitioned result
#     top_indices = partitioned_indices[-num_peaks:]
#     print("Top indices (unsorted):", top_indices)
    
#     if sort:
#         # Step 3: Get the correlation values for the top indices
#         top_values = np.array(correlation)[top_indices]
#         print("Correlation values for top indices:", top_values)
        
#         # Step 4: Sort these correlation values and get the sorted indices
#         sorted_top_values_indices = np.argsort(top_values)
#         print("Indices that would sort the top values:", sorted_top_values_indices)
        
#         # Step 5: Reverse the sorted order to get the highest values first
#         reversed_sorted_indices = sorted_top_values_indices[::-1]
#         print("Reversed sorted indices:", reversed_sorted_indices)
        
#         # Step 6: Map the sorted indices back to the original top indices
#         final_sorted_indices = top_indices[reversed_sorted_indices]
#         print("Final sorted top indices:", final_sorted_indices)
        
#         return final_sorted_indices
#     else:
#         return top_indices


def find_period(peaks):
    if len(peaks) < 2:
        return 0
    differences = np.diff(peaks)
    return abs(int(np.mean(differences)))

def find_top_peaks_with_period(correlation, period, num_peaks=5):
    print("Finding peaks with period:", period)
    
    peaks = []
    if period > 0:
        half_period = period // 2 

        for i in range(0, len(correlation), half_period):
            segment_start = i
            segment_end = min(i + half_period, len(correlation))
            segment = correlation[segment_start:segment_end]
            if len(segment) == 0:
                continue
            peak_index = segment_start + np.argmax(segment)
            peaks.append(peak_index)

    if len(peaks) > num_peaks:
        peaks = np.array(peaks)
        peak_values = correlation[peaks]
        top_indices = peaks[np.argsort(peak_values)[::-1][:5]]
        # top_indices = np.argpartition(peak_values, -num_peaks)[-num_peaks:]
        # sorted_indices = np.argsort(peak_values[top_indices])[::-1]
        # peaks = peaks[top_indices[sorted_indices]]
        
    else:
        top_indices = np.argpartition(correlation, -num_peaks)[-num_peaks:]

    peaks = sorted(top_indices)

    print("Peaks found:", peaks)
    return peaks

def find_top_peaks(correlation, initial_period=500, num_peaks=5):
    period = initial_period
    previous_period = -1

    while period != previous_period:
        previous_period = period
        peaks = find_top_peaks_with_period(np.array(correlation), period, num_peaks)
        period = find_period(peaks)
        print("Refined period:", period)

    return peaks


# def find_peaks_and_dips(correlation, period):
#     peaks = []
#     dips = []
#     half_period = period // 2
#     # half_period = period 
#     num_segments = len(correlation) // half_period

#     for i in range(num_segments):
#         segment_start = i * half_period
#         segment_end = segment_start + half_period
#         segment = correlation[segment_start:segment_end]
        
#         peak_idx = segment_start + np.argmax(segment)
#         dip_idx = segment_start + np.argmin(segment)

#         if len(peaks) == 0 or peak_idx > peaks[-1]:
#             peaks.append(peak_idx)
#         if len(dips) == 0 or dip_idx > dips[-1]:
#             dips.append(dip_idx)

#     return peaks, dips



def remove_close_peaks_and_dips(peaks, correlation, half_period, kind="peak"):
    refined_peaks = []
    i = 0
    while i < len(peaks):
        current_peak = peaks[i]
        j = i + 1
        while j < len(peaks) and (peaks[j] - current_peak) < half_period:
            if kind == "peak":
                if correlation[peaks[j]] > correlation[current_peak]:
                    current_peak = peaks[j]
            else:
                if correlation[peaks[j]] < correlation[current_peak]:
                    current_peak = peaks[j]
            j += 1
        refined_peaks.append(current_peak)
        i = j
    return refined_peaks

def find_largest_smallest_peaks_and_dips(correlation, peaks, dips, N=5):
    if len(peaks) < N:
        largest_peaks = peaks
    else:
        peak_values = correlation[peaks]
        largest_peak_indices = np.argpartition(peak_values, -N)[-N:]
        largest_peaks = peaks[largest_peak_indices[np.argsort(peak_values[largest_peak_indices])[::-1]]]

    if len(dips) < N:
        smallest_dips = dips
    else:
        dip_values = correlation[dips]
        smallest_dip_indices = np.argpartition(dip_values, N)[:N]
        smallest_dips = dips[smallest_dip_indices[np.argsort(dip_values[smallest_dip_indices])]]

    return largest_peaks, smallest_dips

def find_peaks_and_dips(correlation, period):
    peaks = []
    dips = []
    half_period = period // 3
    quarter_period = period // 4
    num_segments = len(correlation) // half_period

    def detect_in_segments(offset):
        segment_peaks = []
        segment_dips = []
        for i in range(num_segments):
            segment_start = i * half_period + offset
            segment_end = segment_start + half_period
            if segment_start >= len(correlation) or segment_end > len(correlation):
                continue
            segment = correlation[segment_start:segment_end]

            peak_idx = segment_start + np.argmax(segment)
            dip_idx = segment_start + np.argmin(segment)

            if len(segment_peaks) == 0 or peak_idx > segment_peaks[-1]:
                segment_peaks.append(peak_idx)
            if len(segment_dips) == 0 or dip_idx > segment_dips[-1]:
                segment_dips.append(dip_idx)
        
        return segment_peaks, segment_dips

    peaks, dips = detect_in_segments(0)
    shifted_peaks, shifted_dips = detect_in_segments(quarter_period)

    # Combine results and remove duplicates
    all_peaks = sorted(set(peaks + shifted_peaks))
    all_dips = sorted(set(dips + shifted_dips))

    refined_peaks = remove_close_peaks_and_dips(all_peaks, correlation, half_period, "peak")
    refined_dips = remove_close_peaks_and_dips(all_dips, correlation, half_period, "dip")

    refined_peaks, refined_dips = find_largest_smallest_peaks_and_dips(np.array(correlation), np.array(refined_peaks), np.array(refined_dips))

    return sorted(refined_peaks), sorted(refined_dips)

def plot_correlations(lags, autocorrelation, reverse_original_correlation, original_reverse_correlation, peaks_auto, peaks_reverse_original, dips_reverse_original, peaks_original_reverse, dips_original_reverse, output_path):
    plt.figure(figsize=(10, 6))
    plt.xlim(-len(autocorrelation)//2-100, len(autocorrelation)//2+100)
    plt.ylim(-0.0025, 0.01)
    plt.plot(lags, autocorrelation, label='Autocorrelation', alpha=0.5)
    plt.plot(lags, reverse_original_correlation, label='Reverse Original Correlation', alpha=0.5)
    plt.plot(lags, original_reverse_correlation, label='Original Reverse Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_auto], [autocorrelation[p] for p in peaks_auto], color='red', marker='o', label='Peaks Autocorrelation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_reverse_original], [reverse_original_correlation[p] for p in peaks_reverse_original], color='red', marker='x', label='Peaks Reverse-Original Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in dips_reverse_original], [reverse_original_correlation[m] for m in dips_reverse_original], color='blue', marker='x', label='Mins Reverse-Original Correlation', alpha=0.5)
    plt.scatter([lags[p] for p in peaks_original_reverse], [original_reverse_correlation[p] for p in peaks_original_reverse], color='red', marker='s', label='Peaks Original-Reverse Correlation', alpha=0.5)
    plt.scatter([lags[m] for m in dips_original_reverse], [original_reverse_correlation[m] for m in dips_original_reverse], color='blue', marker='s', label='Mins Original-Reverse Correlation', alpha=0.5)

    plt.title('Autocorrelation and Reverse-Original Correlation along time dimension')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved autocorrelation and reverse correlation plot to {output_path}")

def save_periods_info(
    peaks_auto, 
    peaks_reverse_original, dips_reverse_original, 
    peaks_original_reverse, dips_original_reverse, 
    lags, 
    autocorrelation, reverse_original_correlation, original_reverse_correlation, 
    prelude, aftermath,
    period,
    output_path, 
):
    with open(output_path, 'w') as f:
        f.write("Peak indices for Autocorrelation:\n")
        f.write(", ".join(map(str, peaks_auto)) + "\n\n")
        f.write("Autocorrelation values at peaks:\n")
        f.write(", ".join(map(str, [autocorrelation[p] for p in peaks_auto])) + "\n\n")
        
        f.write("Peak indices for Reverse-Original Correlation:\n")
        f.write(", ".join(map(str, peaks_reverse_original)) + "\n\n")
        f.write("Reverse-Original Correlation values at peaks:\n")
        f.write(", ".join(map(str, [reverse_original_correlation[p] for p in peaks_reverse_original])) + "\n\n")
        
        f.write("Min indices for Reverse-Original Correlation:\n")
        f.write(", ".join(map(str, dips_reverse_original)) + "\n\n")
        f.write("Reverse-Original Correlation values at dips:\n")
        f.write(", ".join(map(str, [reverse_original_correlation[m] for m in dips_reverse_original])) + "\n\n")

        f.write("Peak indices for Original-Reverse Correlation:\n")
        f.write(", ".join(map(str, peaks_original_reverse)) + "\n\n")
        f.write("Original-Reverse Correlation values at peaks:\n")
        f.write(", ".join(map(str, [original_reverse_correlation[p] for p in peaks_original_reverse])) + "\n\n")

        f.write("Min indices for Original-Reverse Correlation:\n")
        f.write(", ".join(map(str, dips_original_reverse)) + "\n\n")
        f.write("Original-Reverse Correlation values at dips:\n")
        f.write(", ".join(map(str, [original_reverse_correlation[m] for m in dips_original_reverse])) + "\n\n")

        if len(peaks_auto) > 1:
            periods_auto = np.diff([lags[p] for p in peaks_auto])
            half_periods_auto = periods_auto / 2
            f.write("Full periods (Autocorrelation):\n")
            f.write(", ".join(map(str, periods_auto)) + "\n\n")
            f.write("Half periods (Autocorrelation):\n")
            f.write(", ".join(map(str, half_periods_auto)) + "\n\n")

        if len(peaks_reverse_original) > 1:
            periods_reverse = np.diff([lags[p] for p in peaks_reverse_original])
            half_periods_reverse = periods_reverse / 2
            f.write("Full periods (Reverse-Original Correlation):\n")
            f.write(", ".join(map(str, periods_reverse)) + "\n\n")
            f.write("Half periods (Reverse-Original Correlation):\n")
            f.write(", ".join(map(str, half_periods_reverse)) + "\n\n")

        if len(peaks_original_reverse) > 1:
            periods_original_reverse = np.diff([lags[p] for p in peaks_original_reverse])
            half_periods_original_reverse = periods_original_reverse / 2
            f.write("Full periods (Original-Reverse Correlation):\n")
            f.write(", ".join(map(str, periods_original_reverse)) + "\n\n")
            f.write("Half periods (Original-Reverse Correlation):\n")
            f.write(", ".join(map(str, half_periods_original_reverse)) + "\n\n")

        if len(peaks_auto) <= 1:
            f.write("Not enough peaks to determine periods (Autocorrelation).\n")
        if len(peaks_reverse_original) <= 1:
            f.write("Not enough peaks to determine periods (Reverse-Original Correlation).\n")
        if len(peaks_original_reverse) <= 1:
            f.write("Not enough peaks to determine periods (Original-Reverse Correlation).\n")
        
        f.write(f"period: {period}\nprelude: {prelude}\naftermath: {aftermath}")
    print(f"Saved periods information to {output_path}")

def calculate_periphery(first_peak_auto, first_dip_reverse, period):
    return first_dip_reverse/2 - period/2, first_peak_auto - first_dip_reverse/2 - period/2

def generate_video(tensor1, tensor2, correlation, correlation_label, output_dir, tensor_filename):
    tensor1 = tensor1.mean(dim=1)
    tensor2 = tensor2.mean(dim=1)
    time_dim = tensor1.shape[0]
    height_dim = tensor1.shape[1]

    # Ensure height is divisible by 2
    if height_dim % 2 != 0:
        height_dim -= 1
        tensor1 = tensor1[:, :height_dim]
        tensor2 = tensor2[:, :height_dim]

    max_abs_val = torch.max(torch.abs(tensor1)).item()

    video_folder = os.path.join(output_dir, f"{tensor_filename}_video_frames_{correlation_label.replace(' ', '_')}")
    os.makedirs(video_folder, exist_ok=True)

    for lag in tqdm(range(-time_dim + 1, time_dim, 10), desc=f"Generating video frames for {correlation_label}"):
        fig, axs = plt.subplots(2, 1, figsize=(16, 8))

        # Tensor projection plot
        combined_tensor = np.zeros((height_dim, time_dim * 3))
        combined_tensor[:, time_dim:time_dim*2] += tensor1.cpu().numpy().T
        combined_tensor[:, time_dim+lag:2*time_dim+lag] += tensor2.cpu().numpy().T

        axs[0].imshow(combined_tensor, cmap='bwr', vmin=-max_abs_val, vmax=max_abs_val, aspect='auto')
        axs[0].set_title(f'Tensor Projection with Lag {lag}')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Height')
        axs[0].axvline(x=time_dim, color='k', linestyle='--')
        axs[0].axvline(x=2*time_dim, color='k', linestyle='--')
        axs[0].axvline(x=2*time_dim+lag, color='r', linestyle='--')


        # Correlation plot
        lags = np.arange(-time_dim + 1, time_dim)
        axs[1].plot(lags, correlation, label=correlation_label, alpha=0.5)
        axs[1].axvline(x=lag, color='r', linestyle='--', label='Current Lag')
        axs[1].set_title(f'{correlation_label} with Current Lag')
        axs[1].set_xlabel('Lag')
        axs[1].set_ylabel('Correlation')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_xlim(-len(correlation), len(correlation)//2)


        frame_filename = os.path.join(video_folder, f'frame_{lag + time_dim:04d}.png')
        plt.savefig(frame_filename, bbox_inches='tight')
        plt.close()

    # # Compile video using ffmpeg
    # video_path = os.path.join(output_dir, f"{tensor_filename}_{correlation_label.replace(' ', '_')}_correlation_video.mp4")
    # command = [
    #     'ffmpeg', '-y',
    #     '-framerate', '30',
    #     '-i', os.path.join(video_folder, 'frame_%04d.png'),
    #     '-c:v', 'libx264',
    #     '-pix_fmt', 'yuv420p',
    #     video_path
    # ]
    # subprocess.run(command, check=True)
    # print(f"Compiled video saved to {video_path}")

# def determine_periphery(tensor_path, sample_rate, output_dir=None, device=None):
#     tensor = load_tensor(tensor_path, sample_rate, device)
#     autocorrelation = calculate_autocorrelation(tensor)
#     reverse_original_correlation = calculate_reverse_original_correlation(tensor)
#     original_reverse_correlation = calculate_original_reverse_correlation(tensor)

#     lags = np.arange(-len(autocorrelation) // 2 + 1, len(autocorrelation) // 2 + 1)
#     peaks_auto = find_top_peaks(autocorrelation)

#     if len(peaks_auto) > 1:
#         period = abs(np.diff(peaks_auto).mean().astype(int))
#     else:
#         period = len(autocorrelation) // 10  # Fallback period estimate

#     peaks_reverse_original, dips_reverse_original = find_peaks_and_dips(reverse_original_correlation, period=period)
#     peaks_original_reverse, dips_original_reverse = find_peaks_and_dips(original_reverse_correlation, period=period)

#     output_dir = output_dir or os.path.dirname(tensor_path)
#     os.makedirs(output_dir, exist_ok=True)

#     tensor_filename = os.path.basename(tensor_path).replace('.pt', '')
#     sample_rate_suffix = f"_sample_rate_{sample_rate}" if sample_rate > 1 else ""
#     output_filename = f"{tensor_filename}_correlations{sample_rate_suffix}.png"
#     output_path = os.path.join(output_dir, output_filename)

#     plot_correlations(
#         lags, autocorrelation, 
#         reverse_original_correlation, original_reverse_correlation, 
#         peaks_auto, 
#         peaks_reverse_original, dips_reverse_original, 
#         peaks_original_reverse, dips_original_reverse, 
#         output_path
#     )

#     periods_info_filename = f"{tensor_filename}_periods_info{sample_rate_suffix}.txt"
#     periods_info_path = os.path.join(output_dir, periods_info_filename)

#     # Calculate the desired value
#     first_peak_auto = peaks_auto[0]
#     first_dip_reverse = dips_reverse_original[0]
#     # for i, dip in enumerate(dips_reverse_original):
#     #     print("dip: ", dip)
#     #     print("period: ", period)
#     #     if dip - period > 0:
#     #         first_dip_reverse = dips_reverse_original[i]
#     #         break


#     prelude, aftermath = calculate_periphery(first_peak_auto, first_dip_reverse, period)
#     print(f"prelude: {prelude}\naftermath: {aftermath}")

#     save_periods_info(
#         peaks_auto, 
#         peaks_reverse_original, dips_reverse_original, 
#         peaks_original_reverse, dips_original_reverse, 
#         lags, 
#         autocorrelation, 
#         reverse_original_correlation, 
#         original_reverse_correlation, 
#         periods_info_path, 
#         prelude, aftermath
#     )

#     # Generate and save the videos
#     # generate_video(tensor, tensor, autocorrelation, 'Autocorrelation', output_dir, tensor_filename)
#     # generate_video(tensor, torch.flip(tensor, [0]), reverse_original_correlation, 'Reverse Original Correlation', output_dir, tensor_filename)
#     # generate_video(torch.flip(tensor, [0]), tensor, original_reverse_correlation, 'Original Reverse Correlation', output_dir, tensor_filename)

#     return prelude, aftermath, period


def determine_periphery(tensor_path, sample_rate, output_dir=None, device=None):
    output_dir = output_dir or os.path.dirname(tensor_path)
    os.makedirs(output_dir, exist_ok=True)

    
    tensor_filename = os.path.basename(tensor_path).replace('.pt', '')
    sample_rate_suffix = f"_sample_rate_{sample_rate}" if sample_rate > 1 else ""
    periods_info_filename = f"{tensor_filename}_periods_info{sample_rate_suffix}.txt"
    periods_info_path = os.path.join(output_dir, periods_info_filename)
    
    if os.path.exists(periods_info_path):
        with open(periods_info_path, 'r') as f:
            content = f.read().splitlines()
            prelude = float(content[-2].split(': ')[1])
            aftermath = float(content[-1].split(': ')[1])
            period = float(content[-3].split(': ')[1])
            return prelude, aftermath, period
    
    tensor = load_tensor(tensor_path, sample_rate, device)
    autocorrelation = calculate_autocorrelation(tensor)
    reverse_original_correlation = calculate_reverse_original_correlation(tensor)
    original_reverse_correlation = calculate_original_reverse_correlation(tensor)

    lags = np.arange(-len(autocorrelation) // 2 + 1, len(autocorrelation) // 2 + 1)
    peaks_auto = find_top_peaks(autocorrelation)

    if len(peaks_auto) > 1:
        period = abs(np.diff(peaks_auto).mean().astype(int))
    else:
        period = len(autocorrelation) // 10  # Fallback period estimate

    peaks_reverse_original, dips_reverse_original = find_peaks_and_dips(reverse_original_correlation, period=period)
    peaks_original_reverse, dips_original_reverse = find_peaks_and_dips(original_reverse_correlation, period=period)

    output_dir = output_dir or os.path.dirname(tensor_path)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{tensor_filename}_correlations{sample_rate_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)

    plot_correlations(
        lags, autocorrelation, 
        reverse_original_correlation, original_reverse_correlation, 
        peaks_auto, 
        peaks_reverse_original, dips_reverse_original, 
        peaks_original_reverse, dips_original_reverse, 
        output_path
    )

    first_peak_auto = peaks_auto[0]
    first_dip_reverse = dips_reverse_original[0]
    
    prelude, aftermath = calculate_periphery(first_peak_auto, first_dip_reverse, period)
    print(f"period: {period},  prelude: {prelude}\naftermath: {aftermath}")

    save_periods_info(
        peaks_auto, 
        peaks_reverse_original, dips_reverse_original, 
        peaks_original_reverse, dips_original_reverse, 
        lags, 
        autocorrelation, 
        reverse_original_correlation, 
        original_reverse_correlation, 
        prelude, aftermath,
        period,
        periods_info_path, 
    )

    return prelude, aftermath, period

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    determine_prephery(tensor_path, sample_rate, output_dir=args.output_dir, device=device)