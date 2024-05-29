import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint
import gc

from integer_shifter import TensorShifter
from block_visualizer import BlockVisualizer
from events2spectrum import EventsSpectrumReconstructor  # Import the EventsSpectrumReconstructor class
from spectrum_visualizer import SpectrumVisualizer  # Import the SpectrumVisualizer class


def apply_mean_kernel(tensor, kernel_size):
    # Create a kernel filled with the appropriate mean value (1/27 for a 3x3x3 kernel)
    kernel_volume = kernel_size ** 3
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=tensor.device) / kernel_volume
    
    # Apply 3D convolution to compute the means with padding to maintain dimensions
    padding = kernel_size // 2
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    result_tensor = F.conv3d(tensor, kernel, padding=padding)
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    return result_tensor

def apply_dominance_kernel(tensor, kernel_size, pos_factor=None, neg_factor=None):
    # python pt_shape.py evk5/frames_720_1280.pt downsampled in 0.1 per row and column
    # Shape of the tensor: torch.Size([3991, 72, 128])

    kernel_volume = kernel_size ** 3
    pos_mask = (tensor > 0).float()
    neg_mask = (tensor < 0).float()

    sum_counts = tensor[0].numel()
    max_neg = (tensor > 0).sum(dim=1).sum(dim=1).max()
    max_pos = (tensor < 0).sum(dim=1).sum(dim=1).max()

    if neg_factor is None:
        neg_factor = sum_counts / max_neg 
        # neg_factor = 1
    if pos_factor is None:
        pos_factor = sum_counts / max_pos
        # pos_factor = 1


    # weighted_tensor = torch.sign(tensor) * (pos_mask * pos_factor + neg_mask * neg_factor)
    weighted_tensor = tensor * (pos_mask * pos_factor + neg_mask * neg_factor)
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=tensor.device)/kernel_volume
    padding = kernel_size // 2
    weighted_tensor = weighted_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    result_tensor = F.conv3d(weighted_tensor, kernel, padding=padding)
    result_tensor = result_tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    return result_tensor#, pos_factor, neg_factor

def accumulate_frames(tensor, n_acc):
    # Accumulate every 'n_acc' frames
    n_frames = tensor.size(0) // n_acc
    accumulated_tensor = tensor[:n_frames * n_acc].view(n_frames, n_acc, *tensor.shape[1:]).sum(dim=1)
    return accumulated_tensor

def subtract_background(tensor):
    median_values = tensor.median(dim=2)[0].median(dim=1)[0]
    # median_values = tensor.median(dim=1, keepdim=True)[0]
    print(median_values.shape)
    pprint(median_values[::100])
    centralized_tensor = tensor - median_values[:, None, None]
    # centralized_tensor = tensor - median_values
    return centralized_tensor, median_values

def visualize_tensor(tensor_path, title, output_dir, file_suffix):
    visualizer = BlockVisualizer(tensor_path)
    # Views to visualize
    views = ["default", "vertical", "horizontal", "side", "r-side", "normal", "normal45", "lateral", "reverse"]
    for view in views:
        visualizer.plot_scatter_tensor(view=view, save=True, save_path=os.path.join(output_dir, f"{file_suffix}_{view}.png"), time_stretch=5, alpha=0.01)
        print(f"{title} saved at: {os.path.join(output_dir, f'{file_suffix}.png')}")

def plot_medians(tensor, output_dir, description=""):
    medians = tensor.median(dim=1)[0].median(dim=1)[0]

    plt.figure(figsize=(10, 5))
    plt.plot(medians, label='Median Values')
    plt.xlabel('Frame Index')
    plt.ylabel('Median Value')
    plt.title('Median Values Across Frames')
    plt.legend()
    save_path = os.path.join(output_dir, f'medians_plot_{description}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Medians plot saved at: {save_path}")

def compute_frame_statistics(tensor, output_dir, filename='frame_statistics.csv'):
    stats = {
        'Min': tensor.min(dim=1)[0].min(dim=1)[0],
        'Median': tensor.median(dim=1)[0].median(dim=1)[0],
        # 'Mean': tensor.mean(dim=1).mean(dim=1),
        'Max': tensor.max(dim=1)[0].max(dim=1)[0],
        
        'NegCount': ((tensor < 0)*torch.abs(tensor)).sum(dim=1).sum(dim=1),
        # 'NegCount': (tensor < 0).sum(dim=1).sum(dim=1),
        'ZeroCount': (tensor == 0).sum(dim=1).sum(dim=1),
        'PosCount': ((tensor > 0)*tensor).sum(dim=1).sum(dim=1),
        # 'PosCount': (tensor > 0).sum(dim=1).sum(dim=1),
    }
    df = pd.DataFrame(stats)
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index_label='Frame Index', sep=",")
    print(f"Frame statistics saved at: {csv_path}")

def plot_histograms(tensor, output_dir, hist_steps):
    histograms_dir = os.path.join(output_dir, "histograms")
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)
    
    print("Generating histograms for each frame...")
    for i in range(tensor.size(0)):
        plt.figure(figsize=(10, 6))
        plt.hist(tensor[i].flatten().numpy(), bins=hist_steps, color='blue', alpha=0.7)
        plt.title(f'Histogram of Frame {i}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(histograms_dir, f"histogram_frame_{i}.png"))
        plt.close()
    print(f"Histograms saved in directory: {histograms_dir}")

def apply_cumulative_sum(tensor):
    return tensor.cumsum(dim=0)

def process_tensor(tensor_path, max_shift, reverse, sample_rate, output_dir, hist_steps, n_acc, kernel_size):
    tensor = torch.load(tensor_path)
    width = tensor.shape[2]
    tensor_shifter = TensorShifter(max_shift, width // sample_rate, reverse)
    tensor = tensor[:, ::sample_rate, ::sample_rate]
    shifted_tensor = tensor_shifter.apply_shift(tensor)

    if n_acc > 1:
        shifted_tensor = accumulate_frames(shifted_tensor, n_acc)

    if kernel_size > 1:
        for _ in range(1):
            # smooth_shifted_tensor = shifted_tensor
            smooth_shifted_tensor = apply_dominance_kernel(shifted_tensor, kernel_size)
            # smooth_shifted_tensor = torch.sign(smooth_shifted_tensor)

    shifted_tensor_path = os.path.join(output_dir, f'shifted_tensor_{max_shift}_sample{sample_rate}.pt')
    torch.save(shifted_tensor, shifted_tensor_path)
    print(f"Shifted tensor saved at: {shifted_tensor_path}")

    centralized_tensor, _ = subtract_background(smooth_shifted_tensor)
    centralized_tensor_path = os.path.join(output_dir, f'centralized_tensor_{max_shift}_sample{sample_rate}.pt')
    torch.save(centralized_tensor, centralized_tensor_path)
    print(f"Centralized tensor saved at: {centralized_tensor_path}")

    cumulative_tensor = apply_cumulative_sum(centralized_tensor)
    cumulative_tensor_path = os.path.join(output_dir, f'cumulative_tensor_{max_shift}_sample{sample_rate}.pt')
    torch.save(cumulative_tensor, cumulative_tensor_path)
    print(f"Cumulative tensor saved at: {cumulative_tensor_path}")

    # visualize_tensor(centralized_tensor_path, 'Centralized Tensor Visualization', output_dir, 'centralized_tensor_visualization')
    # gc.collect()
    # visualize_tensor(shifted_tensor_path, 'Shifted Tensor Visualization', output_dir, 'shifted_tensor_visualization')
    # gc.collect()

    compute_frame_statistics(smooth_shifted_tensor, output_dir, "frame_statistics_smooth.csv")
    compute_frame_statistics(centralized_tensor, output_dir, "frame_statistics_centralized.csv")
    compute_frame_statistics(cumulative_tensor, output_dir, "frame_statistics_cumulative.csv")

    # compute_frame_statistics(smooth_shifted_tensor, output_dir, "frame_statistics.csv")
    compute_frame_statistics(centralized_tensor, output_dir, "frame_statistics.csv")

    plot_medians(smooth_shifted_tensor, output_dir, "smooth")
    plot_medians(centralized_tensor, output_dir, "centralized")
    plot_medians(cumulative_tensor, output_dir, "cumulative")

def main():
    parser = argparse.ArgumentParser(description='Process tensor by shifting, saving, subtracting background, visualizing, and generating histograms.')
    parser.add_argument('tensor_path', type=str, help='Path to the tensor file.')
    parser.add_argument('--shift', type=int, default=824, help='Shift value for the tensor transformation.')
    parser.add_argument('--reverse', action='store_true', help='Apply reverse shifting.')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sampling rate to reduce tensor resolution.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save processed tensors, figures, and statistics.')
    parser.add_argument('--hist_steps', type=int, default=100, help='Number of bins in histograms for each frame.')
    parser.add_argument('--n_acc', type=int, default=1, help='Number of frames to accumulate into one.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Dimension of the cubic kernel for the mean filter.')

    args = parser.parse_args()

    if args.output_dir is None:
        base_dir = os.path.dirname(args.tensor_path)
        base_name = os.path.splitext(os.path.basename(args.tensor_path))[0]
        args.output_dir = os.path.join(base_dir, base_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_tensor(args.tensor_path, args.shift, args.reverse, args.sample_rate, args.output_dir, args.hist_steps, args.n_acc, args.kernel_size)

if __name__ == "__main__":
    main()

    visualizer = SpectrumVisualizer('ciexyz31_1.txt')
    data_folder = 'evk5/frames_720_1280/'
    event_files = [
        ('shifted_tensor_824_sample1.pt', 1),
        ('centralized_tensor_824_sample1.pt', 1),
        ('cumulative_tensor_824_sample1.pt', 1)
    ]
    
    reconstructor = EventsSpectrumReconstructor(visualizer, data_folder, event_files)
    outputs = reconstructor.process_events()