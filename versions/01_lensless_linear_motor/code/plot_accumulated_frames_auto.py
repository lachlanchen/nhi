import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse

def load_tensor(file_path):
    return torch.load(file_path)

def normalize_tensor(tensor):
    # Normalize tensor to range [0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def plot_frames(tensor, output_folder, tensor_name, normalize=False):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Normalize the tensor if required
    if normalize:
        tensor = normalize_tensor(tensor)

    # Use range to iterate over all frames
    indices = range(tensor.size(0))
    titles = [f"Frame {i}" for i in indices]

    norm_status = "normalized" if normalize else "raw"
    for idx, title in zip(indices, titles):
        plt.figure(figsize=(10, 8))
        plt.imshow(tensor[idx].cpu(), cmap='gray')
        plt.colorbar()
        plt.title(f"{title} Frame of {tensor_name} ({norm_status})")
        plt.axis('off')

        # Save the figure
        frame_file_name = f"{tensor_name}_{title.lower()}.png"
        plt.savefig(os.path.join(output_folder, frame_file_name))
        plt.close()

def main(args):
    # Determine output folder base from input paths
    base_name_continuous = os.path.splitext(os.path.basename(args.continuous_path))[0]
    base_name_interval = os.path.splitext(os.path.basename(args.interval_path))[0]

    output_folder_base = os.path.dirname(args.continuous_path)
    continuous_folder = os.path.join(output_folder_base, base_name_continuous, 'continuous_' + ('normalized' if args.normalize else 'raw'))
    interval_folder = os.path.join(output_folder_base, base_name_interval, 'interval_' + ('normalized' if args.normalize else 'raw'))

    # Load tensors
    continuous_tensor = load_tensor(args.continuous_path)
    interval_tensor = load_tensor(args.interval_path)

    # Plot and save frames for both tensors
    plot_frames(continuous_tensor, continuous_folder, 'Continuous', normalize=args.normalize)
    plot_frames(interval_tensor, interval_folder, 'Interval', normalize=args.normalize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot and Save Frames from Accumulated Tensors')
    parser.add_argument('continuous_path', type=str, help='Path to the continuous accumulation tensor file')
    parser.add_argument('interval_path', type=str, help='Path to the interval accumulation tensor file')
    parser.add_argument('--normalize', action='store_true', help='Normalize the tensor data before plotting')

    args = parser.parse_args()
    main(args)
