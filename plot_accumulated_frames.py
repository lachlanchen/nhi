import torch
import matplotlib.pyplot as plt
import os
import sys

def load_tensor(file_path):
    return torch.load(file_path)

def normalize_tensor(tensor):
    # Normalize tensor to range [0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def plot_frames(tensor, output_folder, tensor_name):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Normalize the tensor
    tensor = normalize_tensor(tensor)

    # Use range to iterate over all frames
    indices = range(tensor.size(0))
    titles = [f"Frame {i}" for i in indices]

    for idx, title in zip(indices, titles):
        plt.figure(figsize=(10, 8))
        plt.imshow(tensor[idx].cpu(), cmap='gray')
        plt.colorbar()
        plt.title(f"{title} Frame of {tensor_name}")
        plt.axis('off')

        # Save the figure
        plt.savefig(os.path.join(output_folder, f"{tensor_name}_{title.lower()}.png"))
        plt.close()

def main(continuous_path, interval_path, output_folder):
    # Load tensors
    continuous_tensor = load_tensor(continuous_path)
    interval_tensor = load_tensor(interval_path)

    # Plot and save frames for both tensors
    plot_frames(continuous_tensor, output_folder, 'Continuous')
    plot_frames(interval_tensor, output_folder, 'Interval')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_accumulated_frames.py <continuous_tensor_path> <interval_tensor_path> <output_folder>")
        sys.exit(1)

    continuous_path = sys.argv[1]
    interval_path = sys.argv[2]
    output_folder = sys.argv[3]

    main(continuous_path, interval_path, output_folder)
