import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_tensor_slices(tensor_path, step_size, output_dir):
    # Load the tensor
    tensor = torch.load(tensor_path)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define custom colormap: red for positive values, blue for negative values
    colors = ['blue', 'white', 'red']
    cmap = ListedColormap(colors)

    # Normalize: positive values will be mapped to red, negative values to blue
    bounds = [-1, 0, 1]
    norm = plt.Normalize(vmin=-0.3, vmax=0.3)
    
    # Iterate over the tensor and save slices at the given step size
    for i in range(0, tensor.size(0), step_size):
        slice_ = tensor[i].cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_, cmap=cmap, norm=norm, aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title(f'Frame {i}')
        plt.xlabel('Width')
        plt.ylabel('Height')
        save_path = os.path.join(output_dir, f'frame_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved frame {i} at {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize tensor slices at a specified step size.')
    parser.add_argument('tensor_path', type=str, help='Path to the tensor file.')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for visualizing frames.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the visualized frames.')
    args = parser.parse_args()
    
    if args.output_dir is None:
        base_dir = os.path.dirname(args.tensor_path)
        base_name = os.path.splitext(os.path.basename(args.tensor_path))[0]
        args.output_dir = os.path.join(base_dir, base_name + '_slices')

    visualize_tensor_slices(args.tensor_path, args.step_size, args.output_dir)

if __name__ == "__main__":
    main()
