import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

class BlockVisualizer:
    def __init__(self, tensor_path, sample_rate=1):
        print("Loading tensor...")
        self.tensor = torch.load(tensor_path)
        self.sample_rate = sample_rate
        print(f"Tensor loaded with shape: {self.tensor.shape}")

    def plot_scatter_tensor(self, view="default", polar=True, axis_ind="on", alpha=0.1, alpha_k=0.02, plot=False, save=False, save_path=None, time_stretch=10.0, zoom=1.0):
        """Plot a 3D scatter plot of the tensor data."""
        tensor_sampled = self.tensor[:, ::self.sample_rate, ::self.sample_rate]
        print(f"Sampled tensor shape: {tensor_sampled.shape}")

        # self.tensor = tensor_sampled

        positions = torch.nonzero(tensor_sampled, as_tuple=True)
        x, y, t = positions[2], positions[1], positions[0]  # Extract positions

        z = tensor_sampled[t, y, x].float().numpy()
        colors = ['blue' if val < 0 else 'red' for val in z]

        fig = plt.figure("Tensor Data Visualization", figsize=(40, 20))
        ax = fig.add_subplot(111, projection='3d')

        if polar:
            ax.scatter(x.numpy(), t.numpy() * time_stretch, y.numpy(), c=colors, alpha=alpha, marker=".")
        
        # ax.set_xlim3d(0, self.tensor.size(2) - 1)
        # ax.set_ylim3d(0, self.tensor.size(1) - 1)
        ax.set_xlim3d(0, tensor_sampled.size(2) - 1)
        ax.set_zlim3d(0, tensor_sampled.size(1) - 1)
        ax.set_ylim3d(0, max(t.numpy() * time_stretch))

        view_settings = {
            "vertical": (0, 90),
            "horizontal": (90, 0),
            "lateral": (90, 90),
            "side": (30, 18),
            "r-side": (-30, 18),
            "reverse": (-60, 18),
            "normal": (64, 18),
            "normal45": (45, 0),
            "default": (30, -30)
        }
        elev, azim = view_settings.get(view, (30, -30))
        ax.view_init(elev, azim)
        ax.set_box_aspect([self.tensor.size(2) / self.tensor.size(1), time_stretch, 1.0], zoom=zoom)
        plt.axis(axis_ind)

        if plot:
            plt.show()
        else:
            plt.close(fig)

        if save:
            fig.savefig(save_path, transparent=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize block data from a tensor.")
    parser.add_argument("tensor_path", type=str, help="Path to the tensor file (.pt) containing block data.")
    parser.add_argument("--view", type=str, default=None, help="View type for the 3D scatter plot.")
    parser.add_argument("--save", action="store_true", help="Save the plot as an image instead of displaying it.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the plot image.")
    parser.add_argument("--plot", action="store_true", help="Display the plot.")
    parser.add_argument("--time_stretch", type=float, default=10.0, help="Factor to stretch the time axis for visualization.")
    parser.add_argument("--sample_rate", type=int, default=10, help="Sampling rate for data to reduce point density.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Opacity of the points in the plot.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor for the plot.")

    args = parser.parse_args()

    if args.save_path is None:
        base_dir = os.path.dirname(args.tensor_path)
        base_name = os.path.splitext(os.path.basename(args.tensor_path))[0]
        

    visualizer = BlockVisualizer(args.tensor_path, sample_rate=args.sample_rate)

    # Views to visualize
    views = ["default", "vertical", "horizontal", "side", "r-side", "normal", "normal45", "lateral", "reverse"]

    # Loop through views and visualize only the specified view or all if none specified
    if args.view and args.view in views:
        save_path = os.path.join(base_dir, f"{base_name}_{args.view}.png")
        visualizer.plot_scatter_tensor(view=args.view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha, zoom=args.zoom)
    else:
        for view in views:
            save_path = os.path.join(base_dir, f"{base_name}_{view}.png")
            visualizer.plot_scatter_tensor(view=view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha, zoom=args.zoom)