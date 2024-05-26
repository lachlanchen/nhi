import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from custom_visualizer import Custom3DVisualizer  # Import the custom visualizer

class EventVisualizer:
    def __init__(self, csv_file, sensor_size=(260, 346), sample_rate=1):
        self.sensor_size = sensor_size
        self.sample_rate = sample_rate
        self.load_data(csv_file)
        self.process_data()

    def load_data(self, csv_file):
        """Load event data from a CSV file with the specified sample rate."""
        self.data = pd.read_csv(csv_file).iloc[::self.sample_rate]
        
    def process_data(self):
        """Convert timestamps and polarities."""
        self.data['event_timestamp'] = pd.to_datetime(self.data['event_timestamp'], format='%H:%M:%S.%f')
        self.data['relative_time_us'] = (self.data['event_timestamp'] - self.data['event_timestamp'].iloc[0]).dt.total_seconds() * 1e6
        self.data['polarity'] = self.data['polarity'].astype(int)
        
        self.xs = self.data['x'].values
        self.ys = self.data['y'].values
        self.ts = self.data['relative_time_us'].values
        self.ps = self.data['polarity'].values

    def plot_scatter_bos_rect(self, view="default", plot=False, save=False, save_path=None, time_stretch=10.0, alpha=0.1):
        """Use the Custom3DVisualizer for plotting."""
        colors = ['red' if val == 1 else 'blue' for val in self.ps]  # Color by polarity
        
        fig, ax = plt.subplots(figsize=(40, 20), subplot_kw={'projection': '3d'})
        visualizer = Custom3DVisualizer(self.xs, self.ts * time_stretch, self.ys, colors, alpha=alpha)
        visualizer.plot(ax)

        ax.set_xlim3d(0, self.sensor_size[1] - 1)
        ax.set_zlim3d(0, self.sensor_size[0] - 1)
        ax.set_ylim3d(0, max(self.ts) * time_stretch)

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

        if plot:
            plt.show()
        if save:
            plt.savefig(save_path, transparent=True)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize event data from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing event data.")
    parser.add_argument("-m", "--model", choices=["D", "X", "E"], help="Specify sensor model.")
    parser.add_argument("--sensor_size", nargs=2, type=int, help="Directly specify sensor size.")
    parser.add_argument("--view", type=str, help="View type for the 3D scatter plot.")
    parser.add_argument("--save", action="store_true", help="Save the plot as an image.")
    parser.add_argument("--save_path", type=str, help="Path to save the plot image.")
    parser.add_argument("--plot", action="store_true", help="Display the plot.")
    parser.add_argument("--time_stretch", type=float, default=10.0, help="Factor to stretch the time axis.")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sampling rate for data.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Opacity of the points in the plot.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor for the plot.")

    args = parser.parse_args()

    sensor_sizes = {
        "D": (260, 346),
        "X": (640, 480),
        "E": (720, 1280)
    }
    sensor_size = sensor_sizes.get(args.model, args.sensor_size) if args.model else args.sensor_size

    if args.save_path is None:
        base_dir = os.path.dirname(args.csv_file)
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]

    visualizer = EventVisualizer(args.csv_file, sensor_size=sensor_size or (260, 346), sample_rate=args.sample_rate)
    if args.view:
        save_path = os.path.join(base_dir, f"{base_name}_{args.view}.png") if args.save_path is None else args.save_path
        visualizer.plot_scatter_bos_rect(view=args.view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha)
    else:
        views = ["default", "vertical", "horizontal", "side", "r-side", "normal", "normal45", "lateral", "reverse"]
        for view in views:
            save_path = os.path.join(base_dir, "block_custom_vis", f"{base_name}_{view}.png") if args.save_path is None else os.path.join(os.path.dirname(args.save_path), f"{base_name}_{view}.png")
            visualizer.plot_scatter_bos_rect(view=view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha)
