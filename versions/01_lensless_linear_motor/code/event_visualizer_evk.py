import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import argparse

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
        """Process the data to convert timestamps and polarities."""
        self.data['event_timestamp'] = pd.to_datetime(self.data['event_timestamp'], format='%H:%M:%S.%f')
        self.data['relative_time_us'] = (self.data['event_timestamp'] - self.data['event_timestamp'].iloc[0]).dt.total_seconds() * 1e6
        self.data['polarity'] = self.data['polarity'].astype(int)
        
        self.xs = self.data['x'].values
        self.ys = self.data['y'].values
        self.ts = self.data['relative_time_us'].values
        self.ps = self.data['polarity'].values

    def plot_scatter_bos_rect(self, 
        view="default", polar=True, axis_ind="on", 
        alpha=0.1, alpha_k=0.02, plot=False, save=False, save_path=None, time_stretch=10.0, zoom=1.0):
        """Plot a 3D scatter plot of the event data."""
        pos_loc = np.where(self.ps == 1)
        neg_loc = np.where(self.ps == 0)
        pos_xs, pos_ys, pos_ts = self.xs[pos_loc], self.ys[pos_loc], self.ts[pos_loc]
        neg_xs, neg_ys, neg_ts = self.xs[neg_loc], self.ys[neg_loc], self.ts[neg_loc]

        fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(40, 20))
        ax = fig.add_subplot(111, projection='3d')

        if polar:
            # Plot positive polarity events
            # ax.scatter(pos_xs, pos_ys, time_stretch * pos_ts / 1e6, c='r', alpha=alpha, marker=".")
            # Plot negative polarity events
            # ax.scatter(neg_xs, neg_ys, time_stretch * neg_ts / 1e6, c='b', alpha=alpha, marker=".")

            # Plot positive polarity events with swapped y and z axes
            ax.scatter(pos_xs, time_stretch * pos_ts / 1e6, pos_ys, c='r', alpha=alpha, marker=".")
            # Plot negative polarity events with swapped y and z axes
            ax.scatter(neg_xs, time_stretch * neg_ts / 1e6, neg_ys, c='b', alpha=alpha, marker=".")
        
        # Uncomment this to plot all events with x, y, z positions set to (xs, 0, ys)
        # ax.scatter(self.xs, np.zeros_like(self.xs), self.ys, c='k', alpha=alpha_k, marker=".")

        ax.set_xlim3d(0, self.sensor_size[1] - 1)
        ax.set_zlim3d(0, self.sensor_size[0] - 1)
        ax.set_ylim3d(self.ts.min() / 1e6, (self.ts.max() / 1e6) * time_stretch)
        
        # Scaling the time axis
        x_scale = 7
        z_scale = 7
        y_scale = time_stretch

        scale = np.diag([x_scale, y_scale, z_scale, 1.0])
        scale = scale * (1.0 / scale.max())
        scale[3, 3] = 1.0

        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)

        ax.get_proj = short_proj

        ax.set_box_aspect([self.sensor_size[1] / self.sensor_size[0], 2, 1.0], zoom=zoom)

        view_settings = {
            "vertical": (0, 90),
            "horizontal": (90, 0),
            "lateral": (90, 90),
            "side": (30, 18),
            "r-side": (-30, 18),
            "reverse": (-60, 18),
            "normal": (64, 18),
            "normal45": (45, 0),
            # "default": (135, 60)  # Rotated 90 degrees left from the previous default
            "default": (30, -30)  # Rotated 90 degrees left from the previous default
        }
        elev, azim = view_settings.get(view, (30, -30))

        ax.view_init(elev, azim)
        
        # Apply zoom effect by adjusting the camera position
        # ax.dist = 10 / zoom  # Default distance is 10, adjust for zoom

        plt.axis(axis_ind)

        # Adjust ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        if plot:
            plt.show()
        else:
            plt.close(fig)

        if save:
            fig.savefig(save_path, transparent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize event data from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing event data.")
    parser.add_argument("-m", "--model", type=str, choices=["D", "X", "E"], help="Sensor model selection: D for Davis 346x260, X for DVXplorer Mini 640x480, E for EVK5 1280x720.")
    parser.add_argument("--sensor_size", type=int, nargs=2, help="Directly specify sensor size as width and height.")
    parser.add_argument("--view", type=str, help="View type for the 3D scatter plot.")
    parser.add_argument("--save", action="store_true", help="Save the plot as an image instead of displaying it.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the plot image.")
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
        if args.save_path is None:
            save_path = os.path.join(base_dir, f"{base_name}_{args.view}.jpeg")
        else:
            save_path = args.save_path
        visualizer.plot_scatter_bos_rect(view=args.view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha, zoom=args.zoom)
    else:
        views = ["default", "vertical", "horizontal", "side", "r-side", "normal", "normal45", "lateral", "reverse"]
        for view in views:
            if args.save_path is None:
                save_path = os.path.join(base_dir, f"{base_name}_{view}.jpeg")
            else:
                save_path = os.path.join(os.path.dirname(args.save_path), f"{base_name}_{view}.jpeg")
            visualizer.plot_scatter_bos_rect(view=view, plot=args.plot, save=args.save, save_path=save_path, time_stretch=args.time_stretch, alpha=args.alpha, zoom=args.zoom)
