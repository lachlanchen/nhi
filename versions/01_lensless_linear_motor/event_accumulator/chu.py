import numpy as np
import torch
import pandas as pd
import argparse

class EventImageCreator:
    def __init__(self, csv_path, sensor_size=(180, 240), interpolation='bilinear', padding=False):
        self.csv_path = csv_path
        self.sensor_size = sensor_size
        self.interpolation = interpolation
        self.padding = padding
        self.data = self.load_data()

    def load_data(self):
        """Load event data from a CSV file and ensure correct types."""
        data = pd.read_csv(self.csv_path)
        print(f"Data loaded from {self.csv_path}")

        # Ensure that x, y, and polarity are of numeric type and handle any NaN values
        data['x'] = pd.to_numeric(data['x'], errors='coerce').fillna(0).astype(int)
        data['y'] = pd.to_numeric(data['y'], errors='coerce').fillna(0).astype(int)
        data['polarity'] = data['polarity'].map({True: 1, False: -1}).astype(int)  # Convert polarity to 1 or -1

        # Drop any rows that still have NaN values if any exist
        data.dropna(subset=['x', 'y', 'polarity'], inplace=True)

        return data[['x', 'y', 'polarity']].to_numpy()

    # def interpolate_to_image(self, pxs, pys, dxs, dys, weights, img):
    #     """Accumulate coordinates to an image using bilinear interpolation."""
    #     img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    #     img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    #     img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    #     img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    #     return img

    def interpolate_to_image(self, pxs, pys, dxs, dys, weights, img):
        """Accumulate coordinates to an image using bilinear interpolation with edge case handling."""
        # Ensuring that the indices do not go out of bounds
        max_y, max_x = img.shape[0] - 1, img.shape[1] - 1
        pxs_clipped = torch.clamp(pxs, 0, max_x - 1)
        pys_clipped = torch.clamp(pys, 0, max_y - 1)

        # Weights calculation with clamping to prevent out-of-bounds access
        img.index_put_((pys_clipped,   pxs_clipped), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
        img.index_put_((pys_clipped,   pxs_clipped + 1), weights * dxs * (1.0 - dys), accumulate=True)
        img.index_put_((pys_clipped + 1, pxs_clipped), weights * (1.0 - dxs) * dys, accumulate=True)
        img.index_put_((pys_clipped + 1, pxs_clipped + 1), weights * dxs * dys, accumulate=True)
        return img


    def events_to_image(self):
        """Convert event data to an image using specified interpolation method."""
        xs, ys, ps = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        if self.interpolation == 'bilinear':
            xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
            xt, yt, pt = xt.float(), yt.float(), pt.float()
            img = self.events_to_image_torch(xt, yt, pt)
        else:
            coords = np.stack((ys, xs))
            abs_coords = np.ravel_multi_index(coords, self.sensor_size)
            img = np.bincount(abs_coords, weights=ps, minlength=np.prod(self.sensor_size)).reshape(self.sensor_size)
        return img

    def events_to_image_torch(self, xs, ys, ps):
        """Process event data to an image using PyTorch, with optional bilinear interpolation."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        xs, ys, ps = xs.to(device), ys.to(device), ps.to(device)

        img = torch.zeros(self.sensor_size).to(device)
        pxs, pys = xs.floor().long(), ys.floor().long()
        dxs, dys = (xs - pxs).float(), (ys - pys).float()

        img = self.interpolate_to_image(pxs, pys, dxs, dys, ps, img)
        return img.cpu().numpy()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert event data from CSV to image.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing event data.")
    parser.add_argument("-s", "--size", type=str, help="Sensor size as H,W", default=None)
    parser.add_argument("-m", "--model", choices=['D', 'X'], help="Predefined model sizes: D for 346x260, X for 640x480", default=None)
    args = parser.parse_args()

    # Determine sensor size
    if args.size:
        sensor_size = tuple(map(int, args.size.split(',')))
    elif args.model == 'D':
        sensor_size = (260, 346)
    elif args.model == 'X':
        sensor_size = (480, 640)
    else:
        # Default to a small size, but should be set by user or data
        # Load CSV to determine max x and y if not specified
        data = pd.read_csv(args.csv_path)
        sensor_size = (data['y'].max() + 1, data['x'].max() + 1)
    
    return args.csv_path, sensor_size

if __name__ == "__main__":
    csv_path, sensor_size = parse_arguments()
    creator = EventImageCreator(csv_path, sensor_size=sensor_size)
    image = creator.events_to_image()
    print("Image created with shape:", image.shape)
