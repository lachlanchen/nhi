import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read data from CSV file, parsing the timestamp column
# csv_path = "blank_recording_2024-10-16_20-43-30_segment_1_forward_start_ts_757000_max_duration_1031000.0_delta_t_500.csv"
# csv_path = "blank_recording_2024-10-16_20-43-30_segment_2_backward_start_ts_1788000_max_duration_1031000.0_delta_t_500.csv"
csv_path = "sanqin_recording_2024-10-16_20-35-04_segment_1_forward_start_ts_4617000.0_max_duration_1031000.0_delta_t_500.csv"
# csv_path = "sanqin_recording_2024-10-16_20-35-04_segment_2_backward_start_ts_5648000_max_duration_1031000.0_delta_t_500.csv"
df = pd.read_csv(csv_path)

# Parse the timestamp column
# Assuming the timestamp format is 'HH:MM:SS.sssssss'
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='%H:%M:%S.%f')

# Convert timestamps to microseconds starting from zero
df['timestamp_us'] = (df['event_timestamp'] - df['event_timestamp'].min()).dt.total_seconds() * 1e6

# Extract x, y, p, t
ts = df['timestamp_us'].values
xs = df['x'].values.astype(np.float32)
ys = df['y'].values.astype(np.float32)
ps = df['polarity'].values.astype(np.float32)

ps = (ps - 0.5) * 2

# ps = np.abs(ps)

# ps.min()

# Convert to tensors
xs = torch.tensor(xs, device=device)
ys = torch.tensor(ys, device=device)
ts = torch.tensor(ts, device=device)
ps = torch.tensor(ps, device=device)

# Set sensor size (height, width)
sensor_height = H = 720  # Replace with your sensor's height
sensor_width = W = 1280   # Replace with your sensor's width
sensor_size = (sensor_height, sensor_width)

bin_width = 1e5

class ScanCompensation(nn.Module):
    def __init__(self, initial_params):
        super().__init__()
        # Initialize the parameters a_x and a_y that will be optimized during training.
        self.params = nn.Parameter(initial_params)
    
    def warp(self, x_coords, y_coords, timestamps):
        """
        Adjust timestamps based on x and y positions scaled by sensor dimensions.
        """
        a_x = self.params[0]
        a_y = self.params[1]
        # t_warped = timestamps -  a_x * x_coords / W - a_y * y_coords / H
        # t_warped = timestamps - a_y * y_coords / H
        t_warped = timestamps -  a_x * x_coords - a_y * y_coords 
        # t_warped = timestamps -  a_y * y_coords 
        return x_coords, y_coords, t_warped
    
    def forward(self, x_coords, y_coords, timestamps, polarities):
        """
        Process events through the model by warping them and then computing the loss.
        """
        # t_start = ts.min()
        # t_end = ts.max()
        
        x_warped, y_warped, t_warped = self.warp(x_coords, y_coords, timestamps)
        
        # Define time binning parameters
        time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)  # 100ms in microseconds
        t_start = t_warped.min()
        t_end = t_warped.max()
        num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

        # Normalize time to [0, num_bins)
        t_norm = (t_warped - t_start) / time_bin_width

        # Compute floor and ceil indices for time bins
        t0 = torch.floor(t_norm)
        t1 = t0 + 1

        # Compute weights for linear interpolation over time
        wt = (t_norm - t0).float()  # Ensure float32

        # Clamping indices to valid range
        t0_clamped = t0.clamp(0, num_bins - 1)
        t1_clamped = t1.clamp(0, num_bins - 1)

        # Cast x and y to long for indexing
        x_indices = x_warped.long()
        y_indices = y_warped.long()

        # Ensure spatial indices are within bounds
        valid_mask = (x_indices >= 0) & (x_indices < W) & \
                     (y_indices >= 0) & (y_indices < H)

        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        t0_clamped = t0_clamped[valid_mask]
        t1_clamped = t1_clamped[valid_mask]
        wt = wt[valid_mask]
        polarities = polarities[valid_mask]

        # Compute linear indices for the event tensor
        spatial_indices = y_indices * W + x_indices
        spatial_indices = spatial_indices.long()

        # For t0
        flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
        flat_indices_t0 = flat_indices_t0.long()
        weights_t0 = ((1 - wt) * polarities).float()

        # For t1
        flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
        flat_indices_t1 = flat_indices_t1.long()
        weights_t1 = (wt * polarities).float()

        # Combine indices and weights
        flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
        flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

        # Create the flattened event tensor
        num_elements = num_bins * H * W
        event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

        # Accumulate events into the flattened tensor using scatter_add
        event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

        # Reshape back to (num_bins, H, W)
        event_tensor = event_tensor_flat.view(num_bins, H, W)

        # Compute the variance over x and y within each time bin
        # Variance over x and y for each time bin
        variances = torch.var(event_tensor.view(num_bins, -1), dim=1)
        # Loss is the negative mean variance
        loss = torch.mean(variances)

        return event_tensor, loss

# Initialize parameters a_x and a_y (start with small values)
initial_params = torch.zeros(2, device=device, requires_grad=True) # - 90*1000
model = ScanCompensation(initial_params)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1)

# Training loop
losses = []
num_iterations = 1000

for i in range(num_iterations):
    optimizer.zero_grad()
    event_tensor, loss = model(xs, ys, ts, ps)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if i%100 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}")
    
    # Adjust the learning rate if needed
    if i == 0.5 * num_iterations:
        optimizer.param_groups[0]['lr'] *= 0.5
    elif i == 0.8 * num_iterations:
        optimizer.param_groups[0]['lr'] *= 0.1

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Extract the final parameters
final_params = model.params.detach().cpu().numpy()
print(f"Final parameters (a_x, a_y): {final_params}")

# Optionally, visualize the event frames after warping
# Recompute the warped events
with torch.no_grad():
    x_warped, y_warped, t_warped = model.warp(xs, ys, ts)
    
    # Define time binning parameters
    time_bin_width = torch.tensor(bin_width, dtype=torch.float32, device=device)  # 100ms in microseconds
    t_start = t_warped.min()
    t_end = t_warped.max()
    num_bins = int(((t_end - t_start) / time_bin_width).item()) + 1

    # Normalize time to [0, num_bins)
    t_norm = (t_warped - t_start) / time_bin_width

    # Compute floor and ceil indices for time bins
    t0 = torch.floor(t_norm)
    t1 = t0 + 1

    # Compute weights for linear interpolation over time
    wt = (t_norm - t0).float()  # Ensure float32

    # Clamping indices to valid range
    t0_clamped = t0.clamp(0, num_bins - 1)
    t1_clamped = t1.clamp(0, num_bins - 1)

    # Cast x and y to long for indexing
    x_indices = x_warped.long()
    y_indices = y_warped.long()

    # Ensure spatial indices are within bounds
    valid_mask = (x_indices >= 0) & (x_indices < W) & \
                 (y_indices >= 0) & (y_indices < H)

    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    t0_clamped = t0_clamped[valid_mask]
    t1_clamped = t1_clamped[valid_mask]
    wt = wt[valid_mask]
    polarities = ps[valid_mask]

    # Compute linear indices for the event tensor
    spatial_indices = y_indices * W + x_indices
    spatial_indices = spatial_indices.long()

    # For t0
    flat_indices_t0 = t0_clamped * (H * W) + spatial_indices
    flat_indices_t0 = flat_indices_t0.long()
    weights_t0 = ((1 - wt) * polarities).float()

    # For t1
    flat_indices_t1 = t1_clamped * (H * W) + spatial_indices
    flat_indices_t1 = flat_indices_t1.long()
    weights_t1 = (wt * polarities).float()

    # Combine indices and weights
    flat_indices = torch.cat([flat_indices_t0, flat_indices_t1], dim=0)
    flat_weights = torch.cat([weights_t0, weights_t1], dim=0)

    # Create the flattened event tensor
    num_elements = num_bins * H * W
    event_tensor_flat = torch.zeros(num_elements, device=device, dtype=torch.float32)

    # Accumulate events into the flattened tensor using scatter_add
    event_tensor_flat = event_tensor_flat.scatter_add(0, flat_indices, flat_weights)

    # Reshape back to (num_bins, H, W)
    event_tensor = event_tensor_flat.view(num_bins, H, W)

    # Select a time bin to visualize
    time_bin_to_visualize = num_bins * 2 // 3  # For example, the middle time bin
    event_frame = event_tensor[time_bin_to_visualize].cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(event_frame, cmap='inferno')
    plt.title(f'Event Frame at Time Bin {time_bin_to_visualize}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Event Count')
    plt.show()