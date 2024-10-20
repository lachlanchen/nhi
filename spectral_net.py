import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def debug_print(message):
    print(f"[DEBUG] {message}")

debug_print("Starting script...")

# Load the data
debug_print("Loading data...")
data_path = '/home/lachlan/ProjectsLFS/nhi_reconstruction/david-from15-to50-learning-filtered/filtered_event_data_with_positions.csv'
data_folder = os.path.dirname(data_path)
data = pd.read_csv(data_path)

# Limit the number of events for initial testing
# num_events = 10000
num_events = len(data)
data = data.head(num_events)
debug_print(f"Loaded {len(data)} events.")

# Convert timestamps to seconds since the first event
debug_print("Converting timestamps...")
data['event_timestamp'] = pd.to_datetime(data['event_timestamp'], format='%H:%M:%S.%f')
data['event_timestamp'] = (data['event_timestamp'] - data['event_timestamp'].min()).dt.total_seconds()

# Rename columns for convenience
debug_print("Renaming columns...")

data["axis_y_position_m"] = data["axis_y_position_mm"] * 1e-3
data.rename(columns={'axis_y_position_m': 'xi'}, inplace=True)

# Constants
d = 1 / 600000  # Grating spacing in meters (600 lines/mm)
z1 = 0.1  # z1 in meters
z2 = 0.1  # z2 in meters
pixel_size = 4.8e-6  # Pixel size in meters (4.8 um)
sensor_width = 1280 * pixel_size
sensor_height = 720 * pixel_size

# Normalize x and y to 0-1 range for neural network input
debug_print("Normalizing x and y...")
data['x_norm'] = data['x'] / 1280
data['y_norm'] = data['y'] / 720

# Convert x to real value for lambda calculation
debug_print("Calculating real x values...")
data['x_real'] = data['x_norm'] * sensor_width - (sensor_width / 2)
data['y_real'] = data['y_norm'] * sensor_height - (sensor_height / 2)

# Calculate lambda for each event
debug_print("Calculating lambda values...")
data['lambda'] = d * (data['xi'] / ((data['xi']**2 + z1**2)**0.5) + data['x_real'] / ((data['x_real']**2 + z2**2)**0.5))
data["polarity"] = (data["polarity"] - 0.5)*2

print(data["polarity"].unique())
print(data["lambda"].unique())

print("Data after preprocessing:")
print(data.head())

# Hardcode the speed (d(xi)/dt) as 10 mm/s
debug_print("Hardcoding speed...")
data['d_xi_dt'] = 10.0 * 1e-3  # 10 mm/s in meters/s

# Calculate the additional factor
data['factor'] = d * (z1**2 / ((data['xi']**2 + z1**2)**(3/2))) * data['d_xi_dt']

# Polarity is the target output (event polarity)
data['target'] = data['polarity']

# Prepare the data for PyTorch
debug_print("Preparing data for PyTorch...")
inputs = torch.tensor(data[['x_real', 'y_real', 'lambda']].values, dtype=torch.float32).cuda() * 1e3
targets = torch.tensor(data['polarity'].values, dtype=torch.float32).unsqueeze(1).cuda()
factors = torch.tensor(data['factor'].values, dtype=torch.float32).unsqueeze(1).cuda()

# Create DataLoader
dataset = TensorDataset(inputs, targets, factors)
n_kilo = 256
dataloader = DataLoader(dataset, batch_size=1024*n_kilo, shuffle=True)
debug_print("DataLoader created.")

# Define the neural network
class SpectralNet(nn.Module):
    def __init__(self):
        super(SpectralNet, self).__init__()
        self.fc1 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        return x

debug_print("Neural network defined.")

# Instantiate the model, define the loss function and the optimizer
model = SpectralNet().cuda()
criterion = nn.MSELoss()  # Changed to MSELoss for continuous output
optimizer = optim.Adam(model.parameters(), lr=0.001)
debug_print("Model, loss function, and optimizer instantiated.")

# Prepare the CSV file for logging
log_file = os.path.join(data_folder, 'training_log.csv')
with open(log_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss'])

# Training loop
num_epochs = 100  # Train for more epochs since we have a powerful GPU and large dataset
losses = []
debug_print("Starting training loop...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (inputs, targets, factors) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        # scaled_outputs = torch.tanh(outputs * factors)  # Scale by the factor
        scaled_outputs = outputs 


        loss = criterion(scaled_outputs, targets)

        # # Regularization term
        # regularizer = torch.mean(torch.abs(scaled_outputs))
        # loss += regularizer

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the details
        with open(log_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, batch_idx, loss.item()])

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.9f}")
    
    # Save losses for plotting
    losses.append(loss.item())

    # Save RGB wavelength images every 10 epochs
    if (epoch + 1) % 5 == 0:
        debug_print(f"Saving RGB images for epoch {epoch + 1}...")
        # Example code for saving RGB images
        output_folder = os.path.join(data_folder, 'rgb_images', f'epoch_{epoch + 1}')
        os.makedirs(output_folder, exist_ok=True)
        
        x_grid, y_grid = np.meshgrid(np.linspace(-0.5, 0.5, 1280), np.linspace(-0.5, 0.5, 720))
        x_grid_flat = x_grid.flatten() * sensor_width * 1e3
        y_grid_flat = y_grid.flatten() * sensor_height * 1e3

        for wavelength, color in zip([700e-9, 546.1e-9, 435.8e-9], ['red', 'green', 'blue']):
            lambda_grid = np.full_like(x_grid_flat, wavelength*1e3)
            mesh_inputs = torch.tensor(np.vstack((x_grid_flat, y_grid_flat, lambda_grid)).T, dtype=torch.float32).cuda()
            
            mesh_outputs = []
            batch_size = 1024 * n_kilo
            for i in range(0, mesh_inputs.size(0), batch_size):
                batch_inputs = mesh_inputs[i:i+batch_size]
                with torch.no_grad():
                    batch_outputs = model(batch_inputs)
                mesh_outputs.append(batch_outputs)
            
            mesh_outputs = torch.cat(mesh_outputs).reshape(720, 1280).cpu().numpy()
            mesh_outputs = (mesh_outputs+1)/2
            plt.imsave(os.path.join(output_folder, f'{color}.png'), mesh_outputs, cmap='gray')

# Save the model
torch.save(model.state_dict(), os.path.join(data_folder, 'spectral_net.pth'))
debug_print("Model saved.")

# Save the loss curve
plt.figure()
plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig(os.path.join(data_folder, 'loss_curve.png'))
plt.close()
debug_print("Loss curve saved.")

debug_print("Script finished.")
