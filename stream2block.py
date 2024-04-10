import pandas as pd
import torch

# Load data
df = pd.read_csv('data/filtered_event_data_with_positions.csv')

# Map event_timestamps to unique indices
unique_timestamps, indices = torch.unique(torch.tensor(df['event_timestamp'].astype('category').cat.codes), return_inverse=True)

# Prepare tensor dimensions
num_frames = len(unique_timestamps)
height, width = 480, 640  # Assuming a 640x480 grid

# Create a 3D tensor filled with zeros
frames = torch.zeros((num_frames, height, width), dtype=torch.int)

# Convert x, y, and polarity to tensors
x = torch.tensor(df['x'].values)
y = torch.tensor(df['y'].values)
polarity = torch.tensor(df['polarity'].map({True: 1, False: -1}).values)

# Advanced indexing to fill the frames tensor
frames[indices, y, x] = polarity

# Save the frames tensor
torch.save(frames, 'frames_640x480.pt')

# Print shape and show the middle frame using matplotlib
print("Shape of the 3D tensor (frames):", frames.shape)
middle_frame_index = num_frames // 2
print("Middle frame index:", middle_frame_index)

# Visualizing the middle frame
import matplotlib.pyplot as plt

plt.imshow(frames[middle_frame_index].numpy(), cmap='gray')
plt.title(f"Frame at Index: {middle_frame_index}")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
