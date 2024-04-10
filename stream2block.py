import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


class EventDataTo3DBlock:
    def __init__(self, csv_path, use_cache=True):
        self.csv_path = csv_path
        self.cache_path = 'frames_640x480.pt'
        self.frames = None
        if use_cache and os.path.exists(self.cache_path):
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path)
        else:
            self.load_and_process_data()
            torch.save(self.frames, self.cache_path)
            print(f"Saved frames to {self.cache_path}.")
        
    def load_and_process_data(self):
        # Load data
        df = pd.read_csv(self.csv_path)

        # Map event_timestamps to unique indices
        unique_timestamps, indices = torch.unique(torch.tensor(df['event_timestamp'].astype('category').cat.codes), return_inverse=True)

        # Prepare tensor dimensions
        num_frames = len(unique_timestamps)
        height, width = 480, 640  # Assuming a 640x480 grid

        # Create a 3D tensor filled with zeros
        self.frames = torch.zeros((num_frames, height, width), dtype=torch.int)

        # Convert x, y, and polarity to tensors
        x = torch.tensor(df['x'].values, dtype=torch.long)
        y = torch.tensor(df['y'].values, dtype=torch.long)
        polarity = torch.tensor(df['polarity'].map({True: 1, False: -1}).values, dtype=torch.int)

        # Advanced indexing to fill the frames tensor
        self.frames[indices, y, x] = polarity

    def save_frames(self, filename):
        torch.save(self.frames, filename)
        print(f"Saved frames to {filename}.")

    # def show_middle_frame(self):
    #     middle_frame_index = len(self.frames) // 2
    #     plt.imshow(self.frames[middle_frame_index].numpy(), cmap='gray')
    #     plt.title(f"Frame at Index: {middle_frame_index}")
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # plt.show()
    #     plt.savefig("middle_frame.png")

    def show_middle_frame(self):
        middle_frame_index = len(self.frames) // 2
        frame = self.frames[middle_frame_index].numpy()

        # Extract coordinates for -1 and +1 values
        y_pos, x_pos = (frame == 1).nonzero()
        y_neg, x_neg = (frame == -1).nonzero()

        plt.figure(figsize=(10, 7.5))  # Adjust the figure size as needed
        plt.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
        plt.scatter(x_pos, y_pos, color='green', s=1, label='Polarity: True (+1)')
        plt.scatter(x_neg, y_neg, color='red', s=1, label='Polarity: False (-1)')

        plt.title(f"Frame at Index: {middle_frame_index}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(markerscale=10)  # Adjust the legend's marker scale
        plt.savefig("middle_frame.png")
        # plt.show()

    def convert_to_video(self, output_video='frames_video.mp4', fps=10):
        fig, ax = plt.subplots()

        def update(i):
            ax.clear()
            ax.imshow(self.frames[i].numpy(), cmap='gray')
            return ax,

        ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True)
        ani.save(output_video, fps=fps, extra_args=['-vcodec', 'libx264'])

        print(f"Saved video to {output_video}")


# Example usage
csv_path = 'data/filtered_event_data_with_positions.csv'
data_processor = EventDataTo3DBlock(csv_path)
# data_processor.save_frames('frames_640x480.pt')
# data_processor.show_middle_frame()
data_processor.convert_to_video()

