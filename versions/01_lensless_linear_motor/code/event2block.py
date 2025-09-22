import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import subprocess
from tqdm import tqdm

# Set up the device to use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EventDataTo3DBlock:
    def __init__(self, data_folder, use_cache=True):
        self.data_folder = data_folder  # Changed from csv_path to data_folder for directory management
        self.csv_path = os.path.join(data_folder, 'filtered_event_data_with_positions.csv')
        self.data_folder = os.path.join(self.data_folder, "intermediate")
        os.makedirs(self.data_folder, exist_ok=True)
        self.cache_path = os.path.join(self.data_folder, 'frames_640x480.pt')
        self.frames = None
        if use_cache and os.path.exists(self.cache_path):
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path).to(device)
        else:
            self.load_and_process_data()
            torch.save(self.frames.cpu(), self.cache_path)  # Save to CPU for compatibility
            print(f"Saved frames to {self.cache_path}.")

        self.slice_index = 0
        self.frames = self.frames[self.slice_index:]

    def save_unique_timestamps_and_positions(self):
        df = pd.read_csv(self.csv_path)
        unique_timestamps_positions = df[['event_timestamp', 'axis_y_position_mm']].drop_duplicates()
        output_path = os.path.join(self.data_folder, 'unique_timestamps_and_y_positions.csv')  # Ensure path is within data_folder
        unique_timestamps_positions[self.slice_index:].to_csv(output_path, index=False)
        print("Saved unique timestamps and Y positions to '" + output_path + "'.")

    def load_and_process_data(self):
        df = pd.read_csv(self.csv_path)
        unique_timestamps, indices = torch.unique(torch.tensor(df['event_timestamp'].astype('category').cat.codes), return_inverse=True)
        indices = indices.to(device)

        num_frames = len(unique_timestamps)
        height, width = 480, 640

        self.frames = torch.zeros((num_frames, height, width), dtype=torch.int, device=device)

        x = torch.tensor(df['x'].values, dtype=torch.long, device=device)
        y = torch.tensor(df['y'].values, dtype=torch.long, device=device)
        polarity = torch.tensor(df['polarity'].map({True: 1, False: -1}).values, dtype=torch.int, device=device)

        self.frames[indices, y, x] = polarity

    def save_frames(self, filename='frames_640x480.pt'):
        output_filename = os.path.join(self.data_folder, filename)
        torch.save(self.frames.cpu(), output_filename)
        print(f"Saved frames to {output_filename}.")

    def rotate_and_flip_frames(self):
        rotated_frames = torch.rot90(self.frames, k=1, dims=[-2, -1])
        flipped_frames = torch.flip(rotated_frames, dims=[-1])
        return flipped_frames

    def save_rotated_frames(self, filename='frames_640x480.pt'):
        output_filename = os.path.join(self.data_folder, f"flipped_rotated_{filename}")
        flipped_frames = self.rotate_and_flip_frames()
        torch.save(flipped_frames.cpu(), output_filename)
        print(f"Saved flipped and rotated frames to {output_filename}.")

    def show_middle_frame(self):
        middle_frame_index = len(self.frames) // 2
        frame = self.frames[middle_frame_index].cpu().numpy()
        plt.figure(figsize=(10, 7.5))
        plt.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
        y_pos, x_pos = (frame == 1).nonzero()
        y_neg, x_neg = (frame == -1).nonzero()
        plt.scatter(x_pos, y_pos, color='green', s=1, label='Polarity: True (+1)')
        plt.scatter(x_neg, y_neg, color='red', s=1, label='Polarity: False (-1)')
        plt.title(f"Frame at Index: {middle_frame_index}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(markerscale=10)
        plt.savefig(os.path.join(self.data_folder, "middle_frame.png"))  # Save within data_folder

    def convert_to_video(self, output_video='frames_video.mp4', fps=10):

        def update(i):
            ax.clear()
            frame = self.frames[i].cpu().numpy()  # Move to CPU for visualization
            ax.imshow(frame, cmap='gray', vmin=-1, vmax=1)
            ax.set_xlim(0, 640)
            ax.set_ylim(480, 0)
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=10, label='True (+1)')
            ax.scatter(x_neg, y_neg, color='red', s=10, label='False (-1)')
            if i == 0:
                ax.legend(loc='upper right')
            return ax,
            # 
        fig, ax = plt.subplots(figsize=(12, 9))
        ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(os.path.join(self.data_folder, output_video), writer=writer)
        print(f"Saved video to {os.path.join(self.data_folder, output_video)}")

        

    def save_all_frames_as_images(self, folder='frames'):
        folder_path = os.path.join(self.data_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in tqdm(range(len(self.frames))):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            frame = self.frames[i].cpu().numpy()
            ax.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=1)
            ax.scatter(x_neg, y_neg, color='red', s=1)
            plt.savefig(os.path.join(folder_path, f"frame_{i:04d}.png"), dpi=100)
            plt.close()
        print(f"Saved all frames as images in the {folder_path}/ directory.")

    def frames_to_video(self, folder='frames', output_video='frames_video_with_ffmpeg.mp4', fps=60):
        folder_path = os.path.join(self.data_folder, folder)
        output_path = os.path.join(self.data_folder, output_video)
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder_path, 'frame_%04d.png'),
            '-vf', 'scale=800:600',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_path}")

        # After saving the normal video, create a rotated and flipped version
        self.rotate_and_flip_video(output_path, 'rotated_flipped_' + output_video)

    def rotate_and_flip_video(self, input_video, output_video):
        """
        Rotate and flip an existing video file and save the transformed video.
        """
        output_path = os.path.join(self.data_folder, output_video)
        command = [
            'ffmpeg', '-y',  # Overwrite output files without asking
            '-i', input_video,  # Input video file
            '-vf', "transpose=1,hflip, scale=800:600",  # Rotate 90 degrees counterclockwise, horizontal flip, and scale
            '-c:v', 'libx264',  # Video codec
            '-pix_fmt', 'yuv420p',  # Pixel format
            output_path  # Output video file
        ]
        subprocess.run(command, check=True)
        print(f"Saved rotated and flipped video to {output_path}")

if __name__ == '__main__':
    data_processor = EventDataTo3DBlock('data-final/data-without-sample-s50-r1')
    data_processor.save_frames('frames_640x480.pt')
    data_processor.save_rotated_frames('frames_640x480.pt')
    data_processor.show_middle_frame()
    data_processor.convert_to_video()
    data_processor.save_all_frames_as_images("frames")
    data_processor.frames_to_video("frames", 'frames_video_with_ffmpeg.mp4')
    data_processor.save_unique_timestamps_and_positions()
