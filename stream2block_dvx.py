import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import subprocess

# Set up the device to use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EventDataTo3DBlock:
    def __init__(self, csv_path, use_cache=True):
        self.csv_path = csv_path
        self.cache_path = 'frames_640x480.pt'
        # self.flipped_cache_path = 'flipped_rotated_frames_640x480.pt'
        self.frames = None
        # self.flipped_frames = None
        if use_cache and os.path.exists(self.cache_path):
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path).to(device)
        else:
            self.load_and_process_data()
            torch.save(self.frames.cpu(), self.cache_path)  # Save to CPU for compatibility
            print(f"Saved frames to {self.cache_path}.")


        self.slice_index = 0
        self.frames = self.frames[self.slice_index:]

        # self.flipped_frames = self.rotate_and_flip_frames()

        

    def save_unique_timestamps_and_positions(self):
        df = pd.read_csv(self.csv_path)
        unique_timestamps_positions = df[['event_timestamp', 'axis_y_position_mm']].drop_duplicates()
        unique_timestamps_positions[self.slice_index:].to_csv('unique_timestamps_and_y_positions.csv', index=False)
        print("Saved unique timestamps and Y positions to 'unique_timestamps_and_y_positions.csv'.")

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

    def save_frames(self, filename):
        torch.save(self.frames.cpu(), filename)  # Save to CPU
        print(f"Saved frames to {filename}.")

    # def save_rotated_frames(self, filename):
    #     rotated_frames = torch.rot90(self.frames, k=1, dims=[-2, -1])
    #     torch.save(rotated_frames.cpu(), f"rotated_{filename}")  # Save to CPU
    #     print(f"Saved frames to {filename}.")

    def rotate_and_flip_frames(self):
         # Rotate the frames by 90 degrees counter-clockwise
        rotated_frames = torch.rot90(self.frames, k=1, dims=[-2, -1])
        
        # Flip the rotated frames horizontally
        flipped_frames = torch.flip(rotated_frames, dims=[-1])

        return flipped_frames


    def save_rotated_frames(self, filename):
        # Rotate the frames by 90 degrees counter-clockwise
        rotated_frames = torch.rot90(self.frames, k=1, dims=[-2, -1])
        
        # Flip the rotated frames horizontally
        flipped_frames = torch.flip(rotated_frames, dims=[-1])
        
        # Save the flipped frames to disk, ensuring they are moved to CPU memory if necessary
        torch.save(flipped_frames.cpu(), f"flipped_rotated_{filename}")
        
        # Print the actual filename where the flipped and rotated frames are saved
        print(f"Saved flipped and rotated frames to flipped_rotated_{filename}.")


    def show_middle_frame(self):
        middle_frame_index = len(self.frames) // 2
        frame = self.frames[middle_frame_index].cpu().numpy()  # Move to CPU for visualization
        y_pos, x_pos = (frame == 1).nonzero()
        y_neg, x_neg = (frame == -1).nonzero()

        plt.figure(figsize=(10, 7.5))
        plt.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
        plt.scatter(x_pos, y_pos, color='green', s=1, label='Polarity: True (+1)')
        plt.scatter(x_neg, y_neg, color='red', s=1, label='Polarity: False (-1)')
        plt.title(f"Frame at Index: {middle_frame_index}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(markerscale=10)
        plt.savefig("middle_frame.png")
        # plt.show()

    def convert_to_video(self, output_video='frames_video.mp4', fps=10):
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)

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

        ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(output_video, writer=writer)
        print(f"Saved video to {output_video}")

    def save_all_frames_as_images(self, folder='frames'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in range(len(self.frames)):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            frame = self.frames[i].cpu().numpy()
            ax.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=1)
            ax.scatter(x_neg, y_neg, color='red', s=1)
            plt.savefig(f"{folder}/frame_{i:04d}.png", dpi=100)
            plt.close()

        print(f"Saved all frames as images in the {folder}/ directory.")

    def frames_to_video(self, folder='frames', output_video='frames_video_with_ffmpeg.mp4', fps=60):
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder, 'frame_%04d.png'),
            '-vf', 'scale=800:600',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_video}")

        # After saving the normal video, create a rotated and flipped version
        self.rotate_and_flip_video(output_video, 'rotated_flipped_' + output_video)


    def rotate_and_flip_video(self, input_video, output_video):
        """
        Rotate and flip an existing video file and save the transformed video.
        """
        command = [
            'ffmpeg', '-y',  # Overwrite output files without asking
            '-i', input_video,  # Input video file
            '-vf', "transpose=1,hflip, scale=800:600",  # Rotate 90 degrees counterclockwise, horizontal flip, and scale
            '-c:v', 'libx264',  # Video codec
            '-pix_fmt', 'yuv420p',  # Pixel format
            output_video  # Output video file
        ]
        subprocess.run(command, check=True)
        print(f"Saved rotated and flipped video to {output_video}")

if __name__ == '__main__':
    csv_path = 'data/filtered_event_data_with_positions.csv'
    data_processor = EventDataTo3DBlock(csv_path)
    data_processor.save_frames('frames_640x480.pt')
    data_processor.save_rotated_frames('frames_640x480.pt')
    data_processor.show_middle_frame()
    # data_processor.convert_to_video()
    data_processor.save_all_frames_as_images("frames")
    # data_processor.frames_to_video("frames_high_resol", 'frames_high_resol_video_with_ffmpeg.mp4')
    data_processor.frames_to_video("frames", 'frames_video_with_ffmpeg.mp4')
    data_processor.save_unique_timestamps_and_positions()
