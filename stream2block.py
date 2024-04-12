import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import subprocess


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

    def save_unique_timestamps_and_positions(self):
        # Load data
        df = pd.read_csv(self.csv_path)

        # Assuming each 'event_timestamp' maps to a unique 'axis_y_position_mm' position directly
        unique_timestamps_positions = df[['event_timestamp', 'axis_y_position_mm']].drop_duplicates()

        # Saving the unique timestamps and their corresponding y positions
        unique_timestamps_positions.to_csv('unique_timestamps_and_y_positions.csv', index=False)
        print("Saved unique timestamps and Y positions to 'unique_timestamps_and_y_positions.csv'.")



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

    def save_rotated_frames(self, filename):
        rotated_frames = torch.rot90(self.frames, k=1, dims=[-2, -1])
        torch.save(rotated_frames, f"rotated_{filename}")
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

    # def convert_to_video(self, output_video='frames_video.mp4', fps=10):
    #     fig, ax = plt.subplots()
        
    #     def update(i):
    #         ax.clear()
    #         frame = self.frames[i].numpy()
    #         ax.imshow(frame, cmap='gray')
    #         y_pos, x_pos = (frame == 1).nonzero()
    #         y_neg, x_neg = (frame == -1).nonzero()
    #         ax.scatter(x_pos, y_pos, color='green', s=1)
    #         ax.scatter(x_neg, y_neg, color='red', s=1)
    #         return ax,

    #     ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True)
    #     writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        
    #     ani.save(output_video, writer=writer)
    #     print(f"Saved video to {output_video}")

    def convert_to_video(self, output_video='frames_video.mp4', fps=10):
        # Define the figure size and axes limits to ensure dots are visible and maintain aspect ratio
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Set limits to match the image coordinates
        
        def update(i):
            ax.clear()  # Clear the plot to draw the next frame
            frame = self.frames[i].numpy()
            
            # Use imshow to show the frame with adjusted contrast if needed
            ax.imshow(frame, cmap='gray', vmin=-1, vmax=1)  # Adjust vmin and vmax for contrast
            ax.set_xlim(0, 640)
            ax.set_ylim(480, 0)
            
            # Extract positions where polarity is +1 and -1
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            
            # Increase the size of the scatter plot dots
            ax.scatter(x_pos, y_pos, color='green', s=10, label='True (+1)')
            ax.scatter(x_neg, y_neg, color='red', s=10, label='False (-1)')
            
            # Optionally, add a legend in the first frame or use a fixed legend outside the loop
            if i == 0:
                ax.legend(loc='upper right')
            
            return ax,

        ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True)
        writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        
        ani.save(output_video, writer=writer)
        print(f"Saved video to {output_video}")


    # def save_all_frames_as_images(self, folder='frames'):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
        
    #     for i in range(len(self.frames)):
    #         frame = self.frames[i].numpy()
    #         fig, ax = plt.subplots()
    #         ax.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
    #         y_pos, x_pos = (frame == 1).nonzero()
    #         y_neg, x_neg = (frame == -1).nonzero()
    #         ax.scatter(x_pos, y_pos, color='green', s=1)
    #         ax.scatter(x_neg, y_neg, color='red', s=1)
    #         plt.axis('off')  # Hide axis
    #         plt.savefig(f"{folder}/frame_{i:04d}.png", bbox_inches='tight', pad_inches=0)
    #         plt.close()

    #     print(f"Saved all frames as images in the {folder}/ directory.")

    def save_all_frames_as_images(self, folder='frames_high_resol'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for i in range(len(self.frames)):
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize to maintain aspect ratio
            ax.axis('off')  # Turn off the axis
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove margins

            frame = self.frames[i].numpy()
            ax.imshow(frame, cmap='gray', extent=[0, 640, 480, 0])
            y_pos, x_pos = (frame == 1).nonzero()
            y_neg, x_neg = (frame == -1).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=1)
            ax.scatter(x_neg, y_neg, color='red', s=1)

            # Save the frame as an image
            plt.savefig(f"{folder}/frame_{i:04d}.png", dpi=100)  # Adjust dpi to control the image size
            plt.close()

        print(f"Saved all frames as images in the {folder}/ directory.")


    # def frames_to_video(self, folder='frames', output_video='frames_video_with_ffmpeg.mp4', fps=10):
    #     # self.save_all_frames_as_images(folder=folder)
    #     command = [
    #         'ffmpeg', '-y',  # '-y' option overwrites output file if it exists
    #         '-framerate', str(fps), 
    #         '-i', f'{folder}/frame_%04d.png', 
    #         '-c:v', 'libx264', 
    #         '-pix_fmt', 'yuv420p', 
    #         output_video
    #     ]
    #     subprocess.run(command)
    #     print(f"Compiled frames into video: {output_video}")

    def frames_to_video(self, folder='frames', output_video='frames_video_with_ffmpeg.mp4', fps=60):
        command = [
            'ffmpeg', '-y',  # '-y' option overwrites output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(folder, 'frame_%04d.png'),
            '-vf', 'scale=800:600',  # Scale input frames to 640x480
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_video}")


if __name__ == '__main__':
    # Example usage
    csv_path = 'data/filtered_event_data_with_positions.csv'
    data_processor = EventDataTo3DBlock(csv_path)
    # data_processor.save_frames('frames_640x480.pt')
    # data_processor.save_rotated_frames('frames_640x480.pt')
    data_processor.show_middle_frame()
    # data_processor.convert_to_video()
    # data_processor.save_all_frames_as_images("frames")
    # data_processor.frames_to_video("frames_high_resol", 'frames_high_resol_video_with_ffmpeg.mp4')
    # data_processor.frames_to_video("frames", 'frames_video_with_ffmpeg.mp4')
    data_processor.save_unique_timestamps_and_positions()