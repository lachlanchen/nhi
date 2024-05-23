import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tensor_accumulation import TensorAccumulator  # Ensure this module has all necessary imports and classes
from tensor_shifter import ShiftCalculator, TensorShifter  # Ensure these classes are defined as shown earlier
import shutil
import datetime
from scipy.interpolate import griddata
from tqdm import tqdm, trange
import torch.nn.functional as F

from scipy.interpolate import interp1d
import cv2

from spectrum_visualizer import SpectrumVisualizer
from events2spectrum import EventsSpectrumReconstructor


class HorographySpectrometer:
    def __init__(self, file_path, intervals=1000, magic_code='otter'):
        self.file_path = file_path
        self.intervals = intervals
        self.magic_code = magic_code
        self.file_path, self.output_folder = self.setup_environment()
        self.shift_calculator = ShiftCalculator(width=8, steps=346, lines_per_mm=600, distance=84)
        self.tensor_shifter = TensorShifter(self.shift_calculator.compute_shift_vector())

    # def setup_output_folder(self):
    #     base_dir = os.path.dirname(self.file_path)
    #     output_folder = os.path.join(base_dir, f"data-{self.magic_code}")
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     return output_folder

    def setup_environment(self):
        # Setup the directory with the given magic code
        base_dir = os.path.dirname(self.file_path)
        output_folder = f"data-{self.magic_code}"
        new_file_path = os.path.join(output_folder, os.path.basename(self.file_path))
        # os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(new_file_path):
            os.makedirs(output_folder, exist_ok=True)
            # Copy the tensor file to the new directory
            shutil.copy(self.file_path, new_file_path)
        return new_file_path, output_folder

    def visualize(self, original, processed, description, vis_folder):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        # plt.imshow(original.numpy(), cmap='hot')
        plt.imshow(original.numpy(), cmap='gray')
        plt.title('Original')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        # plt.imshow(processed.numpy(), cmap='hot')
        plt.imshow(processed.numpy(), cmap='gray')
        plt.title('Processed')
        plt.colorbar()
        plt.suptitle(f'Visualization of Changes: {description}')
        plt.savefig(os.path.join(vis_folder, f'{description}.png'))
        plt.close()

    def visualize_along_axis(self, tensor_old, tensor_new, axis, n_steps, vis_folder):
        steps = np.linspace(0, tensor_old.shape[axis] - 1, n_steps, dtype=int)
        axis_folder = os.path.join(vis_folder, f'axis_{axis}')
        os.makedirs(axis_folder, exist_ok=True)
        for step in tqdm(steps):
            slice_description = f'step_{step}'
            self.visualize(tensor_old[:, :, step] if axis == 2 else tensor_old[:, step] if axis == 1 else tensor_old[step],
                           tensor_new[:, :, step] if axis == 2 else tensor_new[:, step] if axis == 1 else tensor_new[step],
                           slice_description, axis_folder)

    def multi_level_visualization(self, tensor_old, tensor_new, timestamp_all, n_steps=10, description=""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        vis_folder = os.path.join(self.output_folder, f"visualization-{timestamp_all}", f"visualization_{description}_{timestamp}")
        os.makedirs(vis_folder, exist_ok=True)
        for axis in range(3):  # Assuming tensor is 3D
            self.visualize_along_axis(tensor_old, tensor_new, axis, n_steps, vis_folder)

    def load_and_accumulate(self):
        accumulator = TensorAccumulator(self.file_path)
        # accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        accumulated_tensor = accumulator.accumulate_continuous(self.intervals)
        return accumulated_tensor

    def load_and_accumulate_interval(self):
        accumulator = TensorAccumulator(self.file_path)
        # accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        return accumulated_tensor

    def accumulate_continuous(self, tensor):

        accumulator = TensorAccumulator(self.file_path)
        accumulator.tensor = tensor.permute(2, 0, 1)
        # accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        accumulated_tensor = accumulator.accumulate_continuous(self.intervals)
        return accumulated_tensor.permute(1, 2, 0)

    def accumulate_interval(self, tensor):
        accumulator = TensorAccumulator(self.file_path)
        accumulator.tensor = tensor.permute(2, 0, 1)
        # accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        accumulated_tensor = accumulator.accumulate_interval(self.intervals)
        return accumulated_tensor.permute(1, 2, 0)

    def rescale(self, tensor):
        return tensor / tensor.max()

    def exponentiate(self, tensor, k):
        return torch.exp(tensor * k)

    def permute(self, tensor, order=(1, 2, 0)):
        return tensor.permute(order)

    def centralize(self, tensor):
        H,_,_ = tensor.shape
        mean = tensor[:H//2].mean(dim=0, keepdim=True)
        return tensor - mean

    def compute_absolute(self, tensor):
        return torch.abs(tensor)


    def shift(self, tensor):
        shifted_tensor = self.tensor_shifter.apply_shift(tensor)
        # self.visualize(tensor[0], shifted_tensor[0], 'shift')  # Visualize the first frame before and after shift
        return shifted_tensor


    def smooth_surface_griddata(self, tensor):
        """ Applies smoothing using grid data interpolation for each frame in the tensor."""
        timestamps, height, width = tensor.shape
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)

        smoothed_tensor = torch.zeros_like(tensor)

        for i in tqdm(range(timestamps)):
            # Convert tensor to numpy for interpolation
            frame = tensor[i].numpy()
            # Flatten the arrays for griddata input
            points = np.vstack((y.ravel(), x.ravel())).T
            values = frame.ravel()

            # Interpolate using cubic spline
            grid_z = griddata(points, values, (y, x), method='cubic')
            smoothed_tensor[i] = torch.from_numpy(grid_z)

        return smoothed_tensor

    
    # def smooth_surface(self, tensor):
    #     # define a iterative logic that moving average H and centralize over H, then moving average W and centralize over W
    #     # avoid coupling
    #     # a separate function for each sub task 

    def smooth_surface(self, tensor):
        """Smooths the surface by applying a moving average over the height and width, then centralizes it."""
        original_dtype = tensor.dtype  # Store the original data type of the tensor
        if original_dtype != torch.float32:
            tensor = tensor.to(torch.float32)  # Convert tensor to float32 for processing

        # Moving average over height
        kernel_size = 5  # Example kernel size
        tensor_smooth = F.avg_pool2d(tensor, (kernel_size, 1), stride=1, padding=(kernel_size//2, 0))

        # Moving average over width
        tensor_smooth = F.avg_pool2d(tensor_smooth, (1, kernel_size), stride=1, padding=(0, kernel_size//2))

        # Centralize each frame
        tensor_smooth -= tensor_smooth.mean(dim=[1, 2], keepdim=True)

        if original_dtype != torch.float32:
            tensor_smooth = tensor_smooth.to(original_dtype)  # Convert back to the original data type

        return tensor_smooth


    # def process(self):
    #     print("Starting processing of tensor data...")
    #     tensor_acc = self.load_and_accumulate()

    #     T, H, W = tensor_acc.shape


    #     tensor_rescaled = tensor_acc / tensor_acc.max()

    #     k = 3
    #     tensor_exp = torch.exp(tensor_rescaled * k)

    #     tensor_per = tensor_exp.permute(1, 2, 0)  # Ensure correct dimension order for processing


    #     tensor_mean = torch.mean(tensor_per[:H//2], axis=0, keepdim=True)
    #     tensor_centralized = tensor_per - tensor_mean
    #     self.multi_level_visualization(tensor_per, tensor_centralized, description="centralize")

    #     tensor_abs = torch.abs(tensor_centralized)
    #     self.multi_level_visualization(tensor_centralized, tensor_abs, description="abs")


    #     tensor_shift = self.shift(tensor_abs)
    #     # Additional steps like smoothing or further processing would go here, with visualization at each step
    #     self.multi_level_visualization(tensor_abs, tensor_shift, description="shift")
    #     print("Processing shift complete. Output saved in:", self.output_folder)

    #     tensor_smooth = self.smooth_surface(tensor_shift)  # Smooth each frame
    #     self.multi_level_visualization(tensor_shift, tensor_smooth, description="smooth")
    #     print("Processing smooth complete. Output saved in:", self.output_folder)

    # def smooth_interpolate_height(self, tensor):
    #     # Apply smoothing for each vertical slice across all frames and width columns
    #     smoothed_tensor = torch.zeros_like(tensor)
    #     for i in tqdm(range(tensor.shape[0])):  # Iterate through each frame
    #         for j in range(tensor.shape[1]):  # Iterate through each width
    #             smoothed_tensor[i, :, j] = self.apply_spline_smoothing(tensor[i, :, j])
    #     return smoothed_tensor

    # def apply_spline_smoothing(self, data_vector):
    #     # Placeholder for spline smoothing or any other smoothing technique
    #     # This could be more sophisticated based on the specific requirements
    #     x = torch.linspace(0, len(data_vector) - 1, steps=len(data_vector))
    #     smoothed_data = torch.tensor(np.interp(x, x, data_vector.numpy()), dtype=torch.float32)
    #     return smoothed_data


    def smooth_interpolate_height(self, tensor):
        """Applies moving average smoothing for each vertical slice across all frames and width columns."""
        H, W, T = tensor.shape  # Ensure dimensions are correctly ordered for TxHxW
        period = H // 16

        # Convert tensor to float for processing if it's not already
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()

        # Pad the tensor with zeros on top to handle boundary conditions for moving average
        padded_tensor = F.pad(tensor, (0, 0, 0, 0, period, 0), "constant", 0)

        # Define a moving average kernel of size H//4
        kernel = torch.ones((1, 1, period), device=tensor.device) / period

        smoothed_tensor = torch.zeros_like(tensor)

        # Apply convolution to each width independently
        for w in trange(W, desc="Smoothing columns"):
            for t in range(T):
                # Extract a single column across all heights for this frame
                column = padded_tensor[:, w, t].unsqueeze(0).unsqueeze(0)
                # Apply 1D convolution along the height dimension
                smoothed_column = F.conv1d(column, kernel, padding=period//2)
                # Store the smoothed result back, trimming the excess due to padding
                smoothed_tensor[:, w, t] = smoothed_column.squeeze().narrow(0, period, H)

        return smoothed_tensor

    # def smooth_interpolate_height(self, tensor):
    #     """Applies quadratic spline smoothing for each vertical slice across all frames and width columns."""
    #     H, W, T = tensor.shape  # Ensure dimensions are correctly ordered for TxHxW

    #     # Convert tensor to float for processing if it's not already
    #     if tensor.dtype == torch.uint8:
    #         tensor = tensor.float()

    #     smoothed_tensor = torch.zeros_like(tensor)

    #     # Apply spline smoothing to each column independently
    #     for w in trange(W, desc="Smoothing columns"):
    #         for t in range(T):
    #             # Extract a single column across all heights for this frame
    #             y = tensor[:, w, t].cpu().numpy()  # Get the column as a NumPy array
    #             x = np.arange(H)  # Independent variable for interpolation

    #             # Fit a quadratic spline
    #             spline = interp1d(x, y, kind='quadratic', fill_value="extrapolate")

    #             # Evaluate the spline over the original x positions
    #             smoothed_y = spline(x)

    #             # Store the smoothed result back
    #             smoothed_tensor[:, w, t] = torch.from_numpy(smoothed_y)

    #     return smoothed_tensor

    def smooth_frame_2d(self, tensor, kernel_size=5):
        """
        Applies a 2D moving average across each frame of the tensor.
        
        Args:
            tensor (torch.Tensor): The input tensor (assumed to be in HxWxT format).
            kernel_size (int): The size of the moving average window, must be an odd number.
        
        Returns:
            torch.Tensor: The smoothed tensor.
        """

        # Convert tensor to float for processing if it's not already
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()

        # Ensure kernel size is odd to have a central pixel
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        # Prepare the 2D averaging kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size * kernel_size)
        kernel = kernel.to(tensor.device)

        # Adding batch and channel dimensions to tensor if they are not present
        if len(tensor.shape) == 3:  # HxWxT
            tensor = tensor.unsqueeze(1)  # Add a channel dimension for conv2d: 1xHxWxT

        # Apply the 2D convolution to each frame across the batch dimension
        padding = kernel_size // 2
        smoothed_tensor = F.conv2d(tensor, kernel, padding=padding, groups=1)

        # Remove the channel dimension and return
        return smoothed_tensor.squeeze(1)

    def apply_fft_processing(self, tensor, threshold=0.1):
        """
        Processes tensor using FFT to smooth uniform areas and enhance areas with variations.
        
        Args:
            tensor (torch.Tensor): Input tensor (HxWxT).
            threshold (float): Threshold for filtering frequencies.
        
        Returns:
            torch.Tensor: Tensor after inverse FFT and selective enhancement.
        """
        # FFT to frequency domain
        fft_tensor = torch.fft.fft2(tensor)
        magnitude_spectrum = torch.abs(fft_tensor)

        # Create a frequency mask that preserves lower frequencies more than higher frequencies
        H, W, T = tensor.shape
        freq_mask = torch.ones((H, W, T), dtype=torch.float32)
        center_h, center_w = H // 2, W // 2
        for h in range(H):
            for w in range(W):
                distance = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
                freq_mask[h, w, :] = 1 / (1 + np.exp((distance - center_w * threshold) / 10))  # Sigmoid function for smooth transition

        # Apply mask and inverse FFT
        filtered_fft_tensor = fft_tensor * freq_mask.to(fft_tensor.device)
        ifft_tensor = torch.fft.ifft2(filtered_fft_tensor).real

        # Enhance local contrast
        ifft_tensor_np = ifft_tensor.cpu().numpy()
        for i in range(T):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ifft_tensor_np[:, :, i] = clahe.apply((ifft_tensor_np[:, :, i] * 255).astype(np.uint8)) / 255

        return torch.from_numpy(ifft_tensor_np).to(tensor.device)


    def process(self):
        print("Starting processing of tensor data...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        self.intervals = 1000
        tensor = self.load_and_accumulate_interval()

        # i want to move permutation here and for accumulate, permute its input and output inside the function

        tensor_old = tensor
        tensor = self.permute(tensor)
        # self.multi_level_visualization(tensor_old, tensor, description="permuted")

        # tensor_old = tensor
        # tensor = self.shift(tensor)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="shifted")

        

        # define a function that interplate each H-vector
        # Apply height vector smoothing
        # tensor_old = tensor
        # tensor = self.smooth_interpolate_height(tensor)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="height_interpolated")

        # tensor_old = tensor
        # tensor = self.apply_fft_processing(tensor)
        # self.multi_level_visualization(tensor_old, tensor, "FFT and CLAHE enhanced")


        # # Apply 2D smoothing to each frame
        # tensor_old = tensor
        # tensor = self.smooth_frame_2d(tensor, kernel_size=5)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="2D_smoothed")

        self.intervals = 22

        tensor = self.accumulate_continuous(tensor)

        print("tensor.shape: ", tensor.shape)

        

        tensor_old = tensor
        tensor = self.rescale(tensor)
        self.multi_level_visualization(tensor_old, tensor, timestamp, description="rescaled")

        # tensor_old = tensor
        # tensor = self.exponentiate(tensor, k=0.5)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="exponentiated")

        tensor_old = tensor
        tensor = self.centralize(tensor)
        self.multi_level_visualization(tensor_old, tensor, timestamp, description="centralized")

        
        # tensor_old = tensor
        # tensor = self.compute_absolute(tensor)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="absolute")

        # tensor_old = tensor
        # tensor = self.shift(tensor)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="shifted")

        

        # tensor_old = tensor
        # tensor = self.smooth_surface(tensor)
        # self.multi_level_visualization(tensor_old, tensor, timestamp, description="smoothed")

        print("Processing complete. Output saved in:", self.output_folder)

        # Save the final processed tensor
        save_path = os.path.join(self.output_folder, f"processed_tensor_{timestamp}.pt")
        torch.save(tensor.permute(2, 0, 1), save_path)
        print(f"Processed tensor saved at: {save_path}")

        print("Processing complete. Output saved in:", self.output_folder)


        visualizer = SpectrumVisualizer('ciexyz31_1.txt')
        data_folder = self.output_folder
        event_files = [(f'processed_tensor_{timestamp}.pt', 1),
                       ]
        reconstructor = EventsSpectrumReconstructor(visualizer, data_folder, event_files)
        outputs = reconstructor.process_events()


def main():
    parser = argparse.ArgumentParser(description='Process tensor data for spectrometer analysis.')
    parser.add_argument('file_path', type=str, help='Path to the tensor file.')
    args = parser.parse_args()
    spectrometer = HorographySpectrometer(args.file_path)
    spectrometer.process()

if __name__ == "__main__":
    main()
