import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from spectrum_visualizer import SpectrumVisualizer
from tensor_shifter import ShiftCalculator, TensorShifter

from scipy.interpolate import interp1d
import torch.nn.functional as F


class SpectrumReconstructor:
    def __init__(self, visualizer, data_folder, segment_files):
        self.visualizer = visualizer
        self.data_folder = data_folder
        self.segment_files = segment_files

    def load_segment(self, filename):
        return np.load(os.path.join(self.data_folder, filename))

    def convert_to_grayscale(self, rgb_frames):
        # Convert RGB to Grayscale using the luminosity method
        return np.dot(rgb_frames[...,:3], [0.2989, 0.5870, 0.1140])

    def calculate_absorption(self, gray_frames):
        # Assume maximum intensity is the no-absorption baseline
        max_intensity = gray_frames.max()
        return 1 - (gray_frames / max_intensity)


    @staticmethod
    def interpolate_spectrum_quadratic(tensor, target_dim):
        """
        Interpolate the last dimension of a tensor from its current size to a target dimension using quadratic interpolation.
        
        Parameters:
        - tensor (torch.Tensor): The input tensor with shape [H, W, SpectrumDimension]
        - target_dim (int): The desired size of the last dimension after interpolation.
        
        Returns:
        - torch.Tensor: The interpolated tensor with shape [H, W, target_dim]
        """
        # Current shape of the tensor
        H, W, current_dim = tensor.shape
        
        # Original indices and target indices
        original_indices = np.linspace(0, current_dim - 1, current_dim)
        target_indices = np.linspace(0, current_dim - 1, target_dim)
        
        # Create an empty array to store interpolated results
        interpolated_data = np.zeros((H, W, target_dim))
        
        # Iterate over all height and width to apply interpolation to each spectrum
        for i in range(H):
            for j in range(W):
                spectrum = tensor[i, j, :].numpy()  # Extract spectrum as numpy array
                interpolator = interp1d(original_indices, spectrum, kind='quadratic', fill_value="extrapolate")
                interpolated_data[i, j, :] = interpolator(target_indices)
        
        # Convert the numpy array back to tensor
        interpolated_tensor = torch.from_numpy(interpolated_data).float()  # Ensure dtype matches typical PyTorch dtype
        
        return interpolated_tensor

    @staticmethod
    def interpolate_spectrum(tensor, target_dim):
        """
        Interpolate the last dimension of a tensor from its current size to a target dimension.
        
        Parameters:
        - tensor (torch.Tensor or numpy.ndarray): The input tensor with shape [H, W, SpectrumDimension]
        - target_dim (int): The desired size of the last dimension after interpolation.
        
        Returns:
        - torch.Tensor or numpy.ndarray: The interpolated tensor with shape [H, W, target_dim]
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)  # Convert to PyTorch tensor if input is numpy
            
        # Change the shape to [H*W, 1, SpectrumDimension] for 1D interpolation
        H, W, _ = tensor.shape
        tensor_reshaped = tensor.view(-1, 1, tensor.shape[2])
        
        # Interpolate to the new dimension
        interpolated = F.interpolate(tensor_reshaped, size=target_dim, mode='quadratic', align_corners=False)
        
        # Reshape back to [H, W, target_dim]
        interpolated_reshaped = interpolated.view(H, W, target_dim)
        
        # Convert back to numpy if originally numpy
        if isinstance(tensor, np.ndarray):
            return interpolated_reshaped.numpy()  # Convert back to numpy array
        return interpolated_reshaped


    def process_segments(self):

        # Define parameters
        width = 8  # mm
        steps = 346
        lines_per_mm = 600
        distance = 84  # mm
        # Instantiate the ShiftCalculator and compute the shift vector
        shift_calculator = ShiftCalculator(width, steps, lines_per_mm, distance)
        shift_vector = shift_calculator.compute_shift_vector()

        # Instantiate the TensorShifter with the shift vector
        tensor_shifter = TensorShifter(shift_vector)

        for segment_file, scan_dir in self.segment_files:
            frames = self.load_segment(segment_file)[::scan_dir]



            gray_frames = self.convert_to_grayscale(frames)

            gray_frames = torch.tensor(gray_frames).permute(1, 2, 0)

            target_dim = 401
            
            # gray_frames = self.interpolate_spectrum(gray_frames.permute(1, 2, 0)).permute(2, 0, 1)
            # Check type and transpose accordingly
            # if isinstance(gray_frames, np.ndarray):
            #     gray_frames = np.transpose(gray_frames, (1, 2, 0))
            #     interpolated_tensor = self.interpolate_spectrum_quadratic(gray_frames, target_dim)
            #     gray_frames = np.transpose(interpolated_tensor, (2, 0, 1))
            # elif torch.is_tensor(gray_frames):
            
            # gray_frames = gray_frames.permute(1, 2, 0)
            gray_frames = self.interpolate_spectrum_quadratic(gray_frames, target_dim)
            # gray_frames = interpolated_tensor.permute(2, 0, 1)
            
            wavelengths = np.linspace(380, 780, gray_frames.shape[-1])

            absorption = self.calculate_absorption(gray_frames)

            

            absorption = tensor_shifter.apply_shift(absorption)


            # Assuming the last dimension of absorption is the spectrum dimension:
            spectrum_length = absorption.shape[-1]  # or any fixed number like 400 if known
            start_wavelength = 380
            end_wavelength = 780  # example end, adjust according to your spectral range
            wavelengths = np.linspace(start_wavelength, end_wavelength, spectrum_length)

            H, W = absorption.shape[0], absorption.shape[1]
            points = [
                (H//2, W//2),  # Center
                (H//4, W//4),  # Top left quarter
                (3*H//4, W//4),  # Bottom left quarter
                (H//4, 3*W//4),  # Top right quarter
                (3*H//4, 3*W//4)  # Bottom right quarter
            ]

            plt.figure(figsize=(10, 2))
            for i, (x, y) in enumerate(points):
                spectrum = absorption[x, y, :]
                plt.plot(wavelengths, spectrum, label=f'Point ({x},{y})')
            plt.legend()
            plt.title(f'Spectrum Plot for {segment_file}')
            plt.xlabel('Wavelength index')
            plt.ylabel('Absorption')
            # plt.show()
            plt.savefig(f"{self.data_folder}/{segment_file}.png")

            # Take the mean of all frames to get a single spectral image
            # mean_spectrum = absorption.mean(axis=0)
            # Convert to XYZ then RGB
            # xyz = self.visualizer.spectrum_to_xyz(torch.tensor(mean_spectrum[None, ...]))
            # rgb = self.visualizer.xyz_to_rgb(xyz)
            # self.visualizer.visualize_rgb(rgb)


            


            # Convert absorption to a tensor and permute dimensions to match CxHxW for visualizer
            # absorption_tensor = absorption.permute(1, 2, 0)  # Assuming original shape is N x H x W
            # print("permuted shape: ", absorption_tensor.shape)



            # Visualize and save each segment's spectrum
            output_file_path = f"{self.data_folder}/{segment_file.split('/')[-1].replace('.npy', '_spectrum.png')}"
            rgb_image = self.visualizer.visualize_and_save(1-absorption, wavelengths, output_file_path)

            print(f"Processed and saved spectrum visualization for {segment_file} to {output_file_path}")

if __name__ == "__main__":
    visualizer = SpectrumVisualizer('ciexyz31_1.txt')
    data_folder = 'data/segmented_frames'
    segment_files = [('segment_32_45.npy', 1), ('segment_45_58.npy', -1)]  # Assuming these are the third and fourth segments
    reconstructor = SpectrumReconstructor(visualizer, data_folder, segment_files)
    reconstructor.process_segments()
