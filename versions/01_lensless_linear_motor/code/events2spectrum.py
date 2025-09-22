import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from spectrum_visualizer import SpectrumVisualizer
# from tensor_shifter import ShiftCalculator, TensorShifter

from scipy.interpolate import interp1d
import torch.nn.functional as F
import cv2

class EventsSpectrumReconstructor:
    def __init__(self, visualizer, data_folder, event_files):
        self.visualizer = visualizer
        self.data_folder = data_folder
        self.event_files = event_files

    def load_event(self, filename):
        return torch.load(os.path.join(self.data_folder, filename))

    def calculate_absorption(self, tensor_data):
        max_intensity = tensor_data.max()
        return 1 - (tensor_data / max_intensity)

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

    def rescale(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    def process_events(self):
        # shift_calculator = ShiftCalculator(width=8, steps=346, lines_per_mm=600, distance=84)
        # shift_vector = shift_calculator.compute_shift_vector()
        # tensor_shifter = TensorShifter(shift_vector)

        outputs = []
        for event_file, scan_dir in self.event_files:
            events = self.load_event(event_file)[::scan_dir]
            events_permuted = events.permute(1, 2, 0)  # Ensure correct dimension order for processing
            target_dim = 401
            # events_interpolated = self.interpolate_spectrum_quadratic(events_permuted, target_dim)
            events_interpolated = events_permuted
            wavelengths = np.linspace(380, 780, events_interpolated.shape[-1])
            absorption = self.calculate_absorption(events_interpolated)
            # shifted_absorption = tensor_shifter.apply_shift(absorption)
            shifted_absorption = absorption

            H, W = shifted_absorption.shape[0], shifted_absorption.shape[1]
            points = [
                (H // 2, W // 2),  # Center point
                (H // 4, W // 4),  # Top left quarter
                (3 * H // 4, W // 4),  # Bottom left quarter
                (H // 4, 3 * W // 4),  # Top right quarter
                (3 * H // 4, 3 * W // 4)  # Bottom right quarter
            ]


            plt.figure(figsize=(10, 2))
            for i, (x, y) in enumerate(points):
                spectrum = shifted_absorption[x, y, :]
                plt.plot(wavelengths, spectrum, label=f'Point ({x},{y})')
            plt.legend()
            plt.title(f'Spectrum Plot for {event_file}')
            plt.xlabel('Wavelength index')
            plt.ylabel('Absorption')
            plt.savefig(f"{self.data_folder}/{event_file.replace('.pt', '_spectrum.png')}")

            output_file_path = f"{self.data_folder}/{event_file.replace('.pt', '_visualized.png')}"
            output = self.visualizer.visualize_and_save(1-shifted_absorption, wavelengths, output_file_path)
            outputs.append(output)

            print(f"Processed and saved spectrum visualization for {event_file} to {output_file_path}")



        # cv2.imwrite(f"{self.data_folder}/continuous_minus_interval.png", self.rescale((self.rescale(outputs[0])-self.rescale(outputs[1])).numpy())*255)
        return outputs

    def apply_gaussian_blur(self, outputs):
        blurred_outputs = []
        for output in outputs:
            # Convert tensor to numpy array and scale to 0-255
            output_image = (output.numpy() * 255).astype(np.uint8)
            # Apply Gaussian blur
            blurred_image = cv2.GaussianBlur(output_image, (5, 5), 1)
            blurred_outputs.append(blurred_image)

        # Calculate blurred difference
        blurred_difference = blurred_outputs[0].astype(float) - blurred_outputs[1].astype(float)
        # Rescale to 0-255
        blurred_difference = np.clip((blurred_difference - blurred_difference.min()) / (blurred_difference.max() - blurred_difference.min()) * 255, 0, 255).astype(np.uint8)

        # Save images
        cv2.imwrite(f"{self.data_folder}/blurred_output0.png", blurred_outputs[0])
        cv2.imwrite(f"{self.data_folder}/blurred_output1.png", blurred_outputs[1])
        cv2.imwrite(f"{self.data_folder}/blurred_difference.png", blurred_difference)

        return blurred_outputs[0], blurred_outputs[1], blurred_difference

    def apply_denoise_and_difference(self, outputs):
        denoised_outputs = []
        for output in outputs:
            # Convert tensor to numpy array and scale to 0-255
            output_image = (output.numpy() * 255).astype(np.uint8)
            # Apply non-local means denoising
            denoised_image = cv2.fastNlMeansDenoisingColored(output_image, None, 10, 10, 7, 21)
            denoised_outputs.append(denoised_image)

        # Calculate denoised difference
        denoised_difference = denoised_outputs[0].astype(float) - denoised_outputs[1].astype(float)
        # Rescale to 0-255
        denoised_difference = np.clip((denoised_difference - denoised_difference.min()) / (denoised_difference.max() - denoised_difference.min()) * 255, 0, 255).astype(np.uint8)

        # Save images
        cv2.imwrite(f"{self.data_folder}/denoised_output0.png", denoised_outputs[0])
        cv2.imwrite(f"{self.data_folder}/denoised_output1.png", denoised_outputs[1])
        cv2.imwrite(f"{self.data_folder}/denoised_difference.png", denoised_difference)

        return denoised_outputs[0], denoised_outputs[1], denoised_difference

if __name__ == "__main__":
    visualizer = SpectrumVisualizer('ciexyz31_1.txt')
    data_folder = 'data100/segmented_events/blurred_frames/'
    event_files = [('continuous_accumulation_22_post.pt', 1),
                   ('interval_accumulation_22_post.pt', 1)]
    reconstructor = EventsSpectrumReconstructor(visualizer, data_folder, event_files)
    outputs = reconstructor.process_events()
    # o0_blurred, o1_blurred, diff_blurred = reconstructor.apply_gaussian_blur(outputs)
    # o0_denoised, o1_denoised, diff_denoised = reconstructor.apply_denoise_and_difference(outputs)
