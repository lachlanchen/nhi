import torch
import numpy as np
from pprint import pprint

class ShiftCalculator:
    def __init__(self, width=8, steps=346, height=8, height_steps=260, lines_per_mm=600, distance=84, reverse=False, adjustment_factor=.8, adjust_at='distance', axis='width'):
        self.width = width  # Width of the sensor in mm
        self.steps = steps  # Number of discrete steps along the width
        self.height = height  # Height of the sensor in mm
        self.height_steps = height_steps  # Number of discrete steps along the height
        self.lines_per_mm = lines_per_mm  # Density of the grating lines per mm
        self.original_distance = distance  # Store the original distance for reporting
        self.distance = distance  # Distance from the grating to the sensor in mm
        self.reverse = -1 if reverse else 1  # Reverse the direction of the wavelength range if needed
        self.adjustment_factor = adjustment_factor  # Adjustment factor for scaling
        self.adjust_at = adjust_at  # Determines whether the adjustment is at 'distance' or 'final'
        self.axis = axis  # Axis along which to compute the shift ('width', 'height', or 'both')

        # If adjustment is to be made at the distance, modify the distance and print the values
        if self.adjust_at == 'distance':
            print(f"Original Distance: {self.distance} mm")
            self.distance = self.distance / self.adjustment_factor  # Apply adjustment by dividing the distance
            print(f"Adjusted Distance: {self.distance} mm")

    def compute_shift_vector(self):
        d = 1 / self.lines_per_mm  # Calculate grating spacing in mm

        if self.axis == 'width':
            theta_min = np.arctan((-self.width / 2) / self.distance)
            theta_max = np.arctan((self.width / 2) / self.distance)
            lambda_min = d * np.sin(theta_min)
            lambda_max = d * np.sin(theta_max)
            wavelengths = torch.linspace(lambda_min, lambda_max, self.steps) * self.reverse
            if self.adjust_at == 'final':
                wavelengths *= self.adjustment_factor
            wavelengths -= 50e-6  # Apply a constant offset
            return wavelengths

        elif self.axis == 'height':
            theta_min = np.arctan((-self.height / 2) / self.distance)
            theta_max = np.arctan((self.height / 2) / self.distance)
            lambda_min = d * np.sin(theta_min)
            lambda_max = d * np.sin(theta_max)
            wavelengths = torch.linspace(lambda_min, lambda_max, self.height_steps) * self.reverse
            if self.adjust_at == 'final':
                wavelengths *= self.adjustment_factor
            wavelengths -= 50e-6  # Apply a constant offset
            return wavelengths

        elif self.axis == 'both':
            shift_matrix = torch.zeros(self.height_steps, self.steps)
            for i in range(self.height_steps):
                y = (i - self.height_steps / 2) * (self.height / self.height_steps)
                for j in range(self.steps):
                    x = (j - self.steps / 2) * (self.width / self.steps)
                    theta = np.arctan2(np.sqrt(x**2 + y**2), self.distance)
                    lambda_shift = d * np.sin(theta)
                    if self.reverse:
                        lambda_shift = -lambda_shift
                    if self.adjust_at == 'final':
                        lambda_shift *= self.adjustment_factor
                    lambda_shift -= 50e-6  # Apply a constant offset
                    shift_matrix[i, j] = lambda_shift
            return shift_matrix

class TensorShifter:
    def __init__(self, shift_vector, axis='width'):
        self.shift_vector = shift_vector
        self.axis = axis  # Axis along which to apply the shift ('width', 'height', or 'both')

    @staticmethod
    def calculate_index_shift(shift, step_size):
        # Convert to numpy float for robust rounding and then convert to int
        return int(np.rint(shift / step_size))

    def apply_shift(self, tensor):
        num_wavelength_steps = tensor.shape[-1]
        wavelength_range = (780 - 380) * 1e-6  # Total wavelength range in meters
        step_size = wavelength_range / (num_wavelength_steps - 1)

        shifted_tensor = torch.ones_like(tensor)
        if self.axis == 'width':
            for i, shift in enumerate(self.shift_vector):
                index_shift = self.calculate_index_shift(shift, step_size)
                if index_shift < 0:
                    shifted_tensor[:, i, :max(num_wavelength_steps + index_shift, 0)] = tensor[:, i, -index_shift:num_wavelength_steps]
                elif index_shift > 0:
                    shifted_tensor[:, i, index_shift:num_wavelength_steps] = tensor[:, i, :num_wavelength_steps - index_shift]
                else:
                    shifted_tensor[:, i, :] = tensor[:, i, :]
        elif self.axis == 'height':
            for i, shift in enumerate(self.shift_vector):
                index_shift = self.calculate_index_shift(shift, step_size)
                if index_shift < 0:
                    shifted_tensor[i, :, :max(num_wavelength_steps + index_shift, 0)] = tensor[i, :, -index_shift:num_wavelength_steps]
                elif index_shift > 0:
                    shifted_tensor[i, :, index_shift:num_wavelength_steps] = tensor[i, :, :num_wavelength_steps - index_shift]
                else:
                    shifted_tensor[i, :, :] = tensor[i, :, :]
        elif self.axis == 'both':
            for i in range(tensor.shape[0]):  # Over height
                for j in range(tensor.shape[1]):  # Over width
                    shift = self.shift_vector[i, j]
                    index_shift = self.calculate_index_shift(shift, step_size)
                    if index_shift < 0:
                        shifted_tensor[i, j, :max(num_wavelength_steps + index_shift, 0)] = tensor[i, j, -index_shift:num_wavelength_steps]
                    elif index_shift > 0:
                        shifted_tensor[i, j, index_shift:num_wavelength_steps] = tensor[i, j, :num_wavelength_steps - index_shift]
                    else:
                        shifted_tensor[i, j, :] = tensor[i, j, :]
        return shifted_tensor

if __name__ == "__main__":
    # Define parameters
    width = 8  # mm
    steps = 346  # Number of width steps
    height = 8  # mm
    height_steps = 260  # Number of height steps
    lines_per_mm = 600
    distance = 84  # mm
    num_wavelength_steps = 13  # Arbitrary choice

    # Create a tensor of shape [260, 346, 13] with random values
    initial_tensor = torch.randn(height_steps, steps, num_wavelength_steps)

    # Instantiate the ShiftCalculator and compute the shift vector along the height axis
    shift_calculator = ShiftCalculator(width, steps, height, height_steps, lines_per_mm, distance, axis='height')
    shift_vector = shift_calculator.compute_shift_vector()

    pprint(shift_vector)

    # Instantiate the TensorShifter with the shift vector
    tensor_shifter = TensorShifter(shift_vector, axis='height')

    pprint(initial_tensor[10, 10])

    # Apply the shift
    shifted_tensor = tensor_shifter.apply_shift(initial_tensor)

    pprint(shifted_tensor[10, 10])

    # Output the shape of the shifted tensor to verify
    print("Shape of the shifted tensor:", shifted_tensor.shape)
