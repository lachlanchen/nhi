import torch
import numpy as np
from pprint import pprint

class ShiftCalculator:
    def __init__(self, width=8, steps=346, lines_per_mm=600, distance=84, reverse=False):
        self.width = width
        self.steps = steps
        self.lines_per_mm = lines_per_mm
        self.distance = distance
        self.reverse = -1 if reverse else 1

    def compute_shift_vector(self):
        d = 1 / self.lines_per_mm  # mm per line
        theta_min = np.arctan((0 - self.width / 2) / self.distance)
        theta_max = np.arctan((self.width / 2) / self.distance)
        lambda_min = d * np.sin(theta_min)
        lambda_max = d * np.sin(theta_max)
        return torch.linspace(lambda_min, lambda_max, self.steps) * self.reverse * 0.7 - 50 * 1e-6

class TensorShifter:
    def __init__(self, shift_vector):
        self.shift_vector = shift_vector

    @staticmethod
    def calculate_index_shift(shift, step_size):
        # Convert to numpy float for robust rounding and then convert to int
        return int(np.rint(shift / step_size))

    def apply_shift(self, tensor):
        num_wavelength_steps = tensor.shape[-1]
        # wavelength_range = self.shift_vector[-1] - self.shift_vector[0]
        wavelength_range = (780 - 380) * 1e-6
        step_size = wavelength_range / (num_wavelength_steps - 1)

        shifted_tensor = torch.ones_like(tensor)
        for i, shift in enumerate(self.shift_vector):
            index_shift = self.calculate_index_shift(shift, step_size)
            if index_shift < 0:
                shifted_tensor[:, i, :max(num_wavelength_steps + index_shift, 0)] = tensor[:, i, -index_shift:num_wavelength_steps]
            elif index_shift > 0:
                shifted_tensor[:, i, index_shift:num_wavelength_steps] = tensor[:, i, :num_wavelength_steps - index_shift]
            else:
                shifted_tensor[:, i, :] = tensor[:, i, :]
        return shifted_tensor

if __name__ == "__main__":
    # Define parameters
    width = 8  # mm
    steps = 346
    lines_per_mm = 600
    distance = 84  # mm
    num_wavelength_steps = 13  # Arbitrary choice

    # Create a tensor of shape [260, 346, 13] with random values
    initial_tensor = torch.randn(260, 346, num_wavelength_steps)

    # Instantiate the ShiftCalculator and compute the shift vector
    shift_calculator = ShiftCalculator(width, steps, lines_per_mm, distance)
    shift_vector = shift_calculator.compute_shift_vector()

    pprint(shift_vector)

    # Instantiate the TensorShifter with the shift vector
    tensor_shifter = TensorShifter(shift_vector)

    pprint(initial_tensor[10, 10])

    # Apply the shift
    shifted_tensor = tensor_shifter.apply_shift(initial_tensor)

    pprint(shifted_tensor[10, 10])

    # Output the shape of the shifted tensor to verify
    print("Shape of the shifted tensor:", shifted_tensor.shape)

