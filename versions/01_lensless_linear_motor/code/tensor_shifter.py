import torch
import numpy as np
from pprint import pprint

# class ShiftCalculator:
#     def __init__(self, width=8, steps=346, lines_per_mm=600, distance=84, reverse=False):
#         self.width = width
#         self.steps = steps
#         self.lines_per_mm = lines_per_mm
#         self.distance = distance
#         self.reverse = -1 if reverse else 1

#     def compute_shift_vector(self):
#         d = 1 / self.lines_per_mm  # mm per line
#         theta_min = np.arctan((0 - self.width / 2) / self.distance)
#         theta_max = np.arctan((self.width / 2) / self.distance)
#         lambda_min = d * np.sin(theta_min)
#         lambda_max = d * np.sin(theta_max)
#         # return torch.linspace(lambda_min, lambda_max, self.steps) * self.reverse * 0.7 - 50 * 1e-6
#         # 0.8 is equivalent to the 1.25 for factor_d
#         return torch.linspace(lambda_min, lambda_max, self.steps) * self.reverse * 0.8 - 50 * 1e-6

class ShiftCalculator:
    def __init__(self, width=8, steps=346, lines_per_mm=600, distance=84, reverse=False, adjustment_factor=.8, adjust_at='distance'):
        self.width = width  # Width of the sensor in mm
        self.steps = steps  # Number of discrete steps to calculate across the sensor
        self.lines_per_mm = lines_per_mm  # Density of the grating lines per mm
        self.original_distance = distance  # Store the original distance for reporting
        self.distance = distance  # Distance from the grating to the sensor in mm
        self.reverse = -1 if reverse else 1  # Reverse the direction of the wavelength range if needed
        self.adjustment_factor = adjustment_factor  # Adjustment factor for scaling
        self.adjust_at = adjust_at  # Determines whether the adjustment is at 'distance' or 'final'

        # If adjustment is to be made at the distance, modify the distance and print the values
        if self.adjust_at == 'distance':
            print(f"Original Distance: {self.distance} mm")
            self.distance = self.distance / self.adjustment_factor  # Apply adjustment by dividing the distance
            print(f"Adjusted Distance: {self.distance} mm")

    def compute_shift_vector(self):
        d = 1 / self.lines_per_mm  # Calculate grating spacing in mm
        theta_min = np.arctan((0 - self.width / 2) / self.distance)
        theta_max = np.arctan((self.width / 2) / self.distance)
        lambda_min = d * np.sin(theta_min)  # Minimum wavelength at theta_min
        lambda_max = d * np.sin(theta_max)  # Maximum wavelength at theta_max
        
        # Print the original min and max wavelengths if adjustment is at the final step
        if self.adjust_at == 'final':
            print(f"Original Min Wavelength: {lambda_min:e} m, Original Max Wavelength: {lambda_max:e} m")
        
        wavelengths = torch.linspace(lambda_min, lambda_max, self.steps) * self.reverse
        
        # Apply the adjustment at the final step and print the adjusted wavelengths
        if self.adjust_at == 'final':
            adjusted_wavelengths = wavelengths * self.adjustment_factor
            print(f"Adjusted Min Wavelength: {adjusted_wavelengths.min():e} m, Adjusted Max Wavelength: {adjusted_wavelengths.max():e} m")
            wavelengths = adjusted_wavelengths
        
        # Apply a constant offset to the wavelengths
        return wavelengths - 50 * 1e-6  # Subtract 50 microns for calibration or other corrections


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

