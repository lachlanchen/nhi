import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tensor_accumulation import TensorAccumulator  # Ensure this module has all necessary imports and classes
from tensor_shifter import ShiftCalculator, TensorShifter  # Ensure these classes are defined as shown earlier
import shutil

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

    def load_and_accumulate(self):
        accumulator = TensorAccumulator(self.file_path)
        accumulated_tensor = accumulator.accumulate_continuous(self.intervals)
        return accumulated_tensor

    def shift_and_visualize(self, tensor):
        shifted_tensor = self.tensor_shifter.apply_shift(tensor)
        self.visualize(tensor[0], shifted_tensor[0], 'shift')  # Visualize the first frame before and after shift
        return shifted_tensor

    def visualize(self, original, processed, description):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original.numpy(), cmap='hot')
        plt.title('Original')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(processed.numpy(), cmap='hot')
        plt.title('Processed - ' + description)
        plt.colorbar()
        plt.suptitle(f'Visualization of Changes: {description}')
        plt.savefig(os.path.join(self.output_folder, f'{description}_comparison.png'))
        plt.close()

    def process(self):
        print("Starting processing of tensor data...")
        tensor = self.load_and_accumulate()
        tensor = tensor.permute(1, 2, 0)  # Ensure correct dimension order for processing
        tensor = self.shift_and_visualize(tensor)
        # Additional steps like smoothing or further processing would go here, with visualization at each step
        print("Processing complete. Output saved in:", self.output_folder)

def main():
    parser = argparse.ArgumentParser(description='Process tensor data for spectrometer analysis.')
    parser.add_argument('file_path', type=str, help='Path to the tensor file.')
    args = parser.parse_args()
    spectrometer = HorographySpectrometer(args.file_path)
    spectrometer.process()

if __name__ == "__main__":
    main()
