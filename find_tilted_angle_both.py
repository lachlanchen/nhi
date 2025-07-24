import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from integer_shifter import TensorShifter  # Ensure integer_shifter.py is updated and in the same directory or correctly added to PYTHONPATH
from tqdm import tqdm

def calculate_std_for_shifts(tensor, min_shift, max_shift, reverse, sample_rate, axis='width'):
    std_values = []
    shift_values = list(range(min_shift, max_shift + 1))

    # Apply sampling rate
    sampled_tensor = tensor[:, ::sample_rate, ::sample_rate]

    if axis == 'width':
        length = sampled_tensor.shape[2]  # Width
        for shift in tqdm(shift_values, desc=f"Shifting along {axis} axis"):
            # Create shift vector
            shift_vector = [shift] * length

            # Instantiate the TensorShifter with the shift vector and axis
            tensor_shifter = TensorShifter(shift_vector, axis=axis, reverse=reverse)

            # Apply the shift
            shifted_tensor = tensor_shifter.apply_shift(sampled_tensor)

            # Calculate the standard deviation for each time slice and sum them
            std_val = torch.sum(torch.std(shifted_tensor, dim=[1, 2])).item()
            std_values.append(std_val)

    elif axis == 'height':
        length = sampled_tensor.shape[1]  # Height
        for shift in tqdm(shift_values, desc=f"Shifting along {axis} axis"):
            # Create shift vector
            shift_vector = [shift] * length

            # Instantiate the TensorShifter with the shift vector and axis
            tensor_shifter = TensorShifter(shift_vector, axis=axis, reverse=reverse)

            # Apply the shift
            shifted_tensor = tensor_shifter.apply_shift(sampled_tensor)

            # Calculate the standard deviation for each time slice and sum them
            std_val = torch.sum(torch.std(shifted_tensor, dim=[1, 2])).item()
            std_values.append(std_val)

    elif axis == 'both':
        height = sampled_tensor.shape[1]
        width = sampled_tensor.shape[2]
        for shift in tqdm(shift_values, desc=f"Shifting along both axes"):
            # Create shift matrix
            shift_matrix = torch.full((height, width), shift)

            # Instantiate the TensorShifter with the shift matrix and axis
            tensor_shifter = TensorShifter(shift_matrix, axis=axis, reverse=reverse)

            # Apply the shift
            shifted_tensor = tensor_shifter.apply_shift(sampled_tensor)

            # Calculate the standard deviation for each time slice and sum them
            std_val = torch.sum(torch.std(shifted_tensor, dim=[1, 2])).item()
            std_values.append(std_val)
    else:
        raise ValueError("Axis must be 'width', 'height', or 'both'")

    return shift_values, std_values

def plot_std_vs_shift(shift_values, std_values, output_path, axis='width'):
    plt.figure(figsize=(10, 5))
    plt.plot(shift_values, std_values, marker='o', linestyle='-')

    # Find and annotate the minimum std value
    min_std = min(std_values)
    min_shift = shift_values[std_values.index(min_std)]
    plt.scatter(min_shift, min_std, color='red')  # Mark the min point
    plt.text(min_shift, min_std, f'Min Std: {min_std:.2f} at Shift: {min_shift}', fontsize=12, color='red')

    # Add a vertical red dotted line at the minimum std value
    plt.axvline(x=min_shift, color='red', linestyle='--')

    plt.xlabel('Shift Value')
    plt.ylabel('Sum of Standard Deviations of Shifted Frames')
    plt.title(f'Sum of Standard Deviations vs. Shift Value along {axis} axis')
    plt.grid(True)
    plt.savefig(output_path)  # Save the figure to the specified output path
    plt.close()  # Close the plot to free up memory
    print(f"Figure saved to {output_path}")
    print(f"Minimum standard deviation: {min_std:.2f} occurs at shift: {min_shift}")

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot the sum of standard deviations of shifted tensor frames to find the optimal shift value.')
    parser.add_argument('tensor_path', type=str, help='Path to the tensor file to be shifted.')
    parser.add_argument('--min_shift', type=int, default=-900, help='Minimum shift value to test.')
    parser.add_argument('--max_shift', type=int, default=-600, help='Maximum shift value to test.')
    parser.add_argument('--reverse', action='store_true', help='Whether to reverse the shift direction.')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sampling rate to downsample the tensor before shifting.')
    parser.add_argument('--axis', type=str, choices=['width', 'height', 'both'], default='width', help='Axis along which to apply the shift.')

    args = parser.parse_args()

    # Load the tensor from the specified file
    tensor = torch.load(args.tensor_path)

    # Calculate std for each shift
    shift_values, std_values = calculate_std_for_shifts(
        tensor,
        args.min_shift,
        args.max_shift,
        args.reverse,
        args.sample_rate,
        axis=args.axis
    )

    # Define the output path for the figure
    dir_name = os.path.dirname(args.tensor_path)
    reverse_str = 'reversed' if args.reverse else 'normal'
    figure_filename = f"std_plot_{args.axis}_{args.min_shift}_to_{args.max_shift}_{reverse_str}_sample{args.sample_rate}.png"
    output_path = os.path.join(dir_name, figure_filename)

    # Plot the results and save the figure
    plot_std_vs_shift(shift_values, std_values, output_path, axis=args.axis)

if __name__ == "__main__":
    main()
