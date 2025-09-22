import torch
import argparse
import os

class TensorShifter:
    def __init__(self, max_shift, width, reverse=False, crop=False):
        self.max_shift = max_shift  # Maximum time shift for the first or last column
        self.width = width  # Width of the tensor to calculate shift per column
        self.reverse = reverse  # Determine if the shift starts from the last column
        self.crop = crop

    def apply_shift(self, tensor):
        # Initialize a tensor of the same shape filled with zeros
        shifted_tensor = torch.zeros_like(tensor)

        # Calculate and apply the time shift for each column
        for col in range(self.width):
            # Determine the shift amount based on column index
            if not self.reverse:
                shift = int(self.max_shift * (self.width - col - 1) / self.width)  # Maximum shift at the first column
            else:
                shift = int(self.max_shift * col / self.width)  # Maximum shift at the last column

            # Apply the time shift to the entire column
            if shift > 0:
                shifted_tensor[shift:, :, col] = tensor[:-shift, :, col]  # Shift forward in time
            elif shift < 0:
                shifted_tensor[:shift, :, col] = tensor[-shift:, :, col]  # Shift backward in time
            else:
                shifted_tensor[:, :, col] = tensor[:, :, col]

        if self.crop:
            if self.max_shift <= 0:
                return shifted_tensor[:self.max_shift]
            else:
                return shifted_tensor[self.max_shift:]
        else:
            return shifted_tensor


def main(tensor_path, max_shift, reverse, sample_rate):
    # Load the tensor from the specified file
    tensor = torch.load(tensor_path)

    # Downsample the tensor
    tensor = tensor[:, ::sample_rate, ::sample_rate]

    # Get the width of the tensor from the last dimension
    width = tensor.shape[2]

    # Instantiate the TensorShifter
    tensor_shifter = TensorShifter(max_shift, width, reverse)

    # Apply the shift
    shifted_tensor = tensor_shifter.apply_shift(tensor)

    # Define output path with shift and possibly reverse indicated in the filename
    dir_name, file_name = os.path.split(tensor_path)
    new_file_name = f"{file_name.split('.')[0]}_shifted{max_shift}_{'reversed' if reverse else 'normal'}_sample{sample_rate}.pt"
    output_path = os.path.join(dir_name, new_file_name)

    # Save the shifted tensor
    torch.save(shifted_tensor, output_path)
    print(f"Shifted tensor saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shift tensor columns along the time dimension based on their horizontal position, with optional reversing and downsampling.')
    parser.add_argument('tensor_path', type=str, help='Path to the tensor file to be shifted.')
    parser.add_argument('max_shift', type=int, help='Maximum shift value for the column at the edge.')
    parser.add_argument('--reverse', action='store_true', help='Reverse the shift direction, shifting the first column maximally instead of the last.')
    parser.add_argument('--sample_rate', type=int, default=10, help='Factor by which to downsample the tensor dimensions.')

    args = parser.parse_args()

    main(args.tensor_path, args.max_shift, args.reverse, args.sample_rate)
