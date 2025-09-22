import torch
import sys

def print_tensor_shape(file_path):
    # Load the tensor from the specified file
    tensor = torch.load(file_path)
    
    # Print the shape of the tensor
    print("Shape of the tensor:", tensor.shape)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pt_shape.py <path_to_tensor_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    print_tensor_shape(file_name)

