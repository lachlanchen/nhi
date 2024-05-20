import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import os

class GaussianBlur:
    def __init__(self, folder_path, kernel_size=5, sigma=3.0, chunk_size=1000):
        self.folder_path = Path(folder_path)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.chunk_size = chunk_size

    def create_gaussian_kernel(self):
        x = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        y = x.view(-1, 1)
        x = x.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y = y.repeat(1, self.kernel_size)
        kernel = torch.exp(-0.5 * (x**2 + y**2) / self.sigma**2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).to(torch.float16)

    def apply_gaussian_blur(self, tensor):
        num_frames, height, width = tensor.shape
        kernel = self.create_gaussian_kernel()

        # Ensure tensor is of type float16
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)

        # Apply Gaussian blur in chunks
        blurred_tensors = []
        for i in tqdm(range(0, num_frames, self.chunk_size), desc="Applying Gaussian Blur"):
            end_idx = min(i + self.chunk_size, num_frames)
            chunk = tensor[i:end_idx].unsqueeze(1)  # Add channel dimension
            blurred_chunk = F.conv2d(chunk, kernel, padding=self.kernel_size // 2)
            blurred_tensors.append(blurred_chunk.squeeze(1))  # Remove channel dimension

        return blurred_tensors

    def process_file(self, file_name):
        file_path = self.folder_path / file_name
        tensor = torch.load(file_path)
        blurred_tensors = self.apply_gaussian_blur(tensor)
        blurred_folder = self.folder_path / "blurred_franmes"
        os.makedirs(blurred_folder, exist_ok=True)
        # Save the blurred tensor in segments
        for i, blurred_tensor in enumerate(blurred_tensors):
            
            blurred_file_path =  blurred_folder / f"blurred_{file_name}_part_{i}.pt"

            torch.save(blurred_tensor, blurred_file_path)
            print(f"Blurred tensor segment saved to {blurred_file_path}")

    def run(self):
        self.process_file('frames_260_346.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply Gaussian blur to tensor files.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing tensor files.')
    args = parser.parse_args()

    gaussian_blur = GaussianBlur(args.folder_path)
    gaussian_blur.run()

