import torch
import os
import argparse
import torch.nn.functional as F

class TensorAccumulator:
    def __init__(self, folder_path, n_sigma=1.0):
        self.folder_path = folder_path
        self.n_sigma = n_sigma
        self.tensor_files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('blurred_frames_260_346.pt_part_') and f.endswith('.pt')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        self.summed_tensors = [self.sum_tensor_file(os.path.join(folder_path, f)) for f in self.tensor_files]
        self.tensor = torch.stack(self.summed_tensors, dim=0)
        self.device = self.tensor.device

        self.gaussian_blur()

    def gaussian_blur(self, kernel_size=5, sigma=1):
        # Create a Gaussian kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        radius = kernel_size // 2
        kernel_range = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        x = kernel_range.reshape(1, -1).repeat(kernel_size, 1)
        y = kernel_range.reshape(-1, 1).repeat(1, kernel_size)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize the kernel

        # Apply the Gaussian kernel
        padded_tensor = F.pad(self.tensor.float(), (radius, radius, radius, radius), mode='reflect')
        blurred_tensor = F.conv2d(padded_tensor.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=0)
        return blurred_tensor.squeeze(1)

    def sum_tensor_file(self, file_path):
        tensor = torch.load(file_path)
        return tensor.sum(dim=0)

    def rescale(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    def centralize(self, tensor):
        # Step 1: Rescale the whole tensor to [0, 1]
        tensor = self.rescale(tensor)
        # tensor = torch.exp(tensor)
        T, H, W = tensor.shape
        
        # Step 2: Current centralization
        tensor_float = tensor.float()
        mean_across_columns = tensor_float[:,:H//2].mean(dim=1, keepdim=True)
        # mean_across_columns,indices = tensor_float[:,:H//2].median(dim=1, keepdim=True)
        centralized_tensor = (tensor_float - mean_across_columns).type(tensor.dtype)

        # Step 3: Create mask based on n_sigma
        std_across_columns = tensor_float[:,:H//2].std(dim=1, keepdim=True)
        mask = (torch.abs(centralized_tensor) <= self.n_sigma * std_across_columns).float()

        # Step 4: Adjust non-masked and masked regions
        non_masked_region = centralized_tensor * (1 - mask)
        # non_masked_min = non_masked_region.min()
        # adjusted_non_masked = non_masked_region - non_masked_min
        adjusted_non_masked = non_masked_region
        adjusted_masked = mask 

        # Combine adjusted regions
        adjusted_tensor = adjusted_non_masked * (1 - mask) + adjusted_masked * mask

        # Step 5: Rescale the tensor back to [0, 1]
        final_tensor = self.rescale(adjusted_tensor)

        return final_tensor

    def accumulate_continuous(self, intervals):
        length, height, width = self.tensor.size()
        interval_length = length // intervals
        interval_results = torch.zeros((intervals, height, width), dtype=self.tensor.dtype, device=self.device)
        cumulative_sum = torch.zeros((height, width), dtype=self.tensor.dtype, device=self.device)

        for index in range(length):
            cumulative_sum += self.tensor[index]
            if (index + 1) % interval_length == 0 or index == length - 1:
                interval_idx = min((index + 1) // interval_length, intervals - 1)
                interval_results[interval_idx] = cumulative_sum.clone()

        return interval_results

    def accumulate_interval(self, intervals):
        length, height, width = self.tensor.size()
        interval_length = length // intervals
        interval_results = torch.zeros((intervals, height, width), dtype=self.tensor.dtype, device=self.device)

        for i in range(intervals):
            start_index = i * interval_length
            end_index = min(start_index + interval_length, length)
            interval_results[i] = self.tensor[start_index:end_index].sum(dim=0)

        return interval_results

def main():
    parser = argparse.ArgumentParser(description='Accumulate tensor values over intervals.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing partitioned tensor files')
    parser.add_argument('--intervals', type=int, default=100, help='Number of intervals to split the tensor into')
    parser.add_argument('--centralize', action='store_true', help='Centralize the tensor by subtracting the mean across rows')
    parser.add_argument('--post', action='store_true', help='Apply centralization post accumulation')
    parser.add_argument('--n_sigma', type=float, default=1.0, help='Number of standard deviations for masking')
    args = parser.parse_args()

    folder_path = args.folder_path
    intervals = args.intervals
    centralize = args.centralize
    post_centralize = args.post
    n_sigma = args.n_sigma

    accumulator = TensorAccumulator(folder_path, n_sigma)
    
    if centralize and not post_centralize:
        accumulator.tensor = accumulator.centralize(accumulator.tensor)
    
    continuous_accumulation = accumulator.accumulate_continuous(intervals)
    if centralize and post_centralize:
        continuous_accumulation = accumulator.centralize(continuous_accumulation)
    
    continuous_path = folder_path
    os.makedirs(continuous_path, exist_ok=True)
    centralize_suffix = ""
    if centralize:
        centralize_suffix = "_post" if post_centralize else "_pre"
    continuous_filename = f'continuous_accumulation_{intervals}{centralize_suffix}.pt'
    torch.save(continuous_accumulation, os.path.join(continuous_path, continuous_filename))
    print('Continuous accumulation saved:', os.path.join(continuous_path, continuous_filename))

    interval_accumulation = accumulator.accumulate_interval(intervals)
    if centralize and post_centralize:
        interval_accumulation = accumulator.centralize(interval_accumulation)

    interval_path = folder_path
    os.makedirs(interval_path, exist_ok=True)
    interval_filename = f'interval_accumulation_{intervals}{centralize_suffix}.pt'
    torch.save(interval_accumulation, os.path.join(interval_path, interval_filename))
    print('Interval accumulation saved:', os.path.join(interval_path, interval_filename))

if __name__ == "__main__":
    main()
