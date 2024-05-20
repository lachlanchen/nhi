import torch
import os
import argparse

class TensorAccumulator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.tensor_files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('blurred_frames_260_346.pt_part_') and f.endswith('.pt')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        self.summed_tensors = [self.sum_tensor_file(os.path.join(folder_path, f)) for f in self.tensor_files]
        self.tensor = torch.stack(self.summed_tensors, dim=0)
        self.device = self.tensor.device

    def sum_tensor_file(self, file_path):
        tensor = torch.load(file_path)
        return tensor.sum(dim=0)

    def centralize(self, tensor):
        tensor_float = tensor.float()
        median_across_rows = tensor_float.mean(dim=1, keepdim=True)
        # median_across_rows, indices = tensor_float.median(dim=1, keepdim=True)
        centralized_tensor = (tensor_float - median_across_rows).type(tensor.dtype)
        centralized_tensor = torch.abs(centralized_tensor)
        # centralized_tensor = centralized_tensor / (centralized_tensor.max()+1e-6)
        return centralized_tensor

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
    parser.add_argument('--centralize', action='store_true', help='Centralize the tensor by subtracting the median across rows')
    parser.add_argument('--post', action='store_true', help='Apply centralization post accumulation')
    args = parser.parse_args()

    folder_path = args.folder_path
    intervals = args.intervals
    centralize = args.centralize
    post_centralize = args.post

    accumulator = TensorAccumulator(folder_path)
    
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

