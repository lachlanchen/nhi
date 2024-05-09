import torch
import os
import argparse

class TensorAccumulator:
    def __init__(self, tensor_path):
        self.tensor = torch.load(tensor_path)
        self.device = self.tensor.device

    def accumulate_continuous(self, intervals):
        length, height, width = self.tensor.size()
        interval_length = length // intervals
        interval_results = torch.zeros((intervals, height, width), dtype=self.tensor.dtype, device=self.device)
        cumulative_sum = torch.zeros((height, width), dtype=self.tensor.dtype, device=self.device)

        for index in range(length):
            cumulative_sum += self.tensor[index]
            if (index + 1) % interval_length == 0 or index == length - 1:
                interval_idx = (index + 1) // interval_length - 1
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
    parser.add_argument('file_path', type=str, help='Path to the tensor file')
    parser.add_argument('--intervals', type=int, default=1000, help='Number of intervals to split the tensor into')
    args = parser.parse_args()

    file_path = args.file_path
    intervals = args.intervals
    file_dir = os.path.dirname(file_path)
    
    accumulator = TensorAccumulator(file_path)
    continuous_accumulation = accumulator.accumulate_continuous(intervals)
    continuous_path = os.path.join(file_dir, 'continuous')
    os.makedirs(continuous_path, exist_ok=True)
    torch.save(continuous_accumulation, os.path.join(continuous_path, f'continuous_accumulation_{intervals}.pt'))
    print('Continuous accumulation saved:', os.path.join(continuous_path, f'continuous_accumulation_{intervals}.pt'))

    interval_accumulation = accumulator.accumulate_interval(intervals)
    interval_path = os.path.join(file_dir, 'interval')
    os.makedirs(interval_path, exist_ok=True)
    torch.save(interval_accumulation, os.path.join(interval_path, f'interval_accumulation_{intervals}.pt'))
    print('Interval accumulation saved:', os.path.join(interval_path, f'interval_accumulation_{intervals}.pt'))

if __name__ == "__main__":
    main()
