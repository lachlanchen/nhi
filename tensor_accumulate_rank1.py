import torch
import os
import argparse
from tqdm import tqdm

class TensorAccumulator:
    def __init__(self, tensor_path):
        self.tensor = torch.load(tensor_path)
        self.device = self.tensor.device

    # def compute_and_save_rank1(self, save_path):
    #     length, height, width = self.tensor.size()
    #     rank1_tensor = torch.zeros((length, height, width), dtype=self.tensor.dtype, device=self.device)
        
    #     for index in range(length):
    #         U, S, V = torch.linalg.svd(self.tensor[index], full_matrices=False)
    #         rank1_approx = S[0] * torch.outer(U[:, 0], V[0, :])
    #         rank1_tensor[index] = rank1_approx

    #     torch.save(rank1_tensor, save_path)
    #     print(f'Rank-1 tensor saved at {save_path}')
    #     return rank1_tensor

    def compute_and_save_rank1(self, save_path):
        length, height, width = self.tensor.size()
        rank1_tensor = torch.zeros((length, height, width), dtype=torch.float32, device=self.device)
        self.tensor = self.tensor.to(dtype=torch.float32)  # Ensure the tensor is floating point
        
        for index in tqdm(range(length)):
            U, S, V = torch.linalg.svd(self.tensor[index], full_matrices=False)
            rank1_approx = S[0] * torch.outer(U[:, 0], V[0, :])
            rank1_tensor[index] = rank1_approx

        torch.save(rank1_tensor, save_path)
        print(f'Rank-1 tensor saved at {save_path}')
        return rank1_tensor


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

def main(args):
    base_dir = os.path.dirname(args.file_path)
    rank1_dir = os.path.join(base_dir, 'rank1')
    normal_dir = os.path.join(base_dir, 'normal')
    
    accumulator = TensorAccumulator(args.file_path)
    
    if args.rank1:
        rank1_path = args.file_path.replace('.pt', '_rank1.pt')
        if not os.path.exists(rank1_path) or not args.cache:
            rank1_tensor = accumulator.compute_and_save_rank1(rank1_path)
        else:
            rank1_tensor = torch.load(rank1_path)
            print(f'Using cached rank-1 tensor from {rank1_path}')
        accumulator.tensor = rank1_tensor
        target_dir = rank1_dir
    else:
        target_dir = normal_dir

    os.makedirs(target_dir, exist_ok=True)

    # Continuous Accumulation
    suffix = 'rank1' if args.rank1 else 'normal'
    continuous_path = os.path.join(target_dir, f'continuous_accumulation_{suffix}_{args.intervals}.pt')
    continuous_results = accumulator.accumulate_continuous(args.intervals)
    torch.save(continuous_results, continuous_path)
    print(f'Continuous accumulation saved at {continuous_path}')

    # Interval Accumulation
    interval_path = os.path.join(target_dir, f'interval_accumulation_{suffix}_{args.intervals}.pt')
    interval_results = accumulator.accumulate_interval(args.intervals)
    torch.save(interval_results, interval_path)
    print(f'Interval accumulation saved at {interval_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensor Accumulation with Optional Rank-1 Approximation and Caching')
    parser.add_argument('file_path', type=str, help='Path to the tensor file')
    parser.add_argument('--intervals', type=int, default=1000, help='Number of intervals for accumulation')
    parser.add_argument('-r1', '--rank1', action='store_true', help='Enable rank-1 approximation')
    parser.add_argument('--cache', action='store_true', default=True, help='Use cached rank-1 tensor if available')
    
    args = parser.parse_args()
    main(args)
