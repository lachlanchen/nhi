import torch
import os
import argparse

class TensorSubtractor:
    def __init__(self, base_folder, output_folder='data_reference_subtraction', ref_folder=None):
        self.base_folder = base_folder
        self.output_folder = output_folder
        self.ref_folder = ref_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_tensor(self, path):
        return torch.load(path)

    def subtract_tensors(self, tensor_name):
        primary_path = os.path.join(self.base_folder, tensor_name)
        primary_tensor = self.load_tensor(primary_path)

        if self.ref_folder:
            reference_path = os.path.join(self.ref_folder, tensor_name)
            reference_tensor = self.load_tensor(reference_path)
            result_tensor = primary_tensor - reference_tensor
        else:
            result_tensor = primary_tensor

        return result_tensor

    def save_tensor(self, tensor, filename):
        output_path = os.path.join(self.output_folder, filename)
        torch.save(tensor, output_path)
        print(f'Tensor saved to {output_path}')

def main():
    parser = argparse.ArgumentParser(description="Subtract two sets of tensors (continuous and interval) and save the results.")
    parser.add_argument('base_folder', type=str, help="Base folder containing the tensor files.")
    parser.add_argument('-ref', '--ref_folder', type=str, help="Reference folder containing the tensor files to subtract.")
    parser.add_argument('-o', '--output_folder', type=str, default='data_reference_subtraction', help="Folder to save the subtracted tensors.")
    args = parser.parse_args()

    subtractor = TensorSubtractor(args.base_folder, args.output_folder, args.ref_folder)

    # Subtract continuous and interval tensors
    result_continuous = subtractor.subtract_tensors('continuous_accumulation_100.pt')
    subtractor.save_tensor(result_continuous, 'subtracted_continuous_100.pt')

    result_interval = subtractor.subtract_tensors('interval_accumulation_100.pt')
    subtractor.save_tensor(result_interval, 'subtracted_interval_100.pt')

if __name__ == "__main__":
    main()
