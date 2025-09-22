import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_tensor(tensor_path):
    tensor = torch.load(tensor_path)
    output_dir = os.path.join(os.path.dirname(tensor_path), f"{os.path.splitext(os.path.basename(tensor_path))[0]}_diagnosis")
    os.makedirs(output_dir, exist_ok=True)

    min_values = tensor.min(dim=1).values.min(dim=1).values.numpy()
    max_values = tensor.max(dim=1).values.max(dim=1).values.numpy()
    median_values = tensor.median(dim=1).values.median(dim=1).values.numpy()
    mean_values = tensor.mean(dim=(1, 2)).numpy()
    neg_count = (tensor < 0).sum(dim=(1, 2)).numpy()
    zero_count = (tensor == 0).sum(dim=(1, 2)).numpy()
    pos_count = (tensor > 0).sum(dim=(1, 2)).numpy()

    diagnosis_data = {
        'Frame': np.arange(tensor.shape[0]),
        'Min': min_values,
        'Max': max_values,
        'Median': median_values,
        'Mean': mean_values,
        'Negative Count': neg_count,
        'Zero Count': zero_count,
        'Positive Count': pos_count
    }

    diagnosis_df = pd.DataFrame(diagnosis_data)
    diagnosis_csv_path = os.path.join(output_dir, 'tensor_diagnosis.csv')
    diagnosis_df.to_csv(diagnosis_csv_path, index=False)
    print(f"Saved diagnosis data to {diagnosis_csv_path}")

    fig, axs = plt.subplots(7, 1, figsize=(15, 20), sharex=True)

    axs[0].plot(diagnosis_df['Frame'], diagnosis_df['Min'], label='Min', color='purple')
    axs[0].set_ylabel('Min')
    axs[0].grid(True)

    axs[1].plot(diagnosis_df['Frame'], diagnosis_df['Max'], label='Max', color='red')
    axs[1].set_ylabel('Max')
    axs[1].grid(True)

    axs[2].plot(diagnosis_df['Frame'], diagnosis_df['Mean'], label='Mean', color='green')
    axs[2].set_ylabel('Mean')
    axs[2].grid(True)

    axs[3].plot(diagnosis_df['Frame'], diagnosis_df['Median'], label='Median', color='blue')
    axs[3].set_ylabel('Median')
    axs[3].grid(True)

    axs[4].plot(diagnosis_df['Frame'], diagnosis_df['Negative Count'], label='Negative Count', color='orange')
    axs[4].set_ylabel('Negative Count')
    axs[4].grid(True)

    axs[5].plot(diagnosis_df['Frame'], diagnosis_df['Zero Count'], label='Zero Count', color='brown')
    axs[5].set_ylabel('Zero Count')
    axs[5].grid(True)

    axs[6].plot(diagnosis_df['Frame'], diagnosis_df['Positive Count'], label='Positive Count', color='pink')
    axs[6].set_ylabel('Positive Count')
    axs[6].set_xlabel('Frame')
    axs[6].grid(True)

    plt.suptitle('Tensor Diagnosis')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diagnosis_plot_path = os.path.join(output_dir, 'diagnosis_plot.png')
    plt.savefig(diagnosis_plot_path)
    plt.close()
    print(f"Saved diagnosis plot to {diagnosis_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose a tensor and save statistics.')
    parser.add_argument('tensor_path', type=str, help='Path to the tensor file.')

    args = parser.parse_args()
    
    diagnose_tensor(args.tensor_path)
