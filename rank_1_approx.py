import numpy as np
import matplotlib.pyplot as plt

# Initialize a tensor of shape 260x346x100
H, W, D = 260, 346, 100
tensor = np.random.rand(H, W, D)  # Randomly generated tensor for demonstration

# Function to perform rank-1 approximation
def rank_1_approximation(tensor):
    approx_tensor = np.zeros_like(tensor)
    
    for d in range(tensor.shape[2]):
        matrix = tensor[:, :, d]
        U, S, VT = np.linalg.svd(matrix, full_matrices=False)
        rank_1_matrix = S[0] * np.outer(U[:, 0], VT[0, :])
        approx_tensor[:, :, d] = rank_1_matrix
    
    return approx_tensor

# Perform the rank-1 approximation
approximated_tensor = rank_1_approximation(tensor)

# Function to plot a selection of slices
def plot_slices(tensor, approx_tensor, indices):
    fig, axes = plt.subplots(nrows=len(indices), ncols=2, figsize=(10, 5 * len(indices)))
    
    for i, index in enumerate(indices):
        ax1, ax2 = axes[i]
        
        # Plot original tensor slice
        ax1.imshow(tensor[:, :, index], cmap='gray')
        ax1.set_title(f'Original Slice {index}')
        ax1.axis('off')
        
        # Plot approximated tensor slice
        ax2.imshow(approx_tensor[:, :, index], cmap='gray')
        ax2.set_title(f'Rank-1 Approximation Slice {index}')
        ax2.axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("rank_1_approx.png")

# Select slices to plot
indices_to_plot = [0, 1, 2, 3, 4]  # Modify as needed to show different or more slices
plot_slices(tensor, approximated_tensor, indices_to_plot)

