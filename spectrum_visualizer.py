#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
# import convert_to_py

class SpectrumVisualizer:
    def __init__(self, cie_path):
        # Adjusted to explicitly convert numbers, removing any trailing commas or other issues
        self.cie_data = pd.read_csv(cie_path, delim_whitespace=True, header=None, 
                                    converters={1: lambda x: float(x.rstrip(',')), 
                                                2: lambda x: float(x.rstrip(',')), 
                                                3: lambda x: float(x.rstrip(','))},
                                    names=['wavelength', 'X', 'Y', 'Z'])
        self.wavelengths = torch.linspace(360, 830, steps=len(self.cie_data))  # 360nm to 830nm
        self.cie_data_tensor = torch.tensor(self.cie_data[['X', 'Y', 'Z']].values.astype(np.float32))


    def interpolate_spectrum(self, spectrum, input_wavelengths):
        """
        Interpolates the given spectrum to match the CIE 1931 wavelength range and resolution.
        Assumes the spectrum tensor shape is HxWxSpectrumDepth.
        """
        interpolated_spectrum = torch.zeros(spectrum.shape[:-1] + (len(self.wavelengths),))
        for i in range(spectrum.shape[0]):
            for j in range(spectrum.shape[1]):
                # Interpolating each spectral curve individually
                interp_func = interp1d(input_wavelengths, spectrum[i, j].cpu().numpy(), kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated_spectrum[i, j] = torch.tensor(interp_func(self.wavelengths.numpy()))
        return interpolated_spectrum

    def spectrum_to_xyz(self, spectrum):
        """
        Converts the given spectrum tensor to an XYZ tensor.
        """
        Y_norm = torch.sum(self.cie_data_tensor[:, 1])
        xyz = torch.tensordot(spectrum, self.cie_data_tensor/Y_norm, dims=([-1], [0]))
        return xyz

    def xyz_to_rgb(self, xyz):
        """
        Converts an XYZ tensor to an RGB tensor using the sRGB color space.
        """
        M = torch.tensor([[3.2406, -1.5372, -0.4986], 
                          [-0.9689, 1.8758, 0.0415], 
                          [0.0557, -0.2040, 1.0570]])
        rgb = torch.matmul(xyz, M.T)
        # rgb = torch.clamp(rgb, 0, 1)  # Clamp to [0, 1] to deal with out-of-gamut colors
        rgb = (rgb - rgb.min())/(rgb.max() - rgb.min())
        return rgb

    def visualize_rgb(self, rgb):
        """
        Visualizes the given RGB tensor as an image.
        """
        plt.imshow(rgb.numpy())
        plt.axis('off')
        plt.show()
        
    def visualize_and_save(self, input_tensor, input_wavelengths, output_file='output_image.png'):
        # Interpolate the input tensor to match CIE data wavelengths, if necessary
        if len(input_wavelengths) != len(self.wavelengths):
            input_tensor = self.interpolate_spectrum(input_tensor, input_wavelengths)

        # Convert to XYZ, then to RGB
        xyz = self.spectrum_to_xyz(input_tensor)
        rgb = self.xyz_to_rgb(xyz)

        # Visualization and saving
        # for screen
        # plt.figure(figsize=(20, 3.6))
        # for DVXplorer MINI 640x480
        # plt.figure(figsize=(4.8, 6.4))
        # for Davis346 346x260 (input tensor shape 260x346)
        plt.figure(figsize=(3.5, 2.6))
        plt.imshow(rgb.numpy(), aspect='auto')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to avoid displaying it in Jupyter notebooks or Python scripts
        
        return rgb

if __name__ == "__main__":
    visualizer = SpectrumVisualizer('ciexyz31_1.txt')

    # Example spectral data
    H, W, SpectrumDepth = 100, 100, 100  # Example dimensions
    input_wavelengths = torch.linspace(300, 900, steps=SpectrumDepth)  # Example input wavelengths
    example_spectrum = torch.rand(H, W, SpectrumDepth)  # Random spectral data for demonstration

    # Interpolate to match the CIE 1931 wavelength range and resolution
    interpolated_spectrum = visualizer.interpolate_spectrum(example_spectrum, input_wavelengths)

    # Convert to XYZ then to RGB
    xyz = visualizer.spectrum_to_xyz(interpolated_spectrum)
    rgb = visualizer.xyz_to_rgb(xyz)

    # Visualize the resulting RGB image
    visualizer.visualize_rgb(rgb)
    
#     visualizer = SpectrumVisualizer('ciexyz31_1.txt')
#     H, W, SpectrumDepth = 100, 100, 471  # Example dimensions
#     example_spectrum = torch.rand(H, W, SpectrumDepth)  # Example random spectral data
#     input_wavelengths = torch.linspace(360, 830, steps=SpectrumDepth)  # Matching the CIE data range
    
    rgb_image = visualizer.visualize_and_save(example_spectrum, input_wavelengths, 'spectral_rgb_visualization.png')


# In[13]:


# !cat ciexyz31_1.txt | tail


# In[14]:


# get_ipython().system('jupyter nbconvert --to script spectrum_visualizer.ipynb')


# In[ ]:




