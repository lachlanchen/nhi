import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

class SpectrumVisualizer:
    def __init__(self, cie_path):
        self.cie_data = pd.read_csv(cie_path, delim_whitespace=True, header=None, 
                                    converters={1: lambda x: float(x.rstrip(',')), 
                                                2: lambda x: float(x.rstrip(',')), 
                                                3: lambda x: float(x.rstrip(','))},
                                    names=['wavelength', 'X', 'Y', 'Z'])
        self.wavelengths = torch.linspace(360, 830, steps=len(self.cie_data))
        self.cie_data_tensor = torch.tensor(self.cie_data[['X', 'Y', 'Z']].values.astype(np.float32))

    def interpolate_spectrum(self, spectrum, input_wavelengths):
        interpolated_spectrum = torch.zeros(spectrum.shape[:-1] + (len(self.wavelengths),))
        for i in range(spectrum.shape[0]):
            for j in range(spectrum.shape[1]):
                interp_func = interp1d(input_wavelengths, spectrum[i, j].cpu().numpy(), kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated_spectrum[i, j] = torch.tensor(interp_func(self.wavelengths.numpy()))
        return interpolated_spectrum

    def spectrum_to_xyz(self, spectrum):
        Y_norm = torch.sum(self.cie_data_tensor[:, 1])
        xyz = torch.tensordot(spectrum, self.cie_data_tensor / Y_norm, dims=([-1], [0]))
        return xyz

    def xyz_to_rgb_or_rgba(self, xyz, include_alpha=True):
        M = torch.tensor([[3.2406, -1.5372, -0.4986], 
                          [-0.9689, 1.8758, 0.0415], 
                          [0.0557, -0.2040, 1.0570]])
        rgb = torch.matmul(xyz, M.T)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0, 1]

        if include_alpha:
            intensity = rgb.mean(dim=-1, keepdim=True)  # Average intensity as alpha value
            alpha = intensity  # More intensity implies less transparency
            # alpha = 1 - intensity
            rgba = torch.cat((rgb, alpha), dim=-1)
            return rgba
        return rgb

    def visualize_and_save(self, input_tensor, input_wavelengths, output_file='output_image.png', rgba=False):
        if len(input_wavelengths) != len(self.wavelengths):
            input_tensor = self.interpolate_spectrum(input_tensor, input_wavelengths)

        xyz = self.spectrum_to_xyz(input_tensor)
        output = self.xyz_to_rgb_or_rgba(xyz, include_alpha=rgba)

        plt.figure(figsize=(3.5*2, 2.6*2), dpi=100)
        plt.imshow(output.numpy(), aspect='auto')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, format='png')
        plt.close()

        return output

if __name__ == "__main__":
    visualizer = SpectrumVisualizer('ciexyz31_1.txt')
    H, W, SpectrumDepth = 100, 100, 100
    input_wavelengths = torch.linspace(300, 900, steps=SpectrumDepth)
    example_spectrum = torch.rand(H, W, SpectrumDepth)

    rgba_image = visualizer.visualize_and_save(example_spectrum, input_wavelengths, 'spectral_rgba_visualization.png', rgba=True)
