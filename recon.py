import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from spectrum_visualizer import SpectrumVisualizer
import os

class GratingLightDispersionModel(nn.Module):
    def __init__(self, file_path, lambda_start=370, lambda_end=790, lambda_step=10, d_grating=1/(600*1e3), sensor_offset=0.05, device='cpu'):
        super(GratingLightDispersionModel, self).__init__()
        self.device = device
        self.d_grating = d_grating
        self.sensor_offset = sensor_offset
        self.lambda_step = lambda_step

        # Load y_positions from CSV
        df = pd.read_csv(file_path)
        self.y_positions = torch.tensor(df['axis_y_position_mm'].values / 1000, dtype=torch.float32, device=device)  # Convert to meters

        # SPD wavelengths and light SPD
        self.wavelengths = torch.arange(lambda_start, lambda_end + 1, lambda_step, device=device, dtype=torch.float32)
        self.light_spd = nn.Parameter(torch.rand(len(self.wavelengths), device=device))

        self.visualizer = SpectrumVisualizer('ciexyz31_1.txt')

        self.H, self.W = 640, 480

    def forward(self):
        L_grating_to_LED = 0.1  # Distance from grating to LED in meters
        L_grating_to_sensor = 0.1  # Distance from grating to sensor in meters
        theta_inc = torch.atan((self.sensor_offset - self.y_positions) / L_grating_to_LED)

        # Screen positions in meters
        sensor_width_m = 4.32e-3  # Adjusted for sensor dimensions
        x_sensor = torch.linspace(-sensor_width_m / 2, sensor_width_m / 2, steps=480, device=self.device) + self.sensor_offset
        theta_diff = torch.atan(x_sensor / L_grating_to_sensor)

        # Calculate the wavelengths for each position based on diffraction
        lambda_diff_nm = self.d_grating * (torch.sin(theta_inc).unsqueeze(-1) + torch.sin(theta_diff)) * 1e9

        # Calculate the difference between each sensor wavelength and SPD wavelengths
        diff = torch.abs(lambda_diff_nm.unsqueeze(-1) - self.wavelengths.unsqueeze(0).unsqueeze(0))
        soft_onehot = F.softmax(-diff * 0.5, dim=-1)  # Apply softmax to determine weights

        print("soft_onehot.shapeL ", soft_onehot.shape)
        print("self.light_spd.shape: ", self.light_spd.shape)

        spectrum = soft_onehot * self.light_spd[None, None, :]

        print("spectrum.shape: ", spectrum.shape)

        # spectrum = spectrum.unsqueeze(1).repeat(1, 2, 1, 1)
        spectrum = spectrum.unsqueeze(1).expand(-1, self.H, -1, -1)


        print("spectrum.shape: ", spectrum.shape)

        # Simulate light intensity distribution based on SPD
        # light_intensity = torch.einsum('nwh,w->nwh', soft_onehot, self.light_spd)

        # Keep light_intensity for visualization or further processing
        # return light_intensity

        return spectrum

    def visualize_output(self, output_tensor, file_name_pattern='dispersed_frames/dispersed_light_{:04d}.png'):
        os.makedirs("dispersed_frames", exist_ok=True)
        for i, frame in enumerate(output_tensor):
            # The visualize_and_save function expects a 3D tensor (C, H, W)
            # Adjust the dimension of frame for visualization
            frame_rgb = self.visualizer.visualize_and_save(frame.detach(), self.wavelengths.cpu(), file_name_pattern.format(i))

if __name__ == "__main__":
    file_path = 'unique_timestamps_and_y_positions.csv'  # Adjust path as needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GratingLightDispersionModel(file_path, device=device)
    dispersed_light = model.forward()
    model.visualize_output(dispersed_light)
