import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from spectrum_visualizer import SpectrumVisualizer
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class GratingLightDispersionModel(nn.Module):
    def __init__(self, file_path, lambda_start=370, lambda_end=790, lambda_step=10, d_grating=1/(600*1e3), sensor_offset=0.05, device='cpu'):
        super(GratingLightDispersionModel, self).__init__()
        self.device = device
        self.d_grating = d_grating
        self.sensor_offset = sensor_offset
        self.lambda_step = lambda_step
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end

        # Load led_axis_y_positions from CSV
        df = pd.read_csv(file_path)
        self.led_axis_y_positions = torch.tensor(df['axis_y_position_mm'].values / 1000, dtype=torch.float32, device=device)  # Convert to meters
        # define the left as minus axis
        self.led_axis_y_positions = -self.led_axis_y_positions

        # # SPD wavelengths and light SPD
        # self.wavelengths = torch.arange(lambda_start, lambda_end + 1, lambda_step, device=device, dtype=torch.float32)
        # self.light_spd = nn.Parameter(torch.rand(len(self.wavelengths), device=device))

        # Define initial SPD points
        initial_spd_points = {
            lambda_start: 0,
            380: 0.0,  # Starting point
            440: 0.8,  # Peak
            470: 0.3,  # Valley
            500: 0.8,  # Start of plateau
            640: 1.0,  # Red rise
            780: 0.0,  # Ending point
            lambda_end: 0
        }

        # Create wavelength grid
        self.wavelengths = torch.arange(self.lambda_start, self.lambda_end + 1, self.lambda_step, device=device, dtype=torch.float32)
        
        # Interpolate SPD
        known_wavelengths = list(initial_spd_points.keys())
        known_intensities = list(initial_spd_points.values())
        interpolator = interp1d(known_wavelengths, known_intensities, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        interpolated_spd = torch.tensor(interpolator(self.wavelengths.numpy()), dtype=torch.float32, device=device)

        # Convert SPD to logits that when passed through sigmoid will yield the interpolated SPD
        interpolated_spd = torch.clamp(interpolated_spd, min=1e-6, max=1 - 1e-6)  # Avoid extreme values close to 0 or 1
        spd_logits = torch.log(interpolated_spd / (1 - interpolated_spd))  # Inverse of sigmoid: logit(p) = log(p / (1 - p))

        # Parameterize spd_logits to be optimized during training
        self.spd_logits = nn.Parameter(spd_logits)

        # Define light SPD as sigmoid of logits
        self.light_spd = torch.sigmoid(self.spd_logits)

        print("light_spd: ", self.light_spd)

        self.visualizer = SpectrumVisualizer('ciexyz31_1.txt')

        self.H, self.W = 640, 480

    def forward(self):
        L_grating_to_LED = 0.1  # Distance from grating to LED in meters
        L_grating_to_sensor = 0.1  # Distance from grating to sensor in meters
        theta_inc = torch.atan(self.led_axis_y_positions / L_grating_to_LED)

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
            if i%10 == 0:
                # The visualize_and_save function expects a 3D tensor (C, H, W)
                # Adjust the dimension of frame for visualization
                frame_rgb = self.visualizer.visualize_and_save(frame.detach(), self.wavelengths.cpu(), file_name_pattern.format(i))


    def save_initial_spd(self, filename='initial_spd_plot.png'):
        plt.figure(figsize=(10, 5))
        plt.ylim(0, 1.2)
        plt.plot(self.wavelengths.numpy(), torch.sigmoid(self.spd_logits).detach().numpy(), label='Initial SPD')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Spectral Power Distribution (SPD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    file_path = 'unique_timestamps_and_y_positions.csv'  # Adjust path as needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GratingLightDispersionModel(file_path, device=device)
    model.save_initial_spd('initial_spd_plot.png')
    dispersed_light = model.forward()
    model.visualize_output(dispersed_light)
