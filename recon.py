import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Make only GPU 1 visible
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import subprocess
from spectrum_visualizer import SpectrumVisualizer

class GratingLightDispersionModel(nn.Module):
    def __init__(self, file_path, lambda_start=260, lambda_end=850, lambda_step=10, d_grating=1/(600*1e3), sensor_offset=0.05, device='cpu'):
        """
        Initializes the Grating Light Dispersion Model with specified spectral and geometric parameters.

        Args:
        - file_path (str): Path to the CSV file containing data for LED positions.
        - lambda_start (int): Starting wavelength for the light spectrum (in nm).
        - lambda_end (int): Ending wavelength for the light spectrum (in nm).
        - lambda_step (int): Step size between wavelengths in the spectrum (in nm).
        - d_grating (float): The grating pitch, which determines the separation of light into its component wavelengths.
        - sensor_offset (float): The offset in meters of the sensor from a defined reference position, typically the center.
        - device (str): The computing device ('cpu' or 'cuda') on which the tensors are to be allocated.
        """
        super(GratingLightDispersionModel, self).__init__()
        self.device = device
        self.d_grating = d_grating
        self.sensor_offset = sensor_offset
        self.lambda_step = lambda_step
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end

        self.H, self.W = 640, 480

        # Load LED axis y-positions from a CSV file and convert from millimeters to meters.
        df = pd.read_csv(file_path)
        self.led_axis_y_positions = torch.tensor(df['axis_y_position_mm'].values / 1000, dtype=torch.float32, device=device)
        
        # Generate wavelengths from lambda_start to lambda_end with a step of lambda_step.
        self.wavelengths = torch.arange(self.lambda_start, self.lambda_end + 1, self.lambda_step, device=device, dtype=torch.float32)
        
        # Define key spectral power distribution (SPD) points for interpolation.
        initial_spd_points = {
            lambda_start: 0,
            380: 0.0,
            440: 0.8,
            470: 0.3,
            500: 0.8,
            640: 1.0,
            780: 0.0,
            lambda_end: 0
        }

        # Use scipy's interp1d to create a smooth SPD curve between defined points.
        known_wavelengths = list(initial_spd_points.keys())
        known_intensities = list(initial_spd_points.values())
        interpolator = interp1d(known_wavelengths, known_intensities, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        interpolated_spd = torch.tensor(interpolator(self.wavelengths.cpu().numpy()), dtype=torch.float32, device=device)
        
        # Avoid values very close to 0 or 1 to ensure numerical stability in sigmoid and logit functions.
        interpolated_spd = torch.clamp(interpolated_spd, min=1e-6, max=1 - 1e-6)
        
        # Convert interpolated SPD to logits to use as parameters for optimization.
        spd_logits = torch.log(interpolated_spd / (1 - interpolated_spd))
        self.spd_logits = nn.Parameter(spd_logits)
        self.light_spd = torch.sigmoid(self.spd_logits)
        
        # Parameters for dynamic control during training or inference.
        self.event_threshold = nn.Parameter(torch.tensor([1e-6], device=device))
        self.absorption_spectrum_logits = nn.Parameter(torch.zeros((self.H, self.W, len(self.wavelengths)), device=device))
        
        # self.spectrum2intensity = nn.Parameter(torch.full((len(self.wavelengths),), 1/len(self.wavelengths), device=self.device))
        # Initialize spectrum-to-intensity conversion vector as trainable logits
        uniform_value = 1.0 / len(self.wavelengths)
        # Calculate initial logits from a uniform distribution
        initial_logits = torch.log(torch.tensor([uniform_value / (1 - uniform_value)], device=device))
        # Repeat the logits across the wavelength dimension
        self.spectrum2intensity_logits = nn.Parameter(initial_logits.repeat(len(self.wavelengths)), requires_grad=False)

        self.polarity_scale = nn.Parameter(torch.tensor([1.], device=device))  # Initial value of 100


        # Visualizer for the light spectrum.
        self.visualizer = SpectrumVisualizer('ciexyz31_1.txt')
        

    # def forward(self):
    #     # Calculate angles of incidence and diffraction based on LED and sensor positions.
    #     L_grating_to_LED = 0.1  # Distance from grating to LED source.
    #     L_grating_to_sensor = 0.1  # Distance from grating to sensor.
    #     theta_inc = torch.atan(self.led_axis_y_positions / L_grating_to_LED)
    #     sensor_width_m = 4.32e-3  # Total width of the sensor in meters.
    #     x_sensor = torch.linspace(-sensor_width_m / 2, sensor_width_m / 2, steps=self.W, device=self.device) + self.sensor_offset
    #     theta_diff = torch.atan(x_sensor / L_grating_to_sensor)

    #     # Calculate diffraction patterns based on the grating equation.
    #     lambda_diff_nm = self.d_grating * (torch.sin(theta_inc).unsqueeze(-1) + torch.sin(theta_diff)) * 1e9

    #     # Debugging: Print shapes of tensors to verify correct alignment
    #     print("Theta_inc shape:", theta_inc.shape)
    #     print("X_sensor shape:", x_sensor.shape)
    #     print("Lambda_diff_nm shape:", lambda_diff_nm.shape)

    #     # Calculate how closely each sensor position's wavelength matches available light SPD wavelengths.
    #     diff = torch.abs(lambda_diff_nm.unsqueeze(-1) - self.wavelengths.unsqueeze(0).unsqueeze(0))
    #     soft_onehot = F.softmax(-diff * 0.5, dim=-1)  # Convert differences to probabilities.

    #     # Generate the light spectrum at each sensor position.
    #     spectrum = soft_onehot * self.light_spd[None, None, :]
    #     spectrum = spectrum.unsqueeze(1)[:32]
    #     print("Spectrum shape:", spectrum.shape)

    #     absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).unsqueeze(0)  # Use unsqueeze to match dimensions
    #     print("Absorption_spectrum shape:", absorption_spectrum.shape)

    #     # modulated_spectrum = spectrum * absorption_spectrum  # Assumes broadcasting
    #     # Using einsum for correctly broadcasting across the desired dimensions.
    #     # 't' for time, 'h' for height, 'w' for width, 's' for spectral dimension
    #     modulated_spectrum = torch.einsum('thws,ahws->tahws', spectrum, absorption_spectrum)

    #     print("Modulated_spectrum shape:", modulated_spectrum.shape)


    #     # return  # Early return for debugging purposes

    #     # Compute overall intensity by summing modulated spectrum, weighted by spectrum-to-intensity conversion factors.
    #     self.spectrum_to_intensity = torch.sigmoid(self.spectrum2intensity_logits)
    #     intensity = torch.sum(modulated_spectrum * self.spectrum_to_intensity[None, None, None, :], dim=-1)

    #     # Calculate temporal gradient of intensity to detect changes.
    #     gradient = torch.diff(intensity, dim=0)
    #     events = torch.bernoulli(torch.sigmoid(torch.abs(gradient) - self.event_threshold))* torch.sign(gradient)

    #     return events

    def forward(self, input_data, batch_size=32):
        # We'll assume that total number of timestamps T can be divided evenly by batch_size for simplicity
        H, W = self.H, self.W  # total timestamps, height, width

        T = input_data.shape[0]  # Assuming input_data is the data tensor with shape [T, H, W]
        num_batches = (T + batch_size - 1) // batch_size  # This ensures that we cover all data
        
        
        all_events = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size + 1  # +1 to include one extra timestamp for diff calculation

            # Calculate angles of incidence and diffraction for the batch
            theta_inc = torch.atan(self.led_axis_y_positions[start_idx:end_idx] / 0.1)
            x_sensor = torch.linspace(-4.32e-3 / 2, 4.32e-3 / 2, steps=W, device=self.device) + self.sensor_offset
            theta_diff = torch.atan(x_sensor / 0.1)

            lambda_diff_nm = self.d_grating * (torch.sin(theta_inc).unsqueeze(-1) + torch.sin(theta_diff)) * 1e9
            diff = torch.abs(lambda_diff_nm.unsqueeze(-1) - self.wavelengths.unsqueeze(0).unsqueeze(0))
            soft_onehot = F.softmax(-diff * 0.5, dim=-1)

            # Spectrum for the current batch
            spectrum = soft_onehot * self.light_spd[None, None, :]
            spectrum = spectrum.unsqueeze(1)

            absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).unsqueeze(0)
            # modulated_spectrum = torch.einsum('thws,ahws->tahws', spectrum, absorption_spectrum)
            modulated_spectrum = spectrum * absorption_spectrum

            print("modulated_spectrum.shape: ", modulated_spectrum.shape)

            # Compute intensity
            spectrum_to_intensity = torch.sigmoid(self.spectrum2intensity_logits)
            intensity = torch.sum(modulated_spectrum * spectrum_to_intensity[None, None, None, :], dim=-1)

            # Calculate gradient (events) and track gradients properly
            intensity_diff = torch.diff(intensity, dim=0)  # differential over time
            prob = torch.sigmoid(torch.abs(intensity_diff) - self.event_threshold)
            sample = torch.bernoulli(prob)
            is_events = (sample - prob).detach() + prob  # reparameterization trick for gradient tracking
            polarity = torch.tanh(intensity_diff * torch.exp(self.polarity_scale))
            all_events.append(is_events * polarity)

        # Combine all batches
        all_events = torch.cat(all_events, dim=0)  # should be [1049, H, W]

        print("all_events.shape: ", all_events.shape)
        print("all_events.max(): ", all_events.max())
        print("all_events.min(): ", all_events.min())

        return all_events




    def save_all_frames_as_images(self, folder='frames_forward', frames=None):
        """
        Save each frame as an image for visualization purposes.

        Args:
        - folder (str): Directory where images will be saved.
        - frames (tensor): Event frames to save.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            frame_display = frame.cpu().numpy()
            ax.imshow(frame_display, cmap='gray', extent=[0, 480, 640, 0])
            y_pos, x_pos = (frame_display > 0.).nonzero()
            y_neg, x_neg = (frame_display < -0.).nonzero()
            ax.scatter(x_pos, y_pos, color='green', s=1)
            ax.scatter(x_neg, y_neg, color='red', s=1)
            plt.savefig(f"{folder}/frame_{i:04d}.png", dpi=100)
            plt.close()

        print(f"Saved all frames as images in the {folder}/ directory.")

    def frames_to_video(self, folder='frames_forward', output_video='frames_forward_video_with_ffmpeg.mp4', fps=60):
        """
        Compile frames into a video using FFmpeg.

        Args:
        - folder (str): Directory containing image frames.
        - output_video (str): Output video file name.
        - fps (int): Frame rate of the output video.
        """
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder, 'frame_%04d.png'),
            '-vf', 'scale=800:600',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_video}")

# Usage example within a main block or function
if __name__ == "__main__":
    file_path = 'unique_timestamps_and_y_positions.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GratingLightDispersionModel(file_path, device=device)
    # model.save_initial_spd('initial_spd_plot.png')
    # dispersed_light = model.forward()
    # events = model.forward()
    # model.visualize_output(dispersed_light)
    # model.save_all_frames_as_images('frames_forward', frames=events.detach())
    # model.frames_to_video()


    data = torch.load("rotated_frames_640x480.pt")


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    outer_batch_size = 512
    inner_batch_size = 32

    for epoch in range(1000):
        running_loss = 0.0
        num_outer_batches = (data.shape[0] + outer_batch_size - 1) // outer_batch_size

        for i in range(0, data.shape[0], outer_batch_size):
            end_idx = min(i + outer_batch_size, data.shape[0])
            outer_batch_data = data[i:end_idx].to(device)

            # Call forward with smaller batch processing
            predicted_events = model.forward(outer_batch_data)

            # Assuming ground truth is prepared correctly for comparison
            ground_truth = data[i:end_idx].to(device)  # Make sure sizes match
            loss = F.mse_loss(predicted_events, ground_truth.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / num_outer_batches}")

