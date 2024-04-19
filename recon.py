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

class BinaryDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (input_flat * target_flat).sum()
        cardinality = input_flat.sum() + target_flat.sum()

        dice_loss = 1 - (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
        return dice_loss

class CombinedDiceMSELoss(nn.Module):
    def __init__(self):
        super(CombinedDiceMSELoss, self).__init__()
        self.dice_loss = BinaryDiceLoss()
        self.mse_loss = nn.MSELoss()  # Using PyTorch's built-in MSE loss

    def forward(self, events_pred, events_target, polarity_pred, polarity_target):
        # Calculate Dice loss for events
        dice_loss = self.dice_loss(events_pred, events_target)

        # Calculate MSE loss for polarity
        mse_loss = self.mse_loss(polarity_pred, polarity_target)

        # Combine losses
        combined_loss = dice_loss + mse_loss  # You can also add weights here if needed
        return combined_loss



class GratingLightDispersionModel(nn.Module):
    def __init__(self, file_path, lambda_start=260, lambda_end=870, lambda_step=10, d_grating=1/(600*1e3), sensor_offset=0.05, device='cpu'):
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
        # df = pd.read_csv(file_path)
        # self.led_axis_y_positions = torch.tensor(df['axis_y_position_mm'].values / 1000, dtype=torch.float32, device=device)
        
        # Generate wavelengths from lambda_start to lambda_end with a step of lambda_step.
        self.wavelengths = torch.arange(self.lambda_start, self.lambda_end + 1, self.lambda_step, device=device, dtype=torch.float32)
        
        # Define key spectral power distribution (SPD) points for interpolation.
        # initial_spd_points = {
        #     lambda_start: 0,
        #     380: 0.0,
        #     440: 0.8,
        #     470: 0.3,
        #     500: 0.8,
        #     640: 1.0,
        #     780: 0.0,
        #     lambda_end: 0
        # }

        initial_spd_points = {
            lambda_start: 1.0,
            380: 1.0,
            440: 1.0,
            470: 1.0,
            500: 1.0,
            640: 1.0,
            780: 1.0,
            lambda_end: 1.0
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
        self.spd_logits = nn.Parameter(spd_logits,)
        
        
        # Parameters for dynamic control during training or inference.
        event_threshold = 1e-2
        event_threshold_logits = torch.log(torch.tensor([event_threshold / (1 - event_threshold)], device=device))
        self.event_threshold_logits = nn.Parameter(event_threshold_logits)

        self.absorption_spectrum_logits = nn.Parameter(torch.zeros((self.H, self.W, len(self.wavelengths)), device=device))
        
        # self.spectrum2intensity = nn.Parameter(torch.full((len(self.wavelengths),), 1/len(self.wavelengths), device=self.device))
        # # Initialize spectrum-to-intensity conversion vector as trainable logits
        # uniform_value = 1.0 / len(self.wavelengths)
        # # Calculate initial logits from a uniform distribution
        # initial_logits = torch.log(torch.tensor([uniform_value / (1 - uniform_value)], device=device))
        # # Repeat the logits across the wavelength dimension
        # self.spectrum2intensity_logits = nn.Parameter(initial_logits.repeat(len(self.wavelengths)), requires_grad=False)

        # Initialize spectrum-to-intensity conversion vector as trainable logits
        # Set initial logits to zero
        # spectrum2intensity_logits = torch.zeros(len(self.wavelengths), device=self.device)
        
        # Make them a nn.Parameter so they can be trained
        # self.spectrum2intensity_logits = nn.Parameter(spectrum2intensity_logits, requires_grad=False)


        # self.polarity_scale = nn.Parameter(torch.tensor([3.], device=device))  # Initial value of 100


        # Visualizer for the light spectrum.
        self.visualizer = SpectrumVisualizer('ciexyz31_1.txt')
        

    
    # def forward(self, input_data, batch_size=32):
    #     self.led_axis_y_positions = input_data

    #     # We'll assume that total number of timestamps T can be divided evenly by batch_size for simplicity
    #     H, W = self.H, self.W  # total timestamps, height, width

    #     T = input_data.shape[0]  # Assuming input_data is the data tensor with shape [T, H, W]
    #     num_batches = (T + batch_size - 1) // batch_size  # This ensures that we cover all data
        
        
    #     all_events = []
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = start_idx + batch_size + 1  # +1 to include one extra timestamp for diff calculation

    #         # Calculate angles of incidence and diffraction for the batch
    #         theta_inc = torch.atan(self.led_axis_y_positions[start_idx:end_idx] / 0.1)
    #         x_sensor = torch.linspace(-4.32e-3 / 2, 4.32e-3 / 2, steps=W, device=self.device) + self.sensor_offset
    #         theta_diff = torch.atan(x_sensor / 0.1)

    #         lambda_diff_nm = self.d_grating * (torch.sin(theta_inc).unsqueeze(-1) + torch.sin(theta_diff)) * 1e9
    #         diff = torch.abs(lambda_diff_nm.unsqueeze(-1) - self.wavelengths.unsqueeze(0).unsqueeze(0))
    #         soft_onehot = F.softmax(-diff * 0.5, dim=-1)

    #         # Spectrum for the current batch
    #         spectrum = soft_onehot * self.light_spd[None, None, :]
    #         spectrum = spectrum.unsqueeze(1)

    #         absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).unsqueeze(0)
    #         # modulated_spectrum = torch.einsum('thws,ahws->tahws', spectrum, absorption_spectrum)
    #         modulated_spectrum = spectrum * absorption_spectrum

    #         # print("modulated_spectrum.shape: ", modulated_spectrum.shape)

    #         # Compute intensity
    #         spectrum_to_intensity = torch.sigmoid(self.spectrum2intensity_logits)
    #         intensity = torch.sum(modulated_spectrum * spectrum_to_intensity[None, None, None, :], dim=-1)

    #         # Calculate gradient (events) and track gradients properly
    #         intensity_diff = torch.diff(intensity, dim=0)  # differential over time
    #         prob = torch.sigmoid(torch.abs(intensity_diff) - self.event_threshold)
    #         sample = torch.bernoulli(prob)
    #         is_events = (sample - prob).detach() + prob  # reparameterization trick for gradient tracking
    #         polarity = torch.tanh(intensity_diff * torch.exp(self.polarity_scale))
    #         all_events.append(is_events * polarity)

    #     # Combine all batches
    #     all_events = torch.cat(all_events, dim=0)  # should be [1049, H, W]

    #     # print("all_events.shape: ", all_events.shape)
    #     # print("all_events.max(): ", all_events.max())
    #     # print("all_events.min(): ", all_events.min())

    #     return all_events



    def forward(self, input_data):
        self.led_axis_y_positions = input_data

        # self.light_spd = torch.sigmoid(self.spd_logits)
        # absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).unsqueeze(0)
        # spectrum_to_intensity = F.softmax(self.spectrum2intensity_logits, dim=0)
        # event_threshold = torch.sigmoid(self.event_threshold_logits)


        # Print the Y positions of LEDs
        # print("LED Y-axis positions:", self.led_axis_y_positions)
        
        # Calculate and print the light spectral power distribution (SPD)
        self.light_spd = torch.sigmoid(self.spd_logits)
        # print("Light SPD:", self.light_spd)
        
        # Calculate and print the absorption spectrum
        absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).unsqueeze(0)
        # print("Absorption spectrum:", absorption_spectrum)
        
        # Calculate and print the spectrum to intensity conversion
        # spectrum_to_intensity = F.softmax(self.spectrum2intensity_logits, dim=0)
        # print("Spectrum to intensity:", spectrum_to_intensity)
        
        # Calculate and print the event threshold
        event_threshold = torch.sigmoid(self.event_threshold_logits)
        # print("Event threshold:", event_threshold)

        # return

        # We'll assume that total number of timestamps T can be divided evenly by batch_size for simplicity
        H, W = self.H, self.W  # total timestamps, height, width
        

        # Calculate angles of incidence and diffraction for the batch
        theta_inc = torch.atan(self.led_axis_y_positions / 0.1)
        x_sensor = torch.linspace(-4.32e-3 / 2, 4.32e-3 / 2, steps=W, device=self.device) + self.sensor_offset
        theta_diff = torch.atan(x_sensor / 0.1)

        lambda_diff_nm = self.d_grating * (torch.sin(theta_inc).unsqueeze(-1) + torch.sin(theta_diff)) * 1e9
        diff = torch.abs(lambda_diff_nm.unsqueeze(-1) - self.wavelengths.unsqueeze(0).unsqueeze(0))
        soft_onehot = F.softmax(-diff * 1.0, dim=-1)

        # Spectrum for the current batch
        spectrum = soft_onehot * self.light_spd[None, None, :]
        spectrum = spectrum.unsqueeze(1)

        
        # modulated_spectrum = torch.einsum('thws,ahws->tahws', spectrum, absorption_spectrum)
        modulated_spectrum = spectrum * absorption_spectrum

        # print("modulated_spectrum.shape: ", modulated_spectrum.shape)

        # Compute intensity
        # spectrum_to_intensity = torch.sigmoid(self.spectrum2intensity_logits)
        # Apply softmax to the logits to get the spectrum_to_intensity values
        
        # intensity = torch.sum(modulated_spectrum * spectrum_to_intensity[None, None, None, :], dim=-1)
        intensity = torch.mean(modulated_spectrum, dim=-1)

        # Calculate gradient (events) and track gradients properly
        intensity_diff = torch.diff(intensity, dim=0)  # differential over time
        # self.event_threshold = torch.sigmoid(self.event_threshold_logits)
        
        # polarith_scale = torch.exp(self.polarity_scale)
        # polarith_scale = self.polarity_scale
        prob = torch.sigmoid(torch.log(torch.abs(intensity_diff) + event_threshold*1e-3) - torch.log(event_threshold))

        # print(prob)

        sample = torch.bernoulli(prob)
        events = (sample - prob).detach() + prob  # reparameterization trick for gradient tracking
        polarity = torch.tanh(intensity_diff / event_threshold)
        
        # events = is_events * polarity


        # print("all_events.shape: ", polarity.shape)
        # print("all_events.max(): ", polarity.max())
        # print("all_events.min(): ", polarity.min())

        return events, polarity


    def save_sample_spectrum(self, directory, epoch):
        os.makedirs(directory, exist_ok=True)
        # Sigmoid activation to convert logits to probabilities
        absorption_spectrum = torch.sigmoid(self.absorption_spectrum_logits).detach()

        # Permute dimensions to move the wavelength to the first dimension (channels)
        # absorption_spectrum_permuted = absorption_spectrum.permute(2, 0, 1)  # New shape [len(wavelengths), H, W]
        absorption_spectrum_permuted = absorption_spectrum  # New shape [len(wavelengths), H, W]


        # Assuming the visualizer can handle the entire tensor as is
        file_name = f"{directory}/spectrum_at_epoch_{epoch}.png"
        # Here you pass the frame which now has dimensions [1, H, W]
        self.visualizer.visualize_and_save(absorption_spectrum_permuted, self.wavelengths.cpu(), file_name)

    def visualize_output(self, output_tensor, file_name_pattern='dispersed_frames/dispersed_light_{:04d}.png'):
        os.makedirs("dispersed_frames", exist_ok=True)
        for i, frame in enumerate(output_tensor):
            # if i%10 == 0 and i>=3200:
            if i%1 == 0:
                # The visualize_and_save function expects a 3D tensor (C, H, W)
                # Adjust the dimension of frame for visualization
                frame_rgb = self.visualizer.visualize_and_save(frame.detach(), self.wavelengths.cpu(), file_name_pattern.format(i))


    def save_initial_spd(self, filename='initial_spd_plot.png'):
        plt.figure(figsize=(10, 5))
        plt.ylim(0, 1.2)
        plt.plot(self.wavelengths.cpu().numpy(), torch.sigmoid(self.spd_logits).detach().cpu().numpy(), label='Initial SPD')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Spectral Power Distribution (SPD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


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
    # Load LED axis y-positions from a CSV file and convert from millimeters to meters.
    led_axis_y_positions_df = pd.read_csv("unique_timestamps_and_y_positions.csv")
    led_axis_y_positions = torch.tensor(led_axis_y_positions_df['axis_y_position_mm'].values / 1000, dtype=torch.float32, device=device)
    


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = CombinedDiceMSELoss()
    outer_batch_size = 32
    

    for epoch in range(5000):
        running_loss = 0.0
        num_batches = (data.shape[0] + outer_batch_size - 1) // outer_batch_size

        

        for i in range(1, data.shape[0], outer_batch_size):
            optimizer.zero_grad()
            
            end_idx = min(i + outer_batch_size, data.shape[0])
            led_positions_batch = led_axis_y_positions[i-1:end_idx].to(device)

            predicted_events, predicted_polarity = model(led_positions_batch)

            


            # Ensure ground truth is aligned with the output from the model
            ground_truth = data[i:end_idx].to(device)
         

            # print("Polarity Prediction Shape:", predicted_polarity.shape)
            # print("Polarity Target Shape:", ground_truth.shape)

            # loss = F.mse_loss(predicted_events, ground_truth.float())
            # Assuming events_pred, events_target, polarity_pred, polarity_target, and num_classes_polarity are available

            loss = loss_function(
                predicted_events, torch.abs(ground_truth.float()), 
                predicted_polarity, ground_truth.float(), 
            )

            
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()


        if epoch % 100 == 0:
            model.save_sample_spectrum("frames_sample_spectrum", epoch)

        print(f"Epoch {epoch+1}, Loss: {running_loss}")

