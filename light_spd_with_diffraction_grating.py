import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_csv(file_path):
    return pd.read_csv(file_path)

def interpolate_data(x, y, new_x):
    interpolator = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    return interpolator(new_x)

def plot_data(wavelengths, light_spd, interpolated_spd, original_efficiency, interpolated_efficiency, product, output_dir):
    plt.figure(figsize=(14, 18))

    # Original Light SPD
    plt.subplot(3, 2, 1)
    plt.plot(light_spd['Wavelength'], light_spd['Intensity'], label='Original Light SPD')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Original Light SPD')
    plt.legend()
    plt.grid(True)

    # Interpolated Light SPD
    plt.subplot(3, 2, 2)
    plt.plot(wavelengths, interpolated_spd, label='Interpolated Light SPD')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Interpolated Light SPD')
    plt.legend()
    plt.grid(True)

    # Original Diffraction Grating Efficiency
    plt.subplot(3, 2, 3)
    plt.plot(original_efficiency['Wavelength(nm) 600 Grooves/mm'], original_efficiency['Absolute Efficiency(%)'], label='Original Efficiency 600 Grooves/mm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Efficiency (%)')
    plt.title('Original Diffraction Grating Efficiency')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    # Interpolated Diffraction Grating Efficiency
    plt.subplot(3, 2, 4)
    plt.plot(wavelengths, interpolated_efficiency, label='Interpolated Efficiency 600 Grooves/mm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Efficiency (rescaled)')
    plt.title('Interpolated Diffraction Grating Efficiency')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # Product of Light SPD and Efficiency
    plt.subplot(3, 2, 6)
    plt.plot(wavelengths, product, label='Product of Light SPD and Efficiency')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Product')
    plt.title('Product of Light SPD and Efficiency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'light_spd_with_diffraction_grating.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main(light_spd_path, diffraction_grating_path, output_dir):
    light_spd = read_csv(light_spd_path)
    diffraction_grating = read_csv(diffraction_grating_path)

    # Sort diffraction grating data by wavelength for 600 grooves/mm
    diffraction_grating = diffraction_grating.sort_values(by=f'Wavelength(nm) 600 Grooves/mm')

    wavelengths_spd = np.arange(400, 801)  # New wavelength range from 400 to 800 nm
    wavelengths_grating = np.arange(300, 1101)  # Keep the 300 to 1100 range for diffraction grating

    interpolated_spd = interpolate_data(light_spd['Wavelength'], light_spd['Intensity'], wavelengths_spd)
    interpolated_efficiency = interpolate_data(diffraction_grating[f'Wavelength(nm) 600 Grooves/mm'], diffraction_grating['Absolute Efficiency(%)'], wavelengths_grating)
    efficiency_in_spd_range = interpolate_data(wavelengths_grating, interpolated_efficiency, wavelengths_spd)

    # Rescale efficiency to 0-1 range
    efficiency_in_spd_range = efficiency_in_spd_range / 100.0

    product = interpolated_spd * efficiency_in_spd_range

    plot_data(wavelengths_spd, light_spd, interpolated_spd, diffraction_grating[[f'Wavelength(nm) 600 Grooves/mm', 'Absolute Efficiency(%)']], efficiency_in_spd_range, product, output_dir)

    # Save product data to CSV
    product_data = pd.DataFrame({
        'Wavelength': wavelengths_spd,
        'Light SPD': interpolated_spd,
        'Efficiency': efficiency_in_spd_range,
        'Intensity': product
    })
    product_csv_path = os.path.join(output_dir, 'light_spd_adjusted.csv')
    product_data.to_csv(product_csv_path, index=False)
    print(f"Saved product data to {product_csv_path}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Process light SPD and diffraction grating data.')
    parser.add_argument('light_spd_path', type=str, help='Path to the light SPD CSV file.')
    parser.add_argument('diffraction_grating_path', type=str, help='Path to the diffraction grating CSV file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output files.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.light_spd_path, args.diffraction_grating_path, args.output_dir)
