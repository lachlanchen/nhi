import numpy as np
import scipy.optimize as opt

# Constants in meters
d = 1 / 600000  # Grating spacing in meters (600 lines/mm)
z1 = 0.1  # z1 in meters
z2 = 0.1  # z2 in meters

# Positions in meters
xi_15mm = 0.015
xi_50mm = 0.050

# Wavelengths in meters (red to blue spectrum)
wavelengths = np.linspace(400e-9, 700e-9, 100)

# Function to calculate x positions
def calculate_x(xi, wavelength, d, z1, z2):
    def equation(x):
        return wavelength - d * (xi / np.sqrt(xi**2 + z1**2) + x / np.sqrt(x**2 + z2**2))
    x_initial_guess = 0  # Initial guess for the solver
    x_solution = opt.fsolve(equation, x_initial_guess)
    return x_solution[0]

# Calculate x positions for both xi values
x_positions_15mm = [calculate_x(xi_15mm, wl, d, z1, z2) for wl in wavelengths]
x_positions_50mm = [calculate_x(xi_50mm, wl, d, z1, z2) for wl in wavelengths]

# Convert x positions from meters to micrometers for easier interpretation
x_positions_15mm_um = np.array(x_positions_15mm) * 1e6
x_positions_50mm_um = np.array(x_positions_50mm) * 1e6

# Print results
print("Wavelength (nm) | x position at 15mm (um) | x position at 50mm (um)")
for wl, x15, x50 in zip(wavelengths * 1e9, x_positions_15mm_um, x_positions_50mm_um):
    print(f"{wl:10.1f}    | {x15:24.4f}    | {x50:24.4f}")

