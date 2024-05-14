import numpy as np

class DiffractionGrating:
    def __init__(self, led_to_grating=100.0, grating_to_screen=100.0, lines_per_mm=600, wavelength_range=(380, 780)):
        self.led_to_grating = led_to_grating  # distance from LED to grating in mm
        self.grating_to_screen = grating_to_screen  # distance from grating to screen in mm
        self.lines_per_mm = lines_per_mm
        self.wavelength_range = wavelength_range
        self.line_spacing = 1 / lines_per_mm  # line spacing in mm (from lines per mm)
        self.diffraction_angle = 0  # diffraction angle in radians, adjustable

    def wavelength_to_rgb(self, wavelength):
        """ Convert wavelength (nm) to approximate RGB color """
        w = int(wavelength)
        if w >= 380 and w <= 440:
            R = -(w - 440.) / (440. - 380.)
            G = 0.0
            B = 1.0
        elif w >= 440 and w <= 490:
            R = 0.0
            G = (w - 440.) / (490. - 440.)
            B = 1.0
        elif w >= 490 and w <= 510:
            R = 0.0
            G = 1.0
            B = -(w - 510.) / (510. - 490.)
        elif w >= 510 and w <= 580:
            R = (w - 510.) / (580. - 510.)
            G = 1.0
            B = 0.0
        elif w >= 580 and w <= 645:
            R = 1.0
            G = -(w - 645.) / (645. - 580.)
            B = 0.0
        elif w >= 645 and w <= 780:
            R = 1.0
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        
        if w >= 380 and w <= 420:
            SSS = 0.3 + 0.7*(w - 380) / (420 - 380)
        elif w >= 420 and w <= 700:
            SSS = 1.0
        elif w >= 700 and w <= 780:
            SSS = 0.3 + 0.7*(780 - w) / (780 - 700)
        else:
            SSS = 0.0
        SSS *= 255
        return (int(SSS*R), int(SSS*G), int(SSS*B))

    def calculate_positions(self):
        wavelengths = np.linspace(self.wavelength_range[0], self.wavelength_range[1], 401)  # 1nm steps
        # Calculate the necessary incident angles for given diffraction angle (currently set to 0)
        incident_angles = np.arcsin((wavelengths * 1e-6) / self.line_spacing - np.sin(self.diffraction_angle))  # Consider diffraction angle in the calculation

        # Calculate the necessary LED position adjustments
        led_positions = -self.led_to_grating * np.tan(incident_angles)  # Calculate LED position adjustments

        # Calculate RGB colors for visualization
        colors = [self.wavelength_to_rgb(wl) for wl in wavelengths]

        return list(zip(wavelengths, led_positions, colors))

# Example usage
grating = DiffractionGrating(led_to_grating=100, grating_to_screen=100, lines_per_mm=600, wavelength_range=(380, 780))
positions = grating.calculate_positions()
for wavelength, led_position, color in positions:
    print(f"Wavelength: {wavelength} nm, LED Position Adjustment: {led_position:.2f} mm, Color: RGB{color}")
