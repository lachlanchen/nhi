# Rotor Arduino Sketches

This folder contains the Arduino sketches used to drive the rotor alignment hardware. Each subfolder is a self-contained project that shares the same display driver (`OLED12864.*`) and pin-change interrupt helper (`PinChangeInt.h`). The variations capture different motion profiles and controller behaviours:

- `step/` - original driver with manual step timing.
- `step42/` - 42-step version tuned for the current rotor gearing.
- `step42_smooth/` - adds motion smoothing for quieter motion.
- `step42_with_key_int/` - integrates key switches for manual control.
- `step42_with_key_int_flat/` - flattened acceleration profile for consistent speed.
- `step42_with_key_int_wider/` - wider timing windows for more forgiving key input.
- `step_accelstepper/` - experimental version using the AccelStepper library.

Open any `.ino` file in the Arduino IDE (2.x preferred), install the `AccelStepper` library when required, and upload to the controller board.