# Sync Image System

This directory hosts the dual camera capture GUI used to synchronise the frame-based Haikang camera with the event-based Metavision camera. The tooling is structured as follows:

- `dual_camera_gui.py` – main tkinter application that coordinates both cameras, handles configuration, and records synchronised streams.
- `haikang_sdk/` – vendor SDK samples and the `MvImport` Python bindings required for the frame camera.

## Requirements

- Python 3.10+
- `opencv-python`, `numpy`, `pillow`, `metavision-sdk` (event camera), `pywin32` on Windows
- Haikang SDK drivers installed on the host machine

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

(See the root project `requirements.txt` for the exact versions.)

## Running

```bash
python dual_camera_gui.py
```

The GUI lets you:

1. Discover and connect to the frame camera (Haikang) and the event camera (Metavision EVK).
2. Configure exposure, gain, and Metavision bias values live.
3. Preview both streams in real time with simple transformations.
4. Record either stream independently or perform unified recordings.

## Repository hygiene

Generated recordings, raw captures, and cache folders are excluded via `.gitignore`. Only the Python source and SDK bindings are versioned.