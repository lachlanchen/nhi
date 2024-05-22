import cv2
import numpy as np
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

def main(raw_file_path):
    # Initialize event iterator for the RAW file
    from metavision_core.event_io import EventsIterator
    event_iterator = EventsIterator(raw_file_path, delta_t=10000)  # adjust delta_t as needed

    # Obtain the sensor dimensions from the iterator
    height, width = event_iterator.get_size()

    # Initialize the frame generator
    frame_generator = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=50000)  # adjust time as needed

    # Create a window using the Metavision UI module
    window = MTWindow("Metavision Viewer", width, height, BaseWindow.RenderMode.BGR)

    # Define the callback for handling key events
    def keyboard_callback(key, scancode, action, mods):
        if action == UIAction.RELEASE and (key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q):
            window.set_close_flag()

    window.set_keyboard_callback(keyboard_callback)

    # Process events and display them
    for events in event_iterator:
        frame_generator.process_events(events)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for ev in events:
            if ev['p'] == 1:  # Check for polarity
                frame[ev['y'], ev['x'], 2] = 255  # Red for positive polarity
            else:
                frame[ev['y'], ev['x'], 0] = 255  # Blue for negative polarity

        window.show(frame)

        if window.should_close():
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evk5_raw_vis.py <path_to_raw_file>")
        sys.exit(1)
    main(sys.argv[1])
a