import cv2
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, OnDemandFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

class MetavisionTools:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.event_iterator = None
        self.frame_generator = None
        self.window = None
        self.width = 0
        self.height = 0

    def setup_event_iterator(self):
        """Setup event iterator based on file or live stream."""
        self.event_iterator = EventsIterator(self.file_path, delta_t=5000)
        self.height, self.width = self.event_iterator.get_size()

    def setup_frame_generator(self, mode='periodic', accumulation_time_us=20000, fps=50):
        """Setup the frame generator either periodic or on demand."""
        if mode == 'periodic':
            self.frame_generator = PeriodicFrameGenerationAlgorithm(self.width, self.height, accumulation_time_us, fps)
        else:
            self.frame_generator = OnDemandFrameGenerationAlgorithm(self.width, self.height, accumulation_time_us)

    def setup_window(self, title="Metavision Viewer"):
        """Setup a window for visualization."""
        self.window = MTWindow(title, self.width, self.height, BaseWindow.RenderMode.BGR)

    def visualize_xyt(self):
        """Visualize events in XYT space using PeriodicFrameGenerationAlgorithm."""
        self.setup_event_iterator()
        self.setup_frame_generator(mode='periodic')
        self.setup_window(title="Metavision XYT Visualization")

        for events in self.event_iterator:
            self.frame_generator.process_events(events)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for event in events:
                if event['p'] == 1:
                    frame[event['y'], event['x'], 2] = 255  # Positive polarity in red
                else:
                    frame[event['y'], event['x'], 0] = 255  # Negative polarity in blue

            self.window.show(frame)
            if self.window.should_close():
                break

        cv2.destroyAllWindows()

    def run_viewer(self):
        """Run generic event viewer based on the file or live stream setup."""
        self.visualize_xyt()  # Currently set to run XYT; modify as needed for other visualizations.

if __name__ == "__main__":
    tool = MetavisionTools("path_to_event_file.raw")
    tool.run_viewer()
