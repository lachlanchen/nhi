import pandas as pd
from timestamp2position import CNCMotorSystem

class EventDataWithPosition:
    def __init__(self, axis_csv_path, event_csv_path):
        self.axis_csv_path = axis_csv_path
        self.event_csv_path = event_csv_path
        self.motor_system = CNCMotorSystem(axis_csv_path)
        self.filtered_events = None
    
    # def load_and_filter_events(self):
    #     # Load the axis positions data to determine the time range
    #     axis_data = pd.read_csv(self.axis_csv_path)
    #     axis_data['system_time'] = pd.to_datetime(axis_data['system_time'], format='%H:%M:%S.%f')
    #     start_time, end_time = axis_data['system_time'].min(), axis_data['system_time'].max()

    #     # Load the event data
    #     event_data = pd.read_csv(self.event_csv_path)
    #     event_data['event_timestamp'] = pd.to_datetime(event_data['event_timestamp'], format='%H:%M:%S.%f')

    #     # Filter event_data to only include events within the time range of axis_1_positions.csv
    #     self.filtered_events = event_data[(event_data['event_timestamp'] >= start_time) & (event_data['event_timestamp'] <= end_time)]

    def timestamp_to_microseconds(self, timestamp):
        # Convert a timestamp string (HH:MM:SS.ffffff) into microseconds
        parts = timestamp.split(':')
        hours, minutes = int(parts[0]), int(parts[1])
        seconds, microseconds = map(int, parts[2].split('.'))
        return (hours * 3600 + minutes * 60 + seconds) * 1000000 + microseconds

    def is_within_range(self, event_time, start_time, end_time):
        # Convert string times to numeric microseconds and compare
        event_time_microseconds = self.timestamp_to_microseconds(event_time)
        start_time_microseconds = self.timestamp_to_microseconds(start_time)
        end_time_microseconds = self.timestamp_to_microseconds(end_time)
        return start_time_microseconds <= event_time_microseconds <= end_time_microseconds


    # def load_and_filter_events(self):
    #     axis_data = pd.read_csv(self.axis_csv_path)
    #     axis_data['system_time'] = pd.to_datetime(axis_data['system_time'], format='%H:%M:%S.%f')
    #     start_time, end_time = axis_data['system_time'].dt.time.min(), axis_data['system_time'].dt.time.max()

    #     event_data = pd.read_csv(self.event_csv_path)
    #     event_data['event_timestamp'] = pd.to_datetime(event_data['event_timestamp'], format='%H:%M:%S.%f')

    #     # Convert event_timestamp to just time for comparison
    #     event_times = event_data['event_timestamp'].dt.time

    #     # Filter based on time comparison
    #     self.filtered_events = event_data[(event_times >= start_time) & (event_times <= end_time)]

    def load_and_filter_events(self):
        axis_data = pd.read_csv(self.axis_csv_path)
        start_time = axis_data['system_time'].min()
        end_time = axis_data['system_time'].max()

        # # Use the first and last rows to determine the time range
        # start_time = axis_data.iloc[0]['system_time']
        # end_time = axis_data.iloc[-1]['system_time']

        event_data = pd.read_csv(self.event_csv_path)

        self.filtered_events = event_data[event_data['event_timestamp'].apply(
            lambda x: self.is_within_range(x, start_time, end_time))]
    
    def predict_positions(self):
        if self.filtered_events is not None:
            self.filtered_events['axis_y_position_mm'] = self.filtered_events['event_timestamp'].apply(
                # lambda x: self.motor_system.predict_position(x.strftime('%H:%M:%S.%f'))
                lambda x: self.motor_system.predict_position(x)
            )
        else:
            print("Filtered events are not loaded. Please run load_and_filter_events() first.")
    
    # def save_filtered_data(self, output_path):
    #     if self.filtered_events is not None:
    #         self.filtered_events.to_csv(output_path, index=False)
    #         print(f"Filtered and augmented event data saved successfully to {output_path}.")
    #     else:
    #         print("Filtered events are not loaded. Please run load_and_filter_events() and predict_positions() first.")

    def save_filtered_data(self, output_path):
        if self.filtered_events is not None:
            # Convert datetime to time string to remove the date part
            # self.filtered_events['event_timestamp'] = self.filtered_events['event_timestamp'].dt.time
            # self.filtered_events.to_csv(output_path, index=False)
            final_data = self.filtered_events.drop(columns=['system_timestamp'])
            final_data.to_csv(output_path, index=False)
            print(f"Filtered and augmented event data saved successfully to {output_path}.")
        else:
            print("Filtered events are not loaded. Please run load_and_filter_events() and predict_positions() first.")


if __name__ == '__main__':
    # Example usage:
    # Initialize the class with paths to your CSV files
    event_data_processor = EventDataWithPosition('data/axis_1_positions.csv', 'data/event_data.csv')

    # Load and filter events
    event_data_processor.load_and_filter_events()

    # Predict positions
    event_data_processor.predict_positions()

    # Save the filtered and augmented data to a new CSV
    event_data_processor.save_filtered_data('data/filtered_event_data_with_positions.csv')
