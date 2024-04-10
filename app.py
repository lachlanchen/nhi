import tornado.ioloop
import tornado.web
import threading
import time
import os
import csv
from led import ArduinoLED
from event_sensor import EventSensor
from cnc.cnc import MotorSystem

# Global variables to store event data
event_data = []

import os
import shutil
from datetime import datetime


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html")

class StartSequenceHandler(tornado.web.RequestHandler):
    def post(self):
        # Start the sequence in a new thread to not block the Tornado server
        threading.Thread(target=start_sequence).start()
        self.write({"status": "Sequence started"})

def start_sequence():

    def cnc_movement_sequence():

        # Perform CNC Y-axis movements sequentially within this thread
        motor_system.move(1, 30, -1, 20)  # Move Y axis -30
        motor_system.move(1, 60, 1, 20)   # Move Y axis +30
        # motor_system.move(1, 30, -1, 20)
        # Move to origin
        # motor_system.move_to_origin()
        
    # Format the current datetime for the directory name
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir_name = f"data_{current_datetime}"

    # Check if the original 'data' directory exists
    if os.path.exists('data'):
        # Move the 'data' directory to 'data_datetime'
        shutil.move('data', new_dir_name)

    # Create a new 'data' directory
    os.makedirs('data', exist_ok=True)

    # Initialize devices
    led_control = ArduinoLED(port='COM4')
    sensor = EventSensor()
    dll_path = "cnc/FMC4030Lib-x64-20220329/FMC4030-Dll.dll"
    motor_system = MotorSystem(dll_path)

    # Start LED
    led_control.led_on()
    time.sleep(3)

    # Start recording events
    sensor.start_recording()


    # Pass the get_current_status method as a parameter
    threading.Thread(target=lambda: sensor.record_events_with_auxiliary_data(
        "event_data", 
        motor_system.get_current_status, 
        recording_duration_sec=300)).start()

    # Move CNC Y-axis
    # motor_system.move(1, 30, -1, 20)  # Move Y axis -50
    # motor_system.move(1, 60, 1, 20)   # Move Y axis +50
    # motor_system.move(1, 30, -1, 20)  # Move Y axis -50

    motor_system.move(1, 30, -1, 100)  # Move Y axis -50
    motor_system.move(1, 60, 1, 100)   # Move Y axis +50
    motor_system.move(1, 30, -1, 100)  # Move Y axis -50
    # Start the CNC movement sequence in a new thread

    


    # threading.Thread(target=cnc_movement_sequence).start()

    # for _ in range(100):
    #     current_status = motor_system.get_current_status()  # Assuming this method exists and returns the status
    #     print(f"Current machine status: {current_status}")
    #     time.sleep(0.1)

    # sensor.record_events_with_auxiliary_data("event_data", motor_system.get_current_status)

    time.sleep(15)


    # Stop recording
    sensor.stop_recording()

    time.sleep(3)

    # Turn off LED
    led_control.led_off()

   

    # Print positions, events, and save to CSV
    # save_events_to_csv()

    # Cleanup
    led_control.close()
    motor_system.close_device()

def save_events_to_csv():
    # Placeholder for the function to save event data to CSV
    # You'll need to modify the event_sensor.py to add events to `event_data`
    # Here's a simple implementation assuming `event_data` is a list of dictionaries
    with open('event_data.csv', 'w', newline='') as file:
        fieldnames = ['timestamp', 'x', 'y', 'polarity', 'y_axis_position']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in event_data:
            writer.writerow(event)

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/start", StartSequenceHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server running on http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
