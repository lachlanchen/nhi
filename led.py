import serial
import time

class ArduinoLED:
    def __init__(self, port, baud_rate=9600):
        """
        Initializes the connection to the Arduino.
        :param port: The serial port Arduino is connected to (e.g., 'COM4' or '/dev/ttyACM0').
        :param baud_rate: Baud rate for serial communication (default: 9600).
        """
        self.connection = serial.Serial(port, baud_rate)
        time.sleep(2)  # Wait for the connection to be established

    def led_on(self):
        """Turns the LED on."""
        self.connection.write(b'1\n')

    def led_off(self):
        """Turns the LED off."""
        self.connection.write(b'0\n')

    def close(self):
        """Closes the serial connection."""
        self.connection.close()

if __name__ == "__main__":
    # Replace 'COM4' with the actual port your Arduino is connected to
    led_control = ArduinoLED(port='COM4')
    
    try:
        while True:
            print("LED on")
            led_control.led_on()
            time.sleep(2)  # Wait for 2 seconds
            
            print("LED off")
            led_control.led_off()
            time.sleep(2)  # Wait for 2 seconds
    except KeyboardInterrupt:
        print("Exiting the program...")
    finally:
        led_control.close()  # Ensure the serial connection is closed on exit
