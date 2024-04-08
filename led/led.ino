const int ledPin = 9; // LED connected to pin 9

void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH); // Ensure LED is off initially
  Serial.begin(9600); // Start serial communication at 9600 baud rate
}

void loop() {
  if (Serial.available() > 0) { // Check if data is available to read
    char command = Serial.read(); // Read the incoming byte
    if (command == '1') {
      digitalWrite(ledPin, LOW); // Turn LED on
    } else if (command == '0') {
      digitalWrite(ledPin, HIGH); // Turn LED off
    }
  }
}
