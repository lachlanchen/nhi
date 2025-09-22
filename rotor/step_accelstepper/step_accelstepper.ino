/**
  *************************************************************************************************
  * @file           step
  * @note           Stepper Motor Control with Degrees and RPM, including Limit Switches and FindMiddle using AccelStepper
  *************************************************************************************************
*/

// Microstepping settings
#define MICRO 16  // Microstepping factor

#include <avr/interrupt.h>
#include "PinChangeInt.h"
#include "OLED12864.h"
#include <AccelStepper.h>  // Include AccelStepper library

#define X_LIM_PIN 9   // X Limit Switch Pin
#define Y_LIM_PIN 10  // Y Limit Switch Pin

OLED12864 oled12864;

const int ADD_PIN = 11;
const int SUB_PIN = 7;
const int SET_PIN = 4;

const int EN_PIN = 8;    // Enable Pin
const int DIR_PIN = 5;   // Direction Pin
const int STEP_PIN = 2;  // Step Pulse Pin

volatile int flag_chang = 0;
String turn_tmp = "POS";

// Define motion control variables
volatile bool clockwiseMotionAllowed = true;
volatile bool counterclockwiseMotionAllowed = true;

// Create an instance of AccelStepper (using DRIVER interface: STEP and DIR)
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// Function prototypes
void move_motor_degrees(float degrees, float rpm);
void findMiddle(float rpm);

// Interrupt service routine for X limit switch
void x_limit_reached() {
    // Disable clockwise motion but keep motor enabled
    clockwiseMotionAllowed = false;
    counterclockwiseMotionAllowed = true;
    Serial.println("X Limit Reached - Stopped");

    // Stop the stepper movement by setting target to current position
    stepper.stop();
}

// Interrupt service routine for Y limit switch
void y_limit_reached() {
    // Disable counterclockwise motion but keep motor enabled
    counterclockwiseMotionAllowed = false;
    clockwiseMotionAllowed = true;
    Serial.println("Y Limit Reached - Stopped");

    // Stop the stepper movement by setting target to current position
    stepper.stop();
}

// ISR for SET button
void set_key_deal() {
    static int flag_set = 0;
    Serial.println("SET KEY");
    flag_chang = 1;

    if (++flag_set >= 3) flag_set = 0;

    if (flag_set == 0) {
        digitalWrite(EN_PIN, HIGH);
        digitalWrite(DIR_PIN, LOW);
        Serial.println("正转");
        turn_tmp = "POS";
        clockwiseMotionAllowed = true; // Allow clockwise motion
    }
    else if (flag_set == 1) {
        digitalWrite(EN_PIN, HIGH);
        digitalWrite(DIR_PIN, HIGH);
        Serial.println("反转");
        turn_tmp = "INV";
        counterclockwiseMotionAllowed = true; // Allow counterclockwise motion
    }
    else if (flag_set == 2) {
        Serial.println("停转");
        turn_tmp = "STOP";
        // Stop motion but keep motor enabled
        stepper.stop();
        clockwiseMotionAllowed = false;
        counterclockwiseMotionAllowed = false;
    }
}

// ISR for ADD button
void add_key_deal() {
    flag_chang = 1;
    // Adjust speed or other parameters if needed
    Serial.println("ADD KEY");
}

// ISR for SUB button
void sub_key_deal() {
    flag_chang = 1;
    // Adjust speed or other parameters if needed
    Serial.println("SUB KEY");
}

void setup() {
    Serial.begin(9600);

    // Initialize limit switch pins
    pinMode(X_LIM_PIN, INPUT_PULLUP);
    pinMode(Y_LIM_PIN, INPUT_PULLUP);

    // Initialize OLED display
    oled12864.init();
    oled12864.clear();

    // Initialize motor control pins
    pinMode(EN_PIN, OUTPUT);
    digitalWrite(EN_PIN, HIGH); // Keep motor enabled

    // Initialize buttons
    pinMode(SET_PIN, INPUT_PULLUP);
    pinMode(ADD_PIN, INPUT_PULLUP);
    pinMode(SUB_PIN, INPUT_PULLUP);

    // Attach interrupts for buttons
    attachPinChangeInterrupt(SET_PIN, set_key_deal, RISING);
    attachPinChangeInterrupt(ADD_PIN, add_key_deal, RISING);
    attachPinChangeInterrupt(SUB_PIN, sub_key_deal, RISING);

    // Attach interrupts for limit switches
    attachPinChangeInterrupt(X_LIM_PIN, x_limit_reached, FALLING);
    attachPinChangeInterrupt(Y_LIM_PIN, y_limit_reached, FALLING);

    Serial.println("Starting...");

    // Display initial information on OLED
    oled12864.show(0, 0, "SPEED:");
    oled12864.show(0, 6, "RPM"); // Display RPM
    oled12864.show(1, 0, "TURN :");
    oled12864.show(1, 6, turn_tmp);
    oled12864.show(2, 0, "MICRO:");
    oled12864.show(2, 6, MICRO);
    oled12864.display();

    // Initialize stepper motor
    // Set maximum speed and acceleration based on initial RPM
    float initial_rpm = 10.0;
    float steps_per_rev = 200.0 * MICRO;
    float steps_per_min = initial_rpm * steps_per_rev;
    float steps_per_sec = steps_per_min / 60.0;

    stepper.setMaxSpeed(steps_per_sec);       // steps per second
    stepper.setAcceleration(steps_per_sec * 2); // steps per second^2

    // Call findMiddle function with initial RPM
    findMiddle(initial_rpm);

    // You can initialize other movement commands here if needed
}

void loop() {
    // Handle OLED display updates
    if (flag_chang == 1) {
        flag_chang = 0;
        oled12864.clear();
        oled12864.show(0, 0, "SPEED:");
        oled12864.show(0, 6, "RPM"); // Update display if needed
        oled12864.show(1, 0, "TURN :");
        oled12864.show(1, 6, turn_tmp);
        oled12864.show(2, 0, "MICRO:");
        oled12864.show(2, 6, MICRO);
        oled12864.display();
    }

    // Handle stepper movements
    // Check if stepper is not moving
    if (stepper.distanceToGo() == 0) {
        // Perform movement commands

        // Example sequence: move -10 degrees at 1 RPM
        move_motor_degrees(-10.0, 1.0);

        // Move -20 degrees at 1 RPM
        move_motor_degrees(-20.0, 1.0);
        delay(1000); // Wait for 1 second

        // Move +20 degrees at 1 RPM
        move_motor_degrees(20.0, 1.0);
        delay(1000); // Wait for 1 second

        // Move -20 degrees at 1 RPM
        move_motor_degrees(-20.0, 1.0);
        delay(1000); // Wait for 1 second

        // Move +20 degrees at 1 RPM
        move_motor_degrees(20.0, 1.0);
        delay(1000); // Wait for 1 second

        // Move -20 degrees at 1 RPM
        move_motor_degrees(-20.0, 1.0);
        delay(1000); // Wait for 1 second

        // Move +20 degrees at 1 RPM
        move_motor_degrees(20.0, 1.0);
        delay(1000); // Wait for 1 second
    }

    // Run stepper (necessary if using non-blocking movement commands)
    stepper.run();
}

// Function to move the motor by a specific number of degrees at a specified RPM using AccelStepper
void move_motor_degrees(float degrees, float rpm) {
    // Calculate total steps for the given degrees
    float steps_per_rev = 200.0 * MICRO; // 200 steps/rev * microstepping
    long steps_to_move = (long)((abs(degrees) / 360.0) * steps_per_rev);

    // Calculate stepper speed in steps per second based on RPM
    float steps_per_min = rpm * steps_per_rev;
    float steps_per_sec = steps_per_min / 60.0;

    // Set stepper speed and acceleration
    stepper.setMaxSpeed(steps_per_sec);       // steps per second
    stepper.setAcceleration(steps_per_sec * 2); // steps per second^2

    // Determine direction and set target position
    long current_position = stepper.currentPosition();
    if (degrees > 0) {
        stepper.moveTo(current_position + steps_to_move);
    }
    else {
        stepper.moveTo(current_position - steps_to_move);
    }

    // Wait until movement is complete
    while (stepper.distanceToGo() != 0) {
        stepper.run();
    }

    // Movement complete
    Serial.println("Movement complete");
}

// Function to find the middle position between the two limits using AccelStepper
void findMiddle(float rpm) {
    long clockwiseSteps = 0;
    long counterclockwiseSteps = 0;

    // Reset stepper position
    stepper.setCurrentPosition(0);

    // Move to clockwise limit
    Serial.println("Moving to clockwise limit...");
    // Calculate speed based on RPM
    float steps_per_rev = 200.0 * MICRO;
    float steps_per_min = rpm * steps_per_rev;
    float steps_per_sec = steps_per_min / 60.0;

    stepper.setMaxSpeed(steps_per_sec);
    stepper.setAcceleration(steps_per_sec * 2);

    // Move in positive direction until limit switch is hit
    stepper.moveTo(stepper.currentPosition() + 1000000); // Large target to move towards limit

    // Run stepper until limit is reached (interrupted by ISR)
    while (stepper.distanceToGo() > 0 && clockwiseMotionAllowed) {
        stepper.run();
    }
    clockwiseSteps = stepper.currentPosition();
    Serial.print("Clockwise steps: ");
    Serial.println(clockwiseSteps);

    // Reset stepper position
    stepper.setCurrentPosition(0);

    // Move to counterclockwise limit
    Serial.println("Moving to counterclockwise limit...");
    stepper.moveTo(stepper.currentPosition() - 1000000); // Large target to move towards limit

    // Run stepper until limit is reached (interrupted by ISR)
    while (stepper.distanceToGo() < 0 && counterclockwiseMotionAllowed) {
        stepper.run();
    }
    counterclockwiseSteps = stepper.currentPosition();
    Serial.print("Counterclockwise steps: ");
    Serial.println(counterclockwiseSteps);

    // Calculate total steps between limits
    long totalSteps = abs(counterclockwiseSteps);
    long middleSteps = totalSteps / 2;

    Serial.print("Total steps between limits: ");
    Serial.println(totalSteps);
    Serial.print("Steps to middle: ");
    Serial.println(middleSteps);

    // Move back to the middle position
    Serial.println("Moving to middle position...");
    stepper.moveTo(middleSteps);
    while (stepper.distanceToGo() != 0) {
        stepper.run();
    }

    Serial.println("Middle position reached.");
}
