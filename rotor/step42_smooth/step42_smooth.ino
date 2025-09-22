/**
  *************************************************************************************************
  * @file           step
  * @note           Stepper Motor Control with Degrees and RPM, including Limit Switches and FindMiddle
  *************************************************************************************************
*/

// Microstepping settings
#define MICRO 32  // Microstepping factor

#include <avr/interrupt.h>
#include "PinChangeInt.h"
#include "OLED12864.h"

#define X_LIM_PIN 9   // X Limit Switch Pin
#define Y_LIM_PIN 10  // Y Limit Switch Pin

// Acceleration parameters
#define ACCELERATION_STEPS 10  // Number of steps to accelerate
#define DECELERATION_STEPS 10  // Number of steps to decelerate
#define MIN_STEP_DELAY 500        // Minimum delay between steps in microseconds (maximum speed)
#define MAX_STEP_DELAY 20000       // Maximum delay between steps in microseconds (minimum speed)

// Custom map function for floating point
float map_float(long x, long in_min, long in_max, float out_min, float out_max) {
    return (float)(x - in_min) * (out_max - out_min) / (float)(in_max - in_min) + out_min;
}

OLED12864 oled12864;

int ADD_PIN = 11;
int SUB_PIN = 7;
int SET_PIN = 4;

int EN_PIN = 8;    // Enable Pin
int DIR_PIN = 5;   // Direction Pin
int STEP_PIN = 2;  // Step Pulse Pin

int flag_chang = 0;
String turn_tmp = "POS";

// Define motion control variables
volatile bool clockwiseMotionAllowed = true;
volatile bool counterclockwiseMotionAllowed = true;

// Step count variable
volatile long stepCount = 0;  // Variable to track steps moved

// Function prototypes
void move_motor_degrees(float degrees, float rpm);
void findMiddle(float rpm);

void x_limit_reached() {
    // Disable clockwise motion but keep motor enabled
    clockwiseMotionAllowed = false;
    counterclockwiseMotionAllowed = true;
    Serial.println("X Limit Reached - Stopped");
}

void y_limit_reached() {
    // Disable counterclockwise motion but keep motor enabled
    counterclockwiseMotionAllowed = false;
    clockwiseMotionAllowed = true;
    Serial.println("Y Limit Reached - Stopped");
}

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
    } else if (flag_set == 1) {
        digitalWrite(EN_PIN, HIGH);
        digitalWrite(DIR_PIN, HIGH);
        Serial.println("反转");
        turn_tmp = "INV";
        counterclockwiseMotionAllowed = true; // Allow counterclockwise motion
    } else if (flag_set == 2) {
        Serial.println("停转");
        turn_tmp = "STOP";
        // Stop motion but keep motor enabled
        digitalWrite(STEP_PIN, LOW);  // Stop stepping
        clockwiseMotionAllowed = false;
        counterclockwiseMotionAllowed = false;
    }
}

void add_key_deal() {
    flag_chang = 1;
    // Adjust speed or other parameters if needed
    Serial.println("ADD KEY");
}

void sub_key_deal() {
    flag_chang = 1;
    // Adjust speed or other parameters if needed
    Serial.println("SUB KEY");
}

void setup() {
    Serial.begin(9600);

    pinMode(X_LIM_PIN, INPUT_PULLUP);
    pinMode(Y_LIM_PIN, INPUT_PULLUP);

    oled12864.init();  // Initialize OLED display
    oled12864.clear();

    pinMode(STEP_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT);
    pinMode(EN_PIN, OUTPUT);
    digitalWrite(STEP_PIN, LOW);
    digitalWrite(DIR_PIN, LOW);
    digitalWrite(EN_PIN, HIGH); // Keep motor enabled

    pinMode(SET_PIN, INPUT_PULLUP);
    pinMode(ADD_PIN, INPUT_PULLUP);
    pinMode(SUB_PIN, INPUT_PULLUP);

    attachPinChangeInterrupt(SET_PIN, set_key_deal, RISING);
    attachPinChangeInterrupt(ADD_PIN, add_key_deal, RISING);
    attachPinChangeInterrupt(SUB_PIN, sub_key_deal, RISING);

    // Attach interrupts for limit switches
    attachPinChangeInterrupt(X_LIM_PIN, x_limit_reached, FALLING);
    attachPinChangeInterrupt(Y_LIM_PIN, y_limit_reached, FALLING);

    Serial.println("Starting...");

    oled12864.show(0, 0, "SPEED:");
    oled12864.show(0, 6, "RPM"); // Display RPM
    oled12864.show(1, 0, "TURN :");
    oled12864.show(1, 6, turn_tmp);
    oled12864.show(2, 0, "MICRO:");
    oled12864.show(2, 6, MICRO);
    oled12864.display();



    // // Example: Move motor 360 degrees at 1 RPM
    // move_motor_degrees(360.0, 10.0);
    // move_motor_degrees(-360.0, 10.0);
    // move_motor_degrees(360.0, 10.0);
    // move_motor_degrees(-360.0, 10.0);
    // move_motor_degrees(360.0, 10.0);
    // move_motor_degrees(-360.0, 10.0);

    // Call findMiddle function with 10 RPM
    // findMiddle(20.0);

    // move_motor_degrees(-10.0, 1.0);

    // move_motor_degrees(-20.0, 1.0);
    // delayMicroseconds(1000000);
    // move_motor_degrees(20.0, 1.0);
    // delayMicroseconds(1000000);
    // move_motor_degrees(-20.0, 1.0);
    // delayMicroseconds(1000000);
    // move_motor_degrees(20.0, 1.0);
    // delayMicroseconds(1000000);
    // move_motor_degrees(-20.0, 1.0);
    // delayMicroseconds(1000000);
    // move_motor_degrees(20.0, 1.0);
    // delayMicroseconds(1000000);
    
}

void loop() {
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

    // Call findMiddle function with 10 RPM
    findMiddle(20.0);

    float motion_speed = 3;
    float angle = 15;

    move_motor_degrees(-10.0, motion_speed);

    delayMicroseconds(1000000);

    move_motor_degrees(-angle, motion_speed);
    delayMicroseconds(1000000);
    move_motor_degrees(angle, motion_speed);
    delayMicroseconds(1000000);
    move_motor_degrees(-angle, motion_speed);
    delayMicroseconds(1000000);
    move_motor_degrees(angle,motion_speed);
    delayMicroseconds(1000000);
    move_motor_degrees(-angle, motion_speed);
    delayMicroseconds(1000000);
    move_motor_degrees(angle, motion_speed);
    delayMicroseconds(1000000);
}

// // Function to move the motor by a specific number of degrees at a specified RPM
// void move_motor_degrees(float degrees, float rpm) {
//     // Calculate total steps for the given degrees
//     float steps_per_rev = 200 * MICRO; // 200 steps/rev * microstepping
//     long steps_to_move = (long)((abs(degrees) / 360.0) * steps_per_rev);

//     // Calculate delay between steps in microseconds
//     float step_delay_us = (60.0 * 1000000.0) / (rpm * steps_per_rev);

//     int direction = (degrees > 0) ? LOW : HIGH; // Determine direction
//     digitalWrite(DIR_PIN, direction);

//     // Movement loop
//     for (long i = 0; i < abs(steps_to_move); i++) {
//         // Check for limit switches
//         if ((direction == LOW && !clockwiseMotionAllowed) ||
//             (direction == HIGH && !counterclockwiseMotionAllowed)) {
//             Serial.println("Movement stopped due to limit switch");
//             break; // Stop movement if limit switch is triggered
//         }

//         digitalWrite(STEP_PIN, HIGH);
//         delayMicroseconds(step_delay_us / 2);
//         digitalWrite(STEP_PIN, LOW);
//         delayMicroseconds(step_delay_us / 2);

//         // Update stepCount
//         if (direction == LOW) {
//             stepCount++;
//         } else {
//             stepCount--;
//         }
//     }

//     // Movement complete
//     Serial.println("Movement complete");
// }

// Modified move_motor_degrees function with acceleration
void move_motor_degrees(float degrees, float rpm) {
    // Calculate total steps for the given degrees
    float steps_per_rev = 200.0 * MICRO; // 200 steps/rev * microstepping
    long steps_to_move = (long)((abs(degrees) / 360.0) * steps_per_rev);

    // Calculate target step delay based on RPM
    float target_step_delay = (60.0 * 1000000.0) / (rpm * steps_per_rev); // in microseconds

    // Direction of movement
    int direction = (degrees > 0) ? LOW : HIGH;
    digitalWrite(DIR_PIN, direction);

    // Determine number of steps for acceleration and deceleration
    long accel_steps = min(ACCELERATION_STEPS, steps_to_move / 2);
    long decel_steps = min(DECELERATION_STEPS, steps_to_move / 2);
    long constant_steps = steps_to_move - accel_steps - decel_steps;

    // Initialize step delay
    float step_delay = MAX_STEP_DELAY;

    // Function to map step number to delay (simple linear acceleration/deceleration)
    auto calculate_step_delay = [&](long current_step, long total_accel_steps, long total_decel_steps, long steps_to_move) -> float {
        if (current_step < total_accel_steps) {
            // Accelerate
            return map_float(current_step, 0, total_accel_steps, MAX_STEP_DELAY, target_step_delay);
        } else if (current_step >= (steps_to_move - total_decel_steps)) {
            // Decelerate
            return map_float(steps_to_move - current_step, 0, total_decel_steps, target_step_delay, MAX_STEP_DELAY);
        } else {
            // Constant speed
            return target_step_delay;
        }
    };

    // Movement loop with acceleration and deceleration
    for (long i = 0; i < steps_to_move; i++) {
        // Check for limit switches
        if ((direction == LOW && !clockwiseMotionAllowed) ||
            (direction == HIGH && !counterclockwiseMotionAllowed)) {
            Serial.println("Movement stopped due to limit switch");
            break; // Stop movement if limit switch is triggered
        }

        // Calculate current step delay
        step_delay = calculate_step_delay(i, accel_steps, decel_steps, steps_to_move);

        // Execute step
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(step_delay / 2);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(step_delay / 2);

        // Update stepCount
        if (direction == LOW) {
            stepCount++;
        } else {
            stepCount--;
        }
    }

    // Movement complete
    Serial.println("Movement complete");
}


// // Function to move the motor by a specific number of degrees at a specified RPM
// void move_motor_degrees(float degrees, float rpm) {
//     // Calculate total steps for the given degrees
//     float steps_per_rev = 200 * MICRO; // 200 steps/rev * microstepping
//     long steps_to_move = (long)((abs(degrees) / 360.0) * steps_per_rev);

//     // Calculate delay between steps in microseconds
//     float step_delay_us = (60.0 * 1000000.0) / (rpm * steps_per_rev);

//     int direction = (degrees > 0) ? LOW : HIGH; // Determine direction
//     digitalWrite(DIR_PIN, direction);

//     // Movement loop
//     for (long i = 0; i < abs(steps_to_move); i++) {
//         // Check for limit switches
//         if ((direction == LOW && !clockwiseMotionAllowed) ||
//             (direction == HIGH && !counterclockwiseMotionAllowed)) {
//             Serial.println("Movement stopped due to limit switch");
//             break; // Stop movement if limit switch is triggered
//         }

//         digitalWrite(STEP_PIN, HIGH);
//         delayMicroseconds(step_delay_us / 2);
//         digitalWrite(STEP_PIN, LOW);
//         delayMicroseconds(step_delay_us / 2);
//     }

//     // Movement complete
//     Serial.println("Movement complete");
// }

// Function to find the middle position between the two limits
void findMiddle(float rpm) {
    long clockwiseSteps = 0;
    long counterclockwiseSteps = 0;

    // Reset step count
    stepCount = 0;
    clockwiseMotionAllowed = true;
    counterclockwiseMotionAllowed = true;

    // Move to clockwise limit
    // Serial.println("Moving to clockwise limit...");
    move_motor_degrees(360.0, rpm);  // Move in positive direction until limit switch stops the motor
    clockwiseSteps = stepCount;
    // Serial.print("Clockwise steps: ");
    // Serial.println(clockwiseSteps);

    // Reset step count
    stepCount = 0;
    clockwiseMotionAllowed = true;
    counterclockwiseMotionAllowed = true;

    // Move to counterclockwise limit
    // Serial.println("Moving to counterclockwise limit...");
    move_motor_degrees(-360.0, rpm);  // Move in negative direction until limit switch stops the motor
    counterclockwiseSteps = stepCount; // This will be negative
    // Serial.print("Counterclockwise steps: ");
    // Serial.println(counterclockwiseSteps);

    // Calculate total steps between limits
    long totalSteps = abs(counterclockwiseSteps);
    long middleSteps = totalSteps / 2;

    // Serial.print("Total steps between limits: ");
    // Serial.println(totalSteps);
    // Serial.print("Steps to middle: ");
    // Serial.println(middleSteps);

    // Move back to the middle position
    stepCount = 0;
    clockwiseMotionAllowed = true;
    counterclockwiseMotionAllowed = true;

    if (middleSteps > 0) {
        // Move towards clockwise direction
        // Serial.println("Moving to middle position...");
        float degreesToMiddle = (middleSteps * 360.0) / (200 * MICRO);
        move_motor_degrees(degreesToMiddle, rpm); // Convert steps back to degrees
    }

    // Serial.println("Middle position reached.");
}
