/**
  *************************************************************************************************
  * @file           step
  * @author         张文超
  * @version        V2.0
  * @note    ´      步进电机基础例程
  * @note    ´      深圳市沁和智能科技有限公司，版权所有，严禁私自传播，否则必究法律责任
  *************************************************************************************************
  */

//ms1   ms2    ms3
//L       L        L       整步(没有细分)
//H      L         L      1/2(2细分)
//L      H        L       1/4(4细分)
//H      H        L       1/8(8细分)
//H      H        H      1/16(16细分)
  
#include <avr/interrupt.h>
#include "PinChangeInt.h"
#include "OLED12864.h"

#define MICRO 32

#define X_LIM_PIN 9  // X Limit Switch Pin
#define Y_LIM_PIN 10 // Y Limit Switch Pin

OLED12864 oled12864;

int ADD_PIN = 11;
int SUB_PIN = 7;
int SET_PIN = 4;

int EN_PIN = 8;    //使能引脚
int DIR_PIN = 5;   //方向引脚
int STEP_PIN = 2;  //脉冲引脚

int flag_chang = 0;
String turn_tmp = "POS";

// Define motion control variables
volatile bool clockwiseMotionAllowed = true;
volatile bool counterclockwiseMotionAllowed = true;

unsigned char tcnt2 = 100; 

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


void set_key_deal(){
    static int flag_set = 0;
    Serial.println("SET KEY");
    flag_chang = 1;
    
    if(++flag_set >= 3) flag_set = 0;

    if(flag_set == 0){  
      digitalWrite(EN_PIN, HIGH);   
      digitalWrite(DIR_PIN, LOW);    
      Serial.println("正转");
      turn_tmp = "POS";
      clockwiseMotionAllowed = true; // Allow clockwise motion
    }else if(flag_set == 1){
      digitalWrite(EN_PIN, HIGH);   
      digitalWrite(DIR_PIN, HIGH);    
      Serial.println("反转");
      turn_tmp = "INV";
      counterclockwiseMotionAllowed = true; // Allow counterclockwise motion
    }else if(flag_set == 2){
      Serial.println("停转");
      turn_tmp = "STOP";
      // Stop motion but keep motor enabled
      digitalWrite(STEP_PIN, LOW);  // Stop stepping
      clockwiseMotionAllowed = false;
      counterclockwiseMotionAllowed = false;
    }    
}

void add_key_deal(){
    flag_chang = 1;
    if(tcnt2 < 240){ tcnt2 = tcnt2 + 10; }
    Serial.println("ADD KEY");
}

void sub_key_deal(){
    flag_chang = 1;
    if(tcnt2 >= 10){ tcnt2 = tcnt2 - 10; }
    Serial.println("SUB KEY");
}


ISR(TIMER2_OVF_vect) {
  TCNT2 = tcnt2;
  // Only step if motion is allowed in the set direction
  if ((digitalRead(DIR_PIN) == LOW && clockwiseMotionAllowed) || 
      (digitalRead(DIR_PIN) == HIGH && counterclockwiseMotionAllowed)) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(1000);
    digitalWrite(STEP_PIN, LOW);
  }
}

void move_motor_degrees(int degrees) {
  int steps_per_rev = 200 * MICRO; // 200 steps per rev for 1.8-degree stepper, multiplied by microstepping
  int steps_to_move = (steps_per_rev / 360.0) * degrees;
  
  for (int i = 0; i < steps_to_move; i++) {
    // Stop motion if not allowed
    if ((digitalRead(DIR_PIN) == LOW && !clockwiseMotionAllowed) ||
        (digitalRead(DIR_PIN) == HIGH && !counterclockwiseMotionAllowed)) {
      break;  // Stop if limit is reached
    }
    
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(1000000);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(1000000);
  }
}

void move_motor_sequence() {
  for (int i = 0; i < 5; i++) {
    // Move 90 degrees positively if allowed
    if (clockwiseMotionAllowed) {
      digitalWrite(DIR_PIN, LOW);
      // move_motor_degrees(90);
    }
    delay(1000); // Wait for 1 second

    // Move 90 degrees negatively if allowed
    if (counterclockwiseMotionAllowed) {
      digitalWrite(DIR_PIN, HIGH);
      // move_motor_degrees(90);
    }
    delay(1000); // Wait for 1 second
  }
}

void setup() {
  Serial.begin(9600);

  pinMode(X_LIM_PIN, INPUT_PULLUP);
  pinMode(Y_LIM_PIN, INPUT_PULLUP);

  oled12864.init();  // initialize with the I2C addr 0x3D (for the 128x64)
  oled12864.clear();
  
  pinMode(STEP_PIN, OUTPUT); 
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
  digitalWrite(EN_PIN, HIGH); // Keep motor enabled

  digitalWrite(SET_PIN, LOW);
  digitalWrite(ADD_PIN, LOW);
  digitalWrite(SUB_PIN, LOW);

  attachPinChangeInterrupt(SET_PIN, set_key_deal, RISING);
  attachPinChangeInterrupt(ADD_PIN, add_key_deal, RISING);
  attachPinChangeInterrupt(SUB_PIN, sub_key_deal, RISING);

  // Attach interrupts for limit switches
  attachPinChangeInterrupt(X_LIM_PIN, x_limit_reached, FALLING);
  attachPinChangeInterrupt(Y_LIM_PIN, y_limit_reached, FALLING);
  
  TIMSK2 &= ~(1<<TOIE2);
  TCCR2A &= ~((1<<WGM21) | (1<<WGM20));   // Normal mode
  TCCR2B &= ~(1<<WGM22);
  ASSR &= ~(1<<AS2);     // Disable asynchronous mode
  TIMSK2 &= ~(1<<OCIE2A);     // Disable compare match interrupt

  TCCR2B |= ( (1<<CS22) );  // Prescaler of 64
  TCCR2B &= ~( (1<<CS21)| (1<<CS20) );

  TCNT2 = tcnt2;     // Initial value
  TIMSK2 |= (1<<TOIE2);  // Enable overflow interrupt

  Serial.println("Starting...");

  oled12864.show(0,0,"SPEED:");
  oled12864.show(0,6,tcnt2);
  oled12864.show(1,0,"TURN :");
  oled12864.show(1,6,turn_tmp);
  oled12864.show(2,0,"MICRO:");
  oled12864.show(2,6,MICRO);
  oled12864.display(); 

  // Start the sequence of movements
  move_motor_sequence();
}

void loop() {
    if (flag_chang == 1) {
        flag_chang = 0;    
        oled12864.clear();
        oled12864.show(0,0,"SPEED:");
        oled12864.show(0,6,tcnt2);
        oled12864.show(1,0,"TURN :");
        oled12864.show(1,6,turn_tmp);
        oled12864.show(2,0,"MICRO:");
        oled12864.show(2,6,MICRO);
        oled12864.display();
    }
}
