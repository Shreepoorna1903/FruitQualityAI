#include <Servo.h>

// Pins
const int SERVO_PIN = 9;     // PWM capable pin
const int MOTOR_PIN = 7;     // Relay or motor driver enable

Servo gate;
int freshAngle = 90;
int rottenAngle = 180;

void setup() {
  Serial.begin(9600);
  gate.attach(SERVO_PIN);
  pinMode(MOTOR_PIN, OUTPUT);
  gate.write(freshAngle);    // default: allow belt
  digitalWrite(MOTOR_PIN, HIGH); // motor on (adjust to your wiring)
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '0') {
      // Fresh: keep gate open
      gate.write(freshAngle);
      digitalWrite(MOTOR_PIN, HIGH);
    } else if (c == '1') {
      // Rotten: close gate briefly to reject
      gate.write(rottenAngle);
      digitalWrite(MOTOR_PIN, HIGH);
      delay(800); // adjust to your belt speed
      gate.write(freshAngle);
    }
  }
}