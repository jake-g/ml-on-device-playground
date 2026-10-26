// MotionSynth: Record plus the gyro, for Motion Synth on the XIAO nRF52840 Sense.
// Serial protocol (115200 baud), every 100 Hz sample, gyro line first:
//   GYR:gx,gy,gz              deg/s, 1 decimal (+-2000 dps default)
//   IMU:ax,ay,az              g, 2 decimals (+-16 g default), same as Record
//   INFO:motion_synth gyro=1  once at startup
// Hosts that only parse IMU: lines skip GYR: and INFO: lines.
#include <LSM6DS3.h>
#include <Wire.h>

namespace {

constexpr float kSampleRateHz = 100.0f;
constexpr unsigned long kSampleIntervalMs =
    (unsigned long)(1000.0f / kSampleRateHz);
// Stream anyway if no host opens the port within this time.
constexpr unsigned long kSerialWaitMs = 4000;

// Library defaults: accel +-16 g, gyro +-2000 dps, both at 416 Hz ODR.
LSM6DS3 imu(I2C_MODE, 0x6A);  // I2C device address 0x6A

unsigned long last_sample_ms = 0;

}  // namespace

void setup() {
  Serial.begin(115200);
  unsigned long start = millis();
  while (!Serial && millis() - start < kSerialWaitMs) {
  }

  if (imu.begin() != 0) {
    Serial.println("ERR:imu init failed");
  }
  Serial.println("INFO:motion_synth gyro=1");
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_ms < kSampleIntervalMs) return;
  last_sample_ms = now;

  // Read all six axes before printing so both lines describe one sample.
  float ax = imu.readFloatAccelX();
  float ay = imu.readFloatAccelY();
  float az = imu.readFloatAccelZ();
  float gx = imu.readFloatGyroX();
  float gy = imu.readFloatGyroY();
  float gz = imu.readFloatGyroZ();

  Serial.print("GYR:");
  Serial.print(gx, 1);
  Serial.print(",");
  Serial.print(gy, 1);
  Serial.print(",");
  Serial.print(gz, 1);
  Serial.print("\n");

  Serial.print("IMU:");
  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print("\n");
}
