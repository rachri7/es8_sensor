#include <Arduino_LSM6DS3.h>

/*
  Extended IMU class for setting LSM6DS3 sample rate.
*/
class IMUExtended : public LSM6DS3Class {
public:
  IMUExtended(TwoWire& wire, uint8_t slaveAddress)
    : LSM6DS3Class{ wire, slaveAddress } {
  }

  void SetAccGyroRate13Hz() {
    writeRegister(0x10, 0b00011000);
    writeRegister(0x11, 0b00011100);
  }

  void SetAccGyroRate26Hz() {
    writeRegister(0x10, 0b00101000);
    writeRegister(0x11, 0b00101100);
  }

  void SetAccGyroRate52Hz() {
    writeRegister(0x10, 0b00111000);
    writeRegister(0x11, 0b00111100);
  }

  void SetAccGyroRate104Hz() {
    writeRegister(0x10, 0b01001000);
    writeRegister(0x11, 0b01001100);
  }
};

IMUExtended myIMU{ Wire, LSM6DS3_ADDRESS };

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!myIMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  myIMU.SetAccGyroRate13Hz();

}

void loop() {
  float ax = 0.0;
  float ay = 0.0;
  float az = 0.0;

 if (myIMU.accelerationAvailable()) {
    myIMU.readAcceleration(ax, ay, az);

    Serial.print(ax, 8);
    Serial.print(",");

    Serial.print(ay, 8);
    Serial.print(",");

    Serial.println(az, 8);

  }
}
