/*
  MLX90393 Magnetometer Example Code
  By: Nathan Seidle
  SparkFun Electronics
  Date: February 6th, 2017
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Read the mag fields on three XYZ axis

  Hardware Connections (Breakoutboard to Arduino):
  3.3V = 3.3V
  GND = GND
  SDA = A4
  SCL = A5

  Serial.print it out at 9600 baud to serial monitor.
*/

#include <Wire.h>
#include <MLX90393.h> //From https://github.com/tedyapo/arduino-MLX90393 by Theodore Yapo

MLX90393 mlx_left;
MLX90393 mlx_right;
MLX90393 ref_mlx;
MLX90393::txyz data_left; //Create a structure, called data, of four floats (t, x, y, and z)
MLX90393::txyz data_right; //Create a structure, called data, of four floats (t, x, y, and z)
MLX90393::txyz ref_data;

long baudRate = 115200;
int delayTime = 25; //milliseconds
uint8_t gainSel = 7; //0x0-0x7
uint8_t resolution[] = {0, 0, 0}; //
uint8_t overSampling = 0;//0,3,4,7
uint8_t digitalFilt = 4; //0x0-0x7


void setup()
{
  Serial.begin(baudRate);
  //Serial.println("MLX90393 Setting Up...");
  Wire.begin();
  //Connect to sensor with I2C address jumpers set: A1 = 1, A0 = 0
  //Use DRDY pin connected to A3
  //Returns byte containing status bytes
  byte status = ref_mlx.begin(1, 1, -1, Wire, 1);
  //Report status from configuration
  Serial.print("Start status: 0x");
  if(status < 0x10) Serial.print("0"); //Pretty output
  Serial.println(status, BIN);

  status = mlx_left.begin(0, 0, -1, Wire, 1);
  Serial.print("Start status: 0x");
  if(status < 0x10) Serial.print("0"); //Pretty output
  Serial.println(status, BIN);

  status = mlx_right.begin(1, 0, -1, Wire, 1);
  Serial.print("Start status: 0x");
  if(status < 0x10) Serial.print("0"); //Pretty output
  Serial.println(status, BIN);
  
  mlx_left.reset();
  mlx_left.setGainSel(gainSel);
  mlx_left.setResolution(resolution[0], resolution[1], resolution[2]); //x, y, z
  mlx_left.setOverSampling(overSampling);
  mlx_left.setDigitalFiltering(digitalFilt);
  mlx_left.setHallConf(0xC);

  mlx_right.reset();
  mlx_right.setGainSel(gainSel);
  mlx_right.setResolution(resolution[0], resolution[1], resolution[2]); //x, y, z
  mlx_right.setOverSampling(overSampling);
  mlx_right.setDigitalFiltering(digitalFilt);
  mlx_right.setHallConf(0xC);

  ref_mlx.reset();
  ref_mlx.setGainSel(gainSel);
  ref_mlx.setResolution(resolution[0], resolution[1], resolution[2]); //x, y, z
  ref_mlx.setOverSampling(overSampling);
  ref_mlx.setDigitalFiltering(digitalFilt);
  ref_mlx.setHallConf(0xC);
  Serial.println("Ready!");
}

void loop()
{   
    ref_mlx.readData(ref_data); 
    mlx_left.readData(data_left);
    mlx_right.readData(data_right); //Read the values from the sensor

    Serial.print(ref_data.x);
    Serial.print(" ");
    Serial.print(ref_data.y);
    Serial.print(" ");
    Serial.print(ref_data.z);
    Serial.print(" ");
    Serial.print(ref_data.t);
    Serial.print(" ");    
    Serial.print(data_left.x);
    Serial.print(" ");
    Serial.print(data_left.y);
    Serial.print(" ");
    Serial.print(data_left.z);
    Serial.print(" ");
    Serial.print(data_left.t);
    Serial.print(" ");    
    Serial.print(data_right.x);
    Serial.print(" ");
    Serial.print(data_right.y);
    Serial.print(" ");
    Serial.print(data_right.z);
    Serial.print(" ");
    Serial.print(data_right.t);
    Serial.println();
    delay(delayTime);

  
}
