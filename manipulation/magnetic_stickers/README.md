# magnetic_stickers

## Installation Instructions

1. Install Arduino 1.6.12 from the official arduino website.
2. Add Adafruit additional boards following this tutorial: https://learn.adafruit.com/adafruit-trinket-m0-circuitpython-arduino/arduino-ide-setup
3. Install Arduino SAMD Boards version 1.6.19 using the Boards Manager
4. Install Adafruit SAMD Boards version 1.5.4 using the Boards Manager
5. Install the MLX90393 library in Arduino
6. Plug in the Adafruit Trinket M0. 
7. Select "Adafruit Trinket M0" as the board
8. Depending on the number of magnetometers (N) you have connected to the Trinket M0, select the folder in the arduino folder that corresponds to the number (N) where it is listed as MagnetBoard_NX_Serial. Open the .ino file in the Arduino IDE and compile and upload to the Trinket M0.
9. The Trinket M0 will open a few windows and then restart. Afterwards, you can open the Serial Monitor in the Arduino IDE and the values should be publishing.
10. Go into the catkin_ws directory and run the command: catkin_make
11. Add the following line to the end of your .bashrc file: source /path/to/magnetic_stickers/catkin_ws/devel/setup.bash
12. Install pyserial using the following command: pip install pyserial

## Running Instructions
1. Start a roscore in a terminal
2. In another terminal, run the following command: rosrun magnetic_stickers magnetic_data_publisher.py
3. In another terminal, run the following command: rosrun magnetic_stickers magnetic_data_subscriber.py
4. The magnetic_data_subscriber.py terminal should start printing the current magnetic data similar to the Serial Monitor in the Arduino IDE.

## Running Scripts
The example scripts in the scripts folder require the use of the iamlab-cmu robot-interface repository. Some of the scripts require the azure_kinect_calibration repository and an azure kinect. 
1. Start a roscore in a terminal
2. In another terminal, run the following command: rosrun magnetic_stickers magnetic_data_publisher.py
3. In another terminal, start the control pc connection using the ./start_control_pc.sh script.
4. If the script requires the azure kinect, start the azure kinect.
4. Start the desired script.