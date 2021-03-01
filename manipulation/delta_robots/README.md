# delta_robots

## Running Instructions
1. Start a roscore
`
roscore
`
2. Start the wireless joystick connection.
`
sudo ds4drv
`
3. Hold the PS and Share Buttons on the PS4 Dualshock controller
4. Start the ROS joystick node.
`
rosrun joy joy_node
`
5. Start the rosserial arduino node.
`
rosrun rosserial_arduino serial_node.py /dev/ttyACM1
`
6. Start the teleop ROS node.
`
python scripts/teleop.py
`