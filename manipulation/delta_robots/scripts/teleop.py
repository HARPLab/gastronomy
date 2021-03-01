#! /usr/bin/env python
import numpy as np
import rospy
from DeltaArray import DeltaArray
from sensor_msgs.msg import Joy

# This ROS Node converts Joystick inputs from the joy node
# into commands for turtlesim or any other robot

# Receives joystick messages (subscribed to Joy topic)
# then converts the joysick inputs into Twist commands
# axis 1 aka left stick vertical controls linear speed
# axis 0 aka left stick horizonal controls angular speed
def callback(data):
    left_joy_stick_x = -data.axes[0]
    left_joy_stick_y = data.axes[1]
    left_up_trigger_z = data.buttons[4]
    left_down_trigger_z = -data.axes[3]
    right_joy_stick_x = data.axes[2]
    right_joy_stick_y = data.axes[5]
    right_up_trigger_z = data.buttons[5]
    right_down_trigger_z = -data.axes[4]

    bad_motor_up_button = data.buttons[3]
    bad_motor_down_button = data.buttons[1]

    delta_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    if(left_down_trigger_z > 0.8):
        delta_velocity[0][3] = 0.005
        delta_velocity[0][4] = 0.005
        delta_velocity[0][5] = 0.005
    elif(left_up_trigger_z > 0.8):
        delta_velocity[0][3] = -0.005
        delta_velocity[0][4] = -0.005
        delta_velocity[0][5] = -0.005
    if(right_down_trigger_z > 0.8):
        delta_velocity[0][0] = 0.005
        delta_velocity[0][1] = 0.005
        delta_velocity[0][2] = 0.005
    elif(right_up_trigger_z > 0.8):
        delta_velocity[0][0] = -0.005
        delta_velocity[0][1] = -0.005
        delta_velocity[0][2] = -0.005

    if(left_joy_stick_x < -0.8):
        delta_velocity[0][3] = 0.005
        delta_velocity[0][4] = 0.005
        delta_velocity[0][5] = -0.005
    elif(left_joy_stick_x > 0.8):
        delta_velocity[0][3] = -0.005
        delta_velocity[0][4] = -0.005
        delta_velocity[0][5] = 0.005
    if(right_joy_stick_x < -0.8):
        delta_velocity[0][0] = -0.005
        delta_velocity[0][1] = 0.005
        delta_velocity[0][2] = 0.005
    elif(right_joy_stick_x > 0.8):
        delta_velocity[0][0] = 0.005
        delta_velocity[0][1] = -0.005
        delta_velocity[0][2] = -0.005

    if(left_joy_stick_y < -0.8):
        delta_velocity[0][3] = -0.005
        delta_velocity[0][4] = 0.005
    elif(left_joy_stick_y > 0.8):
        delta_velocity[0][3] = 0.005
        delta_velocity[0][4] = -0.005
    if(right_joy_stick_y < -0.8):
        delta_velocity[0][1] = 0.005
        delta_velocity[0][2] = -0.005
    elif(right_joy_stick_y > 0.8):
        delta_velocity[0][1] = -0.005
        delta_velocity[0][2] = 0.005

    if(bad_motor_up_button > 0.8):
        delta_velocity[0][3] = -0.005
    elif(bad_motor_down_button > 0.8):
        delta_velocity[0][3] = 0.005

    for i in range(6):
        if delta_velocity[0][i] != 0:
            delta_array.move_delta_velocity(delta_velocity, [0.3])
            break
    

# Intializes everything
def start():
    # publishing to "turtle1/cmd_vel" to control turtle1
    
    #rospy.init_node('DeltaTeleop')
    global delta_array
    delta_array = DeltaArray()
    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("/joy", Joy, callback, queue_size=1)
    # starts the node
    rospy.spin()

if __name__ == '__main__':
    start()
