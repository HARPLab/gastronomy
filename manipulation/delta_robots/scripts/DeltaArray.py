#!/usr/bin/env python

import math
import rospy
import numpy as np
from std_msgs.msg import Empty, Bool
from sensor_msgs.msg import JointState
from delta_msgs.msg import Float32Array
#from pympler.asizeof import asizeof

class DeltaArray:
    def __init__(self, init=True):
        if(init):
            rospy.init_node('DeltaArray')

        self.joint_names = ["delta1_motor1", "delta1_motor2", 
                            "delta1_motor3", "delta2_motor1", 
                            "delta2_motor2", "delta2_motor3"]
        self.current_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.current_joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.minimum_joint_position = 0.0
        self.maximum_joint_position = 0.0762

        self.reset_pub = rospy.Publisher('/delta_array/reset', Empty, queue_size=2)
        self.stop_pub = rospy.Publisher('/delta_array/stop', Bool, queue_size=2)
        self.delta_position_trajectory_pub = rospy.Publisher('/delta_array/joint_position_trajectory', Float32Array, queue_size=2);
        self.delta_velocity_trajectory_pub = rospy.Publisher('/delta_array/joint_velocity_trajectory', Float32Array, queue_size=2);
        
        rospy.sleep(0.3)

    def joint_state_callback(self, joint_state_msg):
        self.current_joint_positions = joint_state_msg.position
        self.current_joint_velocities = joint_state_msg.velocity

    def get_current_joint_states(self):
        return (self.current_joint_positions, self.current_joint_velocities)

    def reset(self):
        empty_msg = Empty()

        self.reset_pub.publish(empty_msg)

        rospy.sleep(0.3)
    
    def stop(self):
        bool_msg = Bool()
        bool_msg.data = True

        self.stop_pub.publish(bool_msg)

        rospy.sleep(0.3)

    def start(self):
        bool_msg = Bool()
        bool_msg.data = False

        self.stop_pub.publish(bool_msg)

        rospy.sleep(0.3)

    def move_delta_position(self, desired_delta_positions):

        np.clip(desired_delta_positions,self.minimum_joint_position,self.maximum_joint_position)

        compressed_array = np.float32(desired_delta_positions.flatten())
        joint_trajectory_msg = Float32Array()
        joint_trajectory_msg.data = list(compressed_array)
        #print('Size: ' + str(asizeof(joint_trajectory_msg)))

        self.delta_position_trajectory_pub.publish(joint_trajectory_msg)

        rospy.sleep(0.3)

        self.delta_position_trajectory_pub.publish(joint_trajectory_msg)

    def move_delta_velocity(self, desired_delta_velocities, durations):

        combined_array = np.hstack((np.array(desired_delta_velocities),np.array(durations).reshape(-1,1)))
        compressed_array = np.float32(combined_array.flatten())
        joint_trajectory_msg = Float32Array()
        joint_trajectory_msg.data = list(compressed_array)
        #print('Size: ' + str(asizeof(joint_trajectory_msg)))

        self.delta_velocity_trajectory_pub.publish(joint_trajectory_msg)

        # rospy.sleep(0.3)

        # self.delta_velocity_trajectory_pub.publish(joint_trajectory_msg)

    def wait_until_done_moving(self):
        done_moving = False

        while not done_moving:
            try:
                done_moving_deltas_msg = rospy.wait_for_message('/delta_array/done_moving_deltas', Bool, timeout=11)
                print(done_moving_deltas_msg.data)
                done_moving = done_moving_deltas_msg.data
            except:
                done_moving = False