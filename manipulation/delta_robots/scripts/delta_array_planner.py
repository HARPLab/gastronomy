#!/usr/bin/env python

import time
import rospy
import numpy as np
import math
from DeltaArray import DeltaArray

#All measurements in meters and radians

class DeltaArrayPlanner:
    def __init__(self):
        ## Tuning Parameters

        ## Vertical Station Offsets
        ## V1 Offsets
        self.raspberry_pi_camera_vertical_station_v1_x_offset = -0.2
        self.raspberry_pi_camera_vertical_station_v1_z_offset = -0.06
        self.vertical_station_v1_goal_x_offset = -0.08
        self.vertical_station_v1_goal_z_offset = 0.01

        ## V2 Offsets
        self.raspberry_pi_camera_vertical_station_v2_x_offset = -0.22
        self.raspberry_pi_camera_vertical_station_v2_z_offset = -0.06
        self.vertical_station_v2_goal_x_offset = -0.1
        self.vertical_station_v2_goal_z_offset = 0.02

        ## V3 Offsets
        self.raspberry_pi_camera_vertical_station_v3_x_offset = -0.2
        self.raspberry_pi_camera_vertical_station_v3_z_offset = -0.07
        self.vertical_station_v3_goal_x_offset = -0.07
        self.vertical_station_v3_goal_z_offset = 0.0

        ## Breaker Offsets
        self.raspberry_pi_camera_vertical_station_breaker_x_off