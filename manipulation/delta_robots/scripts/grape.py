#!/usr/bin/env python

import time
import numpy as np
from DeltaArray import DeltaArray

if __name__ == '__main__':

    delta_array = DeltaArray()

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_position = np.array([0.03, 0.035, 0.035, 0.035, 0.035, 0.03]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(8)

    delta_position = np.array([0.04, 0.035, 0.035, 0.035, 0.035, 0.04]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_position = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.005]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(10)

    # delta_velocity = np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005]).reshape((1,6))

    # delta_array.move_delta_velocity(delta_velocity,[1.0])

    
    