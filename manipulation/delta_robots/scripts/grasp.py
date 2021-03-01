#!/usr/bin/env python

import time
import numpy as np
from DeltaArray import DeltaArray

if __name__ == '__main__':

    delta_array = DeltaArray()

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    time.sleep(3)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    time.sleep(8)

    delta_position = np.array([0.015, 0.01, 0.01, 0.01, 0.01, 0.015]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()

    time.sleep(20)

    delta_position = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    
    time.sleep(5)

    delta_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,6))

    delta_array.move_delta_position(delta_position)

    #delta_array.wait_until_done_moving()
    