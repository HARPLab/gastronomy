#!/usr/bin/env python

import time
import numpy as np
from DeltaArray import DeltaArray

if __name__ == '__main__':

    delta_array = DeltaArray()

    delta_velocity = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005]).reshape((1,6))
    #delta_velocity = np.array([0,0,0, 0, 0.005,0]).reshape((1,6))

    delta_array.move_delta_velocity(delta_velocity,[1.0])
