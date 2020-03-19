import numpy as np
import argparse

from pyrep import PyRep

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

pr = PyRep()
pr.launch('./scene_0.ttt', headless=False) 
pr.start()  # Start the simulation

object = Shape.create(type=PrimitiveShape.CYLINDER, 
                      color=[255, 0, 0], size=[1, 1, 1],
                      position=[1, 1, 1])
object.set_color([255, 0, 0])
object.set_position([1, 1, 1])

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application