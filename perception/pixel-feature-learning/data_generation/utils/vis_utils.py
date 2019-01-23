from visualization import Visualizer3D as vis
import numpy as np

def show_points(points, color=(0,1,0), scale=0.005, frame_size=0.2, frame_radius=0.02):
    vis.figure(bgcolor=(1,1,1), size=(500,500))
    vis.points(np.array(points), color=color, scale=scale)

    vis.plot3d(np.array(([0, 0, 0], [frame_size, 0, 0])).astype(np.float32), color=(1,0,0), tube_radius=frame_radius)
    vis.plot3d(np.array(([0, 0, 0], [0, frame_size, 0])).astype(np.float32), color=(0,1,0), tube_radius=frame_radius)
    vis.plot3d(np.array(([0, 0, 0], [0, 0, frame_size])).astype(np.float32), color=(0,0,1), tube_radius=frame_radius)

    vis.show()

