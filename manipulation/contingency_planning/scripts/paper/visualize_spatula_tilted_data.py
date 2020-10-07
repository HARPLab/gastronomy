import glob
import numpy as np
import matplotlib.pyplot as plt

file_paths = glob.glob('same_block_data/pick_up_only/successful_lift_*pick_up_block_with_spatula_tilted.npy')
#file_paths = glob.glob('same_block_data/contingency_data/*.npz')

for file_path in file_paths:
	print(file_path)
	data = np.load(file_path)

	fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], sorted_x_y_thetas[:,0], c=data[:].flatten());
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Skill id')
    fig.colorbar(p, ax=ax)
    plt.show()
	