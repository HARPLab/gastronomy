import glob
import argparse
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from carbongym_utils.math_utils import np_to_quat, quat_to_rpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', '-f', type=str, default='same_block_data/pick_up_only/')
    args = parser.parse_args()

    file_paths = glob.glob(args.file_dir + 'change_in_pose_*.npy')

    changes_in_block_pose = np.zeros((0,7))

    for file_path in file_paths:

        change_in_block_pose = np.load(file_path)
        changes_in_block_pose = np.concatenate((changes_in_block_pose,change_in_block_pose))


    num_data_points = changes_in_block_pose.shape[0]
    rpy = np.zeros((num_data_points,3))
    for i in range(num_data_points):
        rpy[i,:] = quat_to_rpy(np_to_quat(changes_in_block_pose[i,3:]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(changes_in_block_pose[0::10,0], -changes_in_block_pose[0::10,2], rpy[0::10,2], cmap='Greens')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('pho')
    # ax = plt.axes()
    # ax.scatter(changes_in_block_pose[0::10,2], rpy[0::10,1])
    # ax.set_xlabel('y')
    # ax.set_ylabel('theta')
    plt.show()