# import the necessary packages
import numpy as np
import cv2
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='urdf_data/tongs_side/')
    args = parser.parse_args()

    urdf_data_files = glob.glob(args.data_dir + '*_tongs_side.npz')

    actions = np.zeros((0,5))
    Y = np.zeros((0,1))
    file_names = []

    num_successes = []

    for urdf_data_file in urdf_data_files:

        file_name = urdf_data_file[urdf_data_file.rfind('/')+1:-19]
        print(file_name)

        urdf_data = np.load(urdf_data_file)

        x_y_thetas = urdf_data['x_y_thetas']

        initial_urdf_pose = urdf_data['initial_urdf_pose']
        post_release_urdf_pose = urdf_data['post_release_urdf_pose']
        incorrect_height_dif = post_release_urdf_pose[:,1] - initial_urdf_pose[:,1]

        post_grasp_urdf_pose = urdf_data['post_grasp_urdf_pose']
        height_dif = post_grasp_urdf_pose[:,1] - initial_urdf_pose[:,1]

        height_thresh = 0.15

        successful_trials = height_dif > height_thresh
        incorrect_trials = incorrect_height_dif > height_thresh
        correct_trials = np.logical_and(successful_trials, np.logical_not(incorrect_trials))

        num_successes.append(correct_trials)
        print(np.count_nonzero(correct_trials))

        np.savez(args.data_dir + file_name + '_successful_lift_data.npz', X=x_y_thetas, Y=correct_trials)
    print(np.sum(num_successes))