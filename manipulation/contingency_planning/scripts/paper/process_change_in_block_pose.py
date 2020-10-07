import argparse
import numpy as np
from carbongym_utils.math_utils import np_to_quat, quat_to_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str)
    args = parser.parse_args()

    block_data = np.load(args.file_path)

    initial_block_pose = block_data['initial_block_pose']
    post_release_block_pose = block_data['post_release_block_pose']
    change_in_block_position = post_release_block_pose[:,:3] - initial_block_pose[:,:3]
    

    num_data = initial_block_pose.shape[0]
    change_in_block_orientation = np.zeros((num_data,4))

    for i in range(num_data):
        initial_quat = np_to_quat(initial_block_pose[i,3:])
        end_quat = np_to_quat(post_release_block_pose[i,3:])
        change_in_block_orientation[i] = quat_to_np(end_quat * initial_quat.inverse())
    
    change_in_block_pose = np.hstack((change_in_block_position,change_in_block_orientation))


    starting_idx = args.file_path.rfind('/')
    save_file_path = args.file_path[:starting_idx+1] + 'change_in_pose_' + args.file_path[starting_idx+1:-4]
    np.save(save_file_path, change_in_block_pose)