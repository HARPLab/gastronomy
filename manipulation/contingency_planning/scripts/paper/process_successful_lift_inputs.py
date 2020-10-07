import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str)
    args = parser.parse_args()

    block_data = np.load(args.file_path)

    #x_y_thetas = block_data['x_y_thetas'][:,:-1]
    #x_y_z_theta_tilt_dists = block_data['x_y_z_theta_tilt_dists']
    #x_y_z_theta_dists = block_data['x_y_z_theta_dists']
    #x_y_z_theta_tilt_dists = block_data['x_y_z_theta_tilt_dists']
    #x_y_theta_tilt_dists = block_data['x_y_theta_tilt_dists']
    #inputs = block_data['x_y_theta_dists']
    inputs = block_data['x_y_theta_tilt_dists']

    initial_block_pose = block_data['initial_block_pose']
    # post_grasp_block_pose = block_data['post_grasp_block_pose']
    post_release_block_pose = block_data['post_release_block_pose']
    # height_dif = post_grasp_block_pose[:,1] - initial_block_pose[:,1]
    incorrect_height_dif = post_release_block_pose[:,1] - initial_block_pose[:,1]

    post_pick_up_block_pose = block_data['post_pick_up_block_pose']
    height_dif = post_pick_up_block_pose[:,1] - initial_block_pose[:,1]

    height_thresh = 0.15
    #height_thresh = x_y_z_thetas[:,2] * 0.69
    #height_thresh = x_y_z_theta_tilt_dists[:,2] * 0.69
    #height_thresh = x_y_z_theta_dists[:,2] * 0.5
    #height_thresh = x_y_z_theta_tilt_dists[:,2] * 0.5

    successful_trials = height_dif > height_thresh
    incorrect_trials = incorrect_height_dif > height_thresh
    correct_trials = np.logical_and(successful_trials, np.logical_not(incorrect_trials))

    #successful_inputs = x_y_thetas[correct_trials]
    #successful_inputs = x_y_theta_tilt_dists[successful_trials]
    successful_inputs = inputs[correct_trials]
    #successful_inputs = x_y_z_theta_dists[successful_trials]
    #successful_inputs = x_y_z_theta_tilt_dists[correct_trials]
    #successful_inputs = np.hstack((successful_inputs[:,:2], successful_inputs[:,3:]))

    #unsuccessful_inputs = x_y_thetas[np.logical_not(correct_trials)]
    #unsuccessful_inputs = x_y_z_theta_tilt_dists[np.logical_not(correct_trials)]
    #unsuccessful_inputs = x_y_theta_tilt_dists[np.logical_not(successful_trials)]
    unsuccessful_inputs = inputs[np.logical_not(correct_trials)]
    #unsuccessful_inputs = x_y_z_theta_dists[np.logical_not(successful_trials)]
    #unsuccessful_inputs = np.hstack((unsuccessful_inputs[:,:2], unsuccessful_inputs[:,3:]))

    print(np.count_nonzero(correct_trials))
    print(np.count_nonzero(correct_trials) / initial_block_pose.shape[0])

    # print(np.count_nonzero(successful_trials))
    # print(np.count_nonzero(successful_trials) / initial_block_pose.shape[0])


    starting_idx = args.file_path.find('/')
    save_file_path = args.file_path[:starting_idx+1] + 'successful_lift_inputs_' + args.file_path[starting_idx+1:-4]
    np.save(save_file_path, successful_inputs)

    starting_idx = args.file_path.find('/')
    save_file_path = args.file_path[:starting_idx+1] + 'unsuccessful_lift_inputs_' + args.file_path[starting_idx+1:-4]
    np.save(save_file_path, unsuccessful_inputs)