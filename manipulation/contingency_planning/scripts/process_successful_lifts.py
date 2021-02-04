import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str)
    parser.add_argument('--save_inputs', '-si', action='store_true')
    args = parser.parse_args()

    block_data = np.load(args.file_path)

    if 'franka_fingers' in args.file_path or 'tongs_overhead' in args.file_path:
        inputs = block_data['x_y_thetas']
    elif 'tongs_side' in args.file_path or 'spatula_tilted' in args.file_path:
        inputs = block_data['x_y_theta_dist_tilts']
    elif 'spatula_flat' in args.file_path:
        inputs = block_data['x_y_theta_dists']

    initial_block_pose = block_data['initial_block_pose']
    if 'spatula' in args.file_path:
        post_grasp_block_pose = block_data['post_pick_up_block_pose']
    else:
        post_grasp_block_pose = block_data['post_grasp_block_pose']
    post_release_block_pose = block_data['post_release_block_pose']
    height_dif = post_grasp_block_pose[:,1] - initial_block_pose[:,1]
    incorrect_height_dif = post_release_block_pose[:,1] - initial_block_pose[:,1]

    height_thresh = 0.1

    successful_trials = height_dif > height_thresh
    incorrect_trials = incorrect_height_dif > height_thresh
    correct_trials = np.logical_and(successful_trials, np.logical_not(incorrect_trials))

    successful_inputs = inputs[correct_trials]
    unsuccessful_inputs = inputs[np.logical_not(correct_trials)]

    print(np.count_nonzero(correct_trials))
    print(np.count_nonzero(correct_trials) / initial_block_pose.shape[0])

    starting_idx = args.file_path.rfind('/')
    save_file_path = args.file_path[:starting_idx+1] + 'successful_lift_' + args.file_path[starting_idx+1:-4]
    np.save(save_file_path, correct_trials)

    if args.save_inputs:

        if 'spatula' in args.file_path:
            desired_positions = block_data['desired_push_robot_pose']
            actual_positions = block_data['push_robot_pose']
            diff = np.linalg.norm(desired_positions[correct_trials]-actual_positions[correct_trials], axis=1)
            robot_went_to_correct_pose = diff < 0.01
        else:
            desired_positions = block_data['desired_pre_grasp_robot_pose']
            actual_positions = block_data['pre_grasp_robot_pose']
            diff = np.linalg.norm(desired_positions[correct_trials]-actual_positions[correct_trials], axis=1)
            robot_went_to_correct_pose = diff < 0.01

        print(np.min(diff))
        print(np.median(diff))
        print(np.mean(diff))
        print(np.max(diff))

        

        print(np.count_nonzero(robot_went_to_correct_pose))

        if 'tongs_side' in args.file_path or 'spatula' in args.file_path:
            desired_positions2 = block_data['desired_pre_push_robot_pose']
            actual_positions2 = block_data['pre_push_robot_pose']
            diff2 = np.linalg.norm(desired_positions2[correct_trials]-actual_positions2[correct_trials], axis=1)

            print(np.min(diff2))
            print(np.median(diff2))
            print(np.mean(diff2))
            print(np.max(diff2))

            robot_went_to_correct_pose2 = diff2 < 0.09

            print(np.count_nonzero(robot_went_to_correct_pose2))

            robot_went_to_correct_pose = np.logical_and(robot_went_to_correct_pose,robot_went_to_correct_pose2)
            print(np.count_nonzero(robot_went_to_correct_pose))

        actually_correct_inputs = successful_inputs[robot_went_to_correct_pose]
        #unsuccessful_inputs = np.vstack((unsuccessful_inputs,successful_inputs[np.logical_not(robot_went_to_correct_pose)]))

        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter(actually_correct_inputs[:,0], actually_correct_inputs[:,1]);
        ax.set_title('Surface plot')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Num Samples')
        ax.view_init(azim=0, elev=90)
        plt.show()

        print(actually_correct_inputs.shape)

        starting_idx = args.file_path.rfind('/')
        save_file_path = args.file_path[:starting_idx+1] + 'successful_lift_inputs_' + args.file_path[starting_idx+1:-4] + '.npy'
        print(save_file_path)
        np.save(save_file_path, actually_correct_inputs)

        starting_idx = args.file_path.rfind('/')
        save_file_path = args.file_path[:starting_idx+1] + 'unsuccessful_lift_inputs_' + args.file_path[starting_idx+1:-4] + '.npy'
        np.save(save_file_path, unsuccessful_inputs)