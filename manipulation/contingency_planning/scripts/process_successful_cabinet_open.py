import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='cabinets/cabinet_right_hinge.npz')
    args = parser.parse_args()

    data = np.load(args.file_path)

    inputs = data['x_y_zs']

    initial_handle_pose = data['initial_handle_pose']
    final_handle_pose = data['final_handle_pose']
    
    dif = np.linalg.norm(final_handle_pose - initial_handle_pose, axis=1)

    successful_trials = dif > 0.15

    print(np.min(dif))
    print(np.median(dif))
    print(np.mean(dif))
    print(np.max(dif))
    print(np.count_nonzero(dif > 0.15))

    successful_inputs = inputs[successful_trials]
    unsuccessful_inputs = inputs[np.logical_not(successful_trials)]

    starting_idx = args.file_path.rfind('/')
    save_file_path = args.file_path[:starting_idx+1] + 'successful_' + args.file_path[starting_idx+1:-4]
    np.save(save_file_path, successful_trials)

    save_file_path = args.file_path[:starting_idx+1] + 'successful_inputs_' + args.file_path[starting_idx+1:-4] + '.npy'
    print(save_file_path)
    np.save(save_file_path, successful_inputs)

    # starting_idx = args.file_path.rfind('/')
    # save_file_path = args.file_path[:starting_idx+1] + 'unsuccessful_inputs_' + args.file_path[starting_idx+1:-4] + '.npy'
    # np.save(save_file_path, unsuccessful_inputs)

    