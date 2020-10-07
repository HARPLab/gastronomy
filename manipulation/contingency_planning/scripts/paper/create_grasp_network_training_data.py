import argparse
import glob
import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', '-s', type=str)
    args = parser.parse_args()

    file_paths = glob.glob('same_block_data/pick_up_only/*' + args.suffix + '.npy')

    num_successful = 0

    successful_lift_data = np.zeros(0)

    for file_path in file_paths:
        print(file_path)
        file_data = np.load(file_path)
        if np.count_nonzero(file_data) > 0:
            if num_successful == 0:
                num_successful = file_data.shape[0]
                successful_lift_data = file_data.reshape(-1,1)
            else:
                successful_lift_data = np.hstack((successful_lift_data, file_data.reshape(-1,1)))

    successful_probabilities = np.mean(successful_lift_data, 1)
    print(successful_lift_data)

    successful_x_y_thetas = np.load('same_block_data/successful_lift_inputs_pick_up_only_with_' + args.suffix + '.npy')

    unsuccessful_x_y_thetas = np.load('same_block_data/unsuccessful_lift_inputs_pick_up_only_with_' + args.suffix + '.npy')
    unsuccessful_probabilities = np.zeros(unsuccessful_x_y_thetas.shape[0])

    x_y_thetas = np.vstack((successful_x_y_thetas[:num_successful],unsuccessful_x_y_thetas))
    probabilities = np.vstack((successful_probabilities.reshape(-1,1),unsuccessful_probabilities.reshape(-1,1)))

    print(x_y_thetas.shape)
    print(probabilities.shape)

    np.savez('same_block_data/' + args.suffix + '_pick_up_only_data.npz', X=x_y_thetas, y=probabilities)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # p = ax.scatter3D(successful_x_y_thetas[:2400,0], successful_x_y_thetas[:2400,1], 0, c=successful_probabilities[:2400].flatten());
    # #o = ax.scatter3D(resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], resampled_points[simulation_num+1,0], c='red');
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('Skill id')
    # fig.colorbar(p, ax=ax)
    # plt.show()
    # print(suffices[int(resampled_points[simulation_num+1,0])])