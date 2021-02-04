import argparse
import glob
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', '-s', type=str, default='franka_fingers')
    parser.add_argument('--num_skills', '-n', type=int, default=50)
    parser.add_argument('--num_blocks', '-b', type=int, default=-1)
    args = parser.parse_args()

    file_paths = glob.glob('same_blocks/pick_up/franka_fingers/successful_lift_block*' + args.suffix + '.npy')

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

    num_successful = args.num_skills
    num_blocks = args.num_blocks
    successful_lift_data = successful_lift_data[:num_successful,:num_blocks]

    x_y_thetas = np.load('same_blocks/pick_up/franka_fingers/successful_lift_inputs_pick_up_block_with_' + args.suffix + '.npy')

    x_y_thetas = x_y_thetas[:num_successful]

    # success_success_data = np.zeros((0,3))
    # success_failure_data = np.zeros((0,3))
    # failure_success_data = np.zeros((0,3))
    # failure_failure_data = np.zeros((0,3))

    # for i in range(successful_lift_data.shape[0]):
    #     for j in range(successful_lift_data.shape[1]):
    #         cur_x_y_theta = x_y_thetas[i,:]
    #         for k in range(successful_lift_data.shape[0]):
    #             xs = x_y_thetas[k,:2] - cur_x_y_theta[:2]
    #             theta = np.arctan2(np.sin(x_y_thetas[k,2] - cur_x_y_theta[2]), np.cos(x_y_thetas[k,2] - cur_x_y_theta[2]))
    #             cur_xs = np.array([xs[0],xs[1],theta])
    #             if(successful_lift_data[i,j]):
    #                 if(successful_lift_data[k,j]):
    #                     success_success_data = np.vstack((success_success_data,cur_xs.reshape(1,3)))
    #                 else:
    #                     success_failure_data = np.vstack((success_failure_data,cur_xs.reshape(1,3)))
    #             else:
    #                 if(successful_lift_data[k,j]):
    #                     failure_success_data = np.vstack((failure_success_data,cur_xs.reshape(1,3)))
    #                 else:
    #                     failure_failure_data = np.vstack((failure_failure_data,cur_xs.reshape(1,3)))

    # print(successful_lift_data.shape)

    num_successes = np.sum(successful_lift_data, 1)
    num_failures = successful_lift_data.shape[1] - num_successes

    success_mat = np.zeros((num_successful,num_successful))
    failure_mat = np.zeros((num_successful,num_successful))

    for i in range(successful_lift_data.shape[0]):
        for j in range(successful_lift_data.shape[1]):
            if(successful_lift_data[i,j]):
                success_mat[i,:] += successful_lift_data[:,j].reshape(-1) / num_successes[i]
            else:
                failure_mat[i,:] += successful_lift_data[:,j].reshape(-1) / num_failures[i]

    # print(successful_lift_data)

    # for i in range(successful_lift_data.shape[0]):
    #     for j in range(successful_lift_data.shape[1]):
    #         if (success_mat[i,j] == 0):
    #             print("%d, %d" %(i,j))
    #             print(successful_lift_data[9])


    for i in range(successful_lift_data.shape[0]):
        success_mat[i,i] = 1

    success_mat = np.nan_to_num(success_mat)
    failure_mat = np.nan_to_num(failure_mat)

    Xs = np.zeros((0,3))
    #Xs = np.zeros((0,6))

    for i in range(num_successful):
        cur_x_y_thetas = np.repeat(x_y_thetas[i,:].reshape(1,-1), num_successful, axis=0)
        xs = x_y_thetas[:,:2] - x_y_thetas[i,:2]
        thetas = np.arctan2(np.sin(x_y_thetas[:,2] - x_y_thetas[i,2]), np.cos(x_y_thetas[:,2] - x_y_thetas[i,2]))
        new_xs = np.hstack((xs,thetas.reshape(-1,1)))
        #new_xs = np.hstack((cur_x_y_thetas, xs,thetas.reshape(-1,1)))
        Xs = np.vstack((Xs, new_xs))

    success_Ys = success_mat.reshape(-1,1)
    failure_Ys = failure_mat.reshape(-1,1)

    

    # print(np.max(Xs[:,0]))
    # for i in range(len(Xs[:,0])):
    #     print("i : % 2.2f, j: %2.2f, k: %2.2f" %(Xs[i,0], Xs[i,1], success_Ys[i]))
    # print(x_y_thetas)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # #p = ax.scatter(Xs[::100,0], Xs[::100,1], c=failure_Ys[::100].flatten());
    # p = ax.scatter(success_success_data[:,0], success_success_data[:,1], c=np.ones(success_success_data.shape[0]));
    # ax.set_title('Surface plot')
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('Num Samples')
    # ax.view_init(azim=0, elev=90)
    # fig.colorbar(p)
    # plt.show()

    np.savez('same_blocks/pick_up/'+ args.suffix + '_contingency_data.npz', X=Xs, success_Y=success_Ys, failure_Y=failure_Ys)
    # np.savez('same_blocks/pick_up/'+ args.suffix + '_contingency_data.npz', 
    #          success_success_data=success_success_data,
    #          success_failure_data=success_failure_data,
    #          failure_success_data=failure_success_data,
    #          failure_failure_data=failure_failure_data)
