import glob
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    successful_data = np.load('cabinets/baseline_data.npy')

    x_y_zs = np.load('cabinets/successful_cabinet_inputs.npy')

    for cabinet_num in range(successful_data.shape[1]):

        if cabinet_num == 0:
            successful_cabinet_data = successful_data[:,1:]
        elif cabinet_num == (successful_data.shape[1]-1):
            successful_cabinet_data = successful_data[:,:(successful_data.shape[1]-1)]
        else:
            successful_cabinet_data = np.hstack((successful_data[:,:cabinet_num].reshape(successful_data.shape[0],-1), successful_data[:,(cabinet_num+1):].reshape(successful_data.shape[0],-1)))

        num_successful = successful_cabinet_data.shape[0]
        print(num_successful)
        num_successes = np.sum(successful_cabinet_data, 1)
        num_failures = successful_cabinet_data.shape[1] - num_successes

        success_mat = np.zeros((num_successful,num_successful))
        failure_mat = np.zeros((num_successful,num_successful))

        for i in range(successful_cabinet_data.shape[0]):
            for j in range(successful_cabinet_data.shape[1]):
                if(successful_cabinet_data[i,j]):
                    success_mat[i,:] += successful_cabinet_data[:,j].reshape(-1) / num_successes[i]
                else:
                    failure_mat[i,:] += successful_cabinet_data[:,j].reshape(-1) / num_failures[j]
        
        success_mat = np.nan_to_num(success_mat)
        failure_mat = np.nan_to_num(failure_mat)

        x_y_zs = x_y_zs[:num_successful]

        Xs = np.zeros((0,3))

        for i in range(num_successful):
            xs = x_y_zs - x_y_zs[i,:]
            Xs = np.vstack((Xs, xs))

        success_Ys = success_mat.reshape(-1,1)
        failure_Ys = failure_mat.reshape(-1,1)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # ax.scatter(Xs[::100,0], Xs[::100,1], Xs[::100,2], c=failure_Ys[::100].flatten());
        # ax.set_title('Surface plot')
        # ax.set_xlabel('x (m)')
        # ax.set_ylabel('y (m)')
        # ax.set_zlabel('Num Samples')
        # ax.view_init(azim=0, elev=90)
        # plt.show()

        np.savez('cabinets/'+str(cabinet_num)+'_contingency_data.npz', X=Xs, success_Y=success_Ys, failure_Y=failure_Ys)
