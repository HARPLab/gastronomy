import glob
import numpy as np

num_successful_inputs = 2400
suffix = 'tongs_overhead'

file_paths = glob.glob('same_block_data/pick_up_only/*' + suffix + '.npy')

successful_lift_data = np.zeros((num_successful_inputs,0))

for file_path in file_paths:
    file_data = np.load(file_path)
    successful_lift_data = np.hstack((successful_lift_data, file_data.reshape(-1,1)))


success_mat = np.zeros((num_successful_inputs,num_successful_inputs))
failure_mat = np.zeros((num_successful_inputs,num_successful_inputs))

num_successes = np.sum(successful_lift_data, 1)
num_failures = successful_lift_data.shape[1] - num_successes

for i in range(successful_lift_data.shape[0]):
    for j in range(successful_lift_data.shape[1]):
        if(successful_lift_data[i,j]):
            success_mat[i,:] += successful_lift_data[:,j].reshape(-1) / num_successes
        else:
            failure_mat[i,:] += successful_lift_data[:,j].reshape(-1) / num_successes

success_mat = np.nan_to_num(success_mat)
failure_mat = np.nan_to_num(failure_mat)
#print(success_mat)
#print(failure_mat)

x_y_thetas = np.load('same_block_data/successful_lift_inputs_pick_up_only_with_' + suffix + '.npy')

x_y_thetas = x_y_thetas[:num_successful_inputs]
#x_y_thetas = np.hstack((x_y_z_thetas[:1000,:2],x_y_z_thetas[:1000,3].reshape(-1,1)))

Xs = np.zeros((0,3))
#Xs = np.zeros((0,6))
for i in range(num_successful_inputs):
    cur_x_y_thetas = np.repeat(x_y_thetas[i,:].reshape(1,-1), num_successful_inputs, axis=0)
    xs = x_y_thetas[:,:2] - x_y_thetas[i,:2]
    thetas = np.arctan2(np.sin(x_y_thetas[:,2] - x_y_thetas[i,2]), np.cos(x_y_thetas[:,2] - x_y_thetas[i,2]))
    #new_xs = np.hstack((cur_x_y_thetas, xs,thetas.reshape(-1,1)))
    new_xs = np.hstack((xs,thetas.reshape(-1,1)))
    Xs = np.vstack((Xs, new_xs))

success_Ys = success_mat.reshape(-1,1)
failure_Ys = failure_mat.reshape(-1,1)

np.savez('same_block_data/'+ suffix + '_contingency_data_pick_up_only.npz', X=Xs, success_Y=success_Ys, failure_Y=failure_Ys)

# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # Data for three-dimensional scattered points
# p = ax.scatter3D(Xs[:,0], Xs[:,1], Xs[:,2], c=success_Ys.flatten());
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('theta (rad)')
# fig.colorbar(p, ax=ax)
# plt.show()


#correlation = np.corrcoef(successful_lift_data)
#print(correlation)