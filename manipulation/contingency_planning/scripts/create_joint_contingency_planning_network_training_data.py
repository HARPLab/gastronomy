import glob
import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


suffices = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_flat', 'spatula_tilted']

num_skills = len(suffices)
num_successful_inputs = {}
for suffix in suffices:
    num_successful_inputs[suffix] = 0

file_paths = glob.glob('same_blocks/pick_up/*/*.npy')

successful_lift_data = {}

for i in range(1,41):
    for file_path in file_paths:
        if str(i) in file_path:
            file_data = np.load(file_path)
            for suffix in suffices:
                if suffix in file_path:
                    if num_successful_inputs[suffix] == 0:
                        successful_lift_data[suffix] = file_data.reshape(-1,1)
                        num_successful_inputs[suffix] = successful_lift_data[suffix].shape[0]
                    else:
                        successful_lift_data[suffix] = np.hstack((successful_lift_data[suffix], file_data.reshape(-1,1)))

skill_inputs = {}

for suffix_idx in range(num_skills):
    suffix = suffices[suffix_idx]
    skill_inputs[suffix] = np.load('same_blocks/pick_up/'+suffix+'/successful_lift_inputs_pick_up_block_with_' + suffix + '.npy')
    success_indices = np.nonzero(np.sum(successful_lift_data[suffix], 1).reshape(-1))
    successful_lift_data[suffix] = successful_lift_data[suffix][success_indices[0]]
    skill_inputs[suffix] = np.hstack((np.ones((successful_lift_data[suffix].shape[0],1)) * suffix_idx, skill_inputs[suffix][success_indices[0]]))

# Reduce to only first 500 skills
for suffix_idx in range(num_skills):
    suffix = suffices[suffix_idx]
    skill_inputs[suffix] = skill_inputs[suffix][:500,:]
    successful_lift_data[suffix] = successful_lift_data[suffix][:500,:]

for suffix_idx1 in range(num_skills):
    for suffix_idx2 in range(suffix_idx1, num_skills):
        if (suffix_idx1 == suffix_idx2):
            suffix1 = suffices[suffix_idx1]
            lift_data = successful_lift_data[suffix1]
            lift_inputs = skill_inputs[suffix1]
            num_inputs = lift_inputs.shape[1]

            print(suffix1)
        else:
            suffix1 = suffices[suffix_idx1]
            suffix2 = suffices[suffix_idx2]
            print(suffix1)
            print(suffix2)

            lift_data = np.vstack((successful_lift_data[suffix1], successful_lift_data[suffix2]))

            suffix1_lift_inputs = skill_inputs[suffix1]
            suffix2_lift_inputs = skill_inputs[suffix2]

            num_inputs = max(suffix1_lift_inputs.shape[1], suffix2_lift_inputs.shape[1])

            if(suffix1_lift_inputs.shape[1] == 6):
                suffix1_lift_inputs = np.hstack((suffix1_lift_inputs[:,:4], suffix1_lift_inputs[:,5].reshape(-1,1), suffix1_lift_inputs[:,4].reshape(-1,1)))
            if(suffix2_lift_inputs.shape[1] == 6):
                suffix2_lift_inputs = np.hstack((suffix2_lift_inputs[:,:4], suffix2_lift_inputs[:,5].reshape(-1,1), suffix2_lift_inputs[:,4].reshape(-1,1)))
            if(suffix1_lift_inputs.shape[1] < num_inputs):
                suffix1_lift_inputs = np.hstack((suffix1_lift_inputs, np.zeros((suffix1_lift_inputs.shape[0], num_inputs - suffix1_lift_inputs.shape[1]))))
            if(suffix2_lift_inputs.shape[1] < num_inputs):
                suffix2_lift_inputs = np.hstack((suffix2_lift_inputs, np.zeros((suffix2_lift_inputs.shape[0], num_inputs - suffix2_lift_inputs.shape[1]))))

            lift_inputs = np.vstack((suffix1_lift_inputs, suffix2_lift_inputs))



        num_lift_inputs = lift_data.shape[0]
        success_mat = np.zeros((num_lift_inputs,num_lift_inputs))
        failure_mat = np.zeros((num_lift_inputs,num_lift_inputs))

        print(num_lift_inputs)

        num_successes = np.sum(lift_data, 1)
        num_failures = lift_data.shape[1] - num_successes + 1

        for i in range(lift_data.shape[0]):
            for j in range(lift_data.shape[1]):
                if(lift_data[i,j]):
                    success_mat[i,:] += lift_data[:,j].reshape(-1) / num_successes[i]
                else:
                    failure_mat[i,:] += lift_data[:,j].reshape(-1) / num_failures[i]

        success_mat = np.nan_to_num(success_mat)
        failure_mat = np.nan_to_num(failure_mat)
        success_mat = np.clip(success_mat, 0, 1)
        failure_mat = np.clip(failure_mat, 0, 1)

        Xs = np.zeros((0,num_inputs))
        for i in range(num_lift_inputs):
            #cur_x_y_thetas = np.repeat(lift_inputs[i,:].reshape(1,-1), num_lift_inputs, axis=0)
            xs = lift_inputs[:,:3] - lift_inputs[i,:3]
            thetas = np.arctan2(np.sin(lift_inputs[:,3] - lift_inputs[i,3]), np.cos(lift_inputs[:,3] - lift_inputs[i,3]))
            if num_inputs > 4:
                dists = lift_inputs[:,4] - lift_inputs[i,4]
            if num_inputs > 5:
                tilts = np.arctan2(np.sin(lift_inputs[:,5] - lift_inputs[i,5]), np.cos(lift_inputs[:,5] - lift_inputs[i,5]))
            
            #new_xs = np.hstack((cur_x_y_thetas, xs,thetas.reshape(-1,1)))
            new_xs = np.hstack((xs,thetas.reshape(-1,1)))
            if num_inputs > 4:
                new_xs = np.hstack((new_xs, dists.reshape(-1,1)))
            if num_inputs > 5:
                new_xs = np.hstack((new_xs, tilts.reshape(-1,1)))
            Xs = np.vstack((Xs, new_xs))

        success_Ys = success_mat.reshape(-1,1)
        failure_Ys = failure_mat.reshape(-1,1)

        # Remove data where Xs[:,0] is 0 because that means it is same skill when there are 2 different skills
        if (suffix_idx1 != suffix_idx2):
            indices_where_Xs_is_not_0 = np.nonzero(np.logical_not(Xs[:,0] == 0))
            Xs = Xs[indices_where_Xs_is_not_0[0]]
            success_Ys = success_Ys[indices_where_Xs_is_not_0[0]]
            failure_Ys = failure_Ys[indices_where_Xs_is_not_0[0]]

        # failure_Ys /= np.max(failure_Ys)
        # failure_Ys *= 0.8
        # failure_Ys += 0.1
        # success_Ys *= 0.8
        # success_Ys += 0.1

        # print(np.max(success_Ys))
        # print(np.min(success_Ys))
        # print(np.max(failure_Ys))
        # print(np.min(failure_Ys))

        # if suffix_idx1 == 0 and suffix_idx2 == 0:
        fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(Xs[1:1000000:100,1], Xs[1:1000000:100,2], Xs[1:1000000:100,0], c=success_Ys[1:1000000:100].flatten());
        #o = ax.scatter3D(resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], resampled_points[simulation_num+1,0], c='red');
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Skill id')
        ax.set_title('Success Ys')
        fig.colorbar(p, ax=ax)
        fig = plt.figure(2)
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(Xs[1:1000000:100,1], Xs[1:1000000:100,2], Xs[1:1000000:100,0], c=failure_Ys[1:1000000:100].flatten());
        #o = ax.scatter3D(resampled_points[simulation_num+1,1], resampled_points[simulation_num+1,2], resampled_points[simulation_num+1,0], c='red');
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Skill id')
        ax.set_title('Failure Ys')
        fig.colorbar(p, ax=ax)
        plt.show()
        #print(suffices[int(resampled_points[simulation_num+1,0])])


        

        if (suffix_idx1 == suffix_idx2):
            np.savez('same_blocks/contingency_data/' + suffix1 + '_contingency_data_pick_up.npz', X=Xs, success_Y=success_Ys, failure_Y=failure_Ys)
        else:
            np.savez('same_blocks/contingency_data/' + suffix1 + '_' + suffix2 + '_contingency_data_pick_up.npz', X=Xs, success_Y=success_Ys, failure_Y=failure_Ys)

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