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

num_blocks = 25
num_skills_to_use = 50

for i in range(1,num_blocks+1):
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
    #skill_inputs[suffix] = np.hstack((np.ones((successful_lift_data[suffix].shape[0],1)) * suffix_idx, skill_inputs[suffix][success_indices[0]]))
    skill_inputs[suffix] = skill_inputs[suffix][success_indices[0]]

# Reduce to only first num_skills_to_use skills
for suffix_idx in range(num_skills):
    suffix = suffices[suffix_idx]
    skill_inputs[suffix] = skill_inputs[suffix][:num_skills_to_use,:]
    successful_lift_data[suffix] = successful_lift_data[suffix][:num_skills_to_use,:]

for suffix_idx1 in range(num_skills):
    for suffix_idx2 in range(num_skills):
        if (suffix_idx1 == suffix_idx2):
            suffix1 = suffices[suffix_idx1]
            lift_data1 = successful_lift_data[suffix1]
            lift_inputs1 = skill_inputs[suffix1]
            lift_data2 = successful_lift_data[suffix1]
            lift_inputs2 = skill_inputs[suffix1]
            num_inputs = lift_inputs1.shape[1]

            print(suffix1)
        else:
            suffix1 = suffices[suffix_idx1]
            suffix2 = suffices[suffix_idx2]
            print(suffix1)
            print(suffix2)

            lift_data1 = successful_lift_data[suffix1]
            lift_data2 = successful_lift_data[suffix2]

            suffix1_lift_inputs = skill_inputs[suffix1]
            suffix2_lift_inputs = skill_inputs[suffix2]

            num_inputs = max(suffix1_lift_inputs.shape[1], suffix2_lift_inputs.shape[1])

            if(suffix1_lift_inputs.shape[1] == 5):
                lift_inputs1 = np.hstack((suffix1_lift_inputs[:,:3], suffix1_lift_inputs[:,4].reshape(-1,1), suffix1_lift_inputs[:,3].reshape(-1,1)))
            if(suffix2_lift_inputs.shape[1] == 5):
                lift_inputs2 = np.hstack((suffix2_lift_inputs[:,:3], suffix2_lift_inputs[:,4].reshape(-1,1), suffix2_lift_inputs[:,3].reshape(-1,1)))
            if(suffix1_lift_inputs.shape[1] < num_inputs):
                lift_inputs1 = np.hstack((suffix1_lift_inputs, np.zeros((suffix1_lift_inputs.shape[0], num_inputs - suffix1_lift_inputs.shape[1]))))
            if(suffix2_lift_inputs.shape[1] < num_inputs):
                lift_inputs2 = np.hstack((suffix2_lift_inputs, np.zeros((suffix2_lift_inputs.shape[0], num_inputs - suffix2_lift_inputs.shape[1]))))


        success_success_data = np.zeros((0,num_inputs))
        success_failure_data = np.zeros((0,num_inputs))
        failure_success_data = np.zeros((0,num_inputs))
        failure_failure_data = np.zeros((0,num_inputs))

        for i in range(lift_data1.shape[0]):
            for j in range(lift_data1.shape[1]):
                cur_x_y_theta = lift_inputs1[i,:]
                for k in range(lift_data2.shape[0]):

                    xs = lift_inputs2[k,:2] - cur_x_y_theta[:2]
                    theta = np.arctan2(np.sin(lift_inputs2[k,2] - cur_x_y_theta[2]), np.cos(lift_inputs2[k,2] - cur_x_y_theta[2]))
                    if num_inputs > 3:
                        dist = lift_inputs2[k,3] - cur_x_y_theta[3]
                    if num_inputs > 4:
                        tilt = np.arctan2(np.sin(lift_inputs2[k,4] - cur_x_y_theta[4]), np.cos(lift_inputs2[k,4] - cur_x_y_theta[4]))
                    
                    cur_xs = np.hstack((xs.reshape(1,-1),theta.reshape(1,-1)))
                    if num_inputs > 3:
                        cur_xs = np.hstack((cur_xs, dist.reshape(1,-1)))
                    if num_inputs > 4:
                        cur_xs = np.hstack((cur_xs, tilt.reshape(1,-1)))

                    if(lift_data1[i,j]):
                        if(lift_data2[k,j]):
                            success_success_data = np.vstack((success_success_data,cur_xs.reshape(1,-1)))
                        else:
                            success_failure_data = np.vstack((success_failure_data,cur_xs.reshape(1,-1)))
                    else:
                        if(lift_data2[k,j]):
                            failure_success_data = np.vstack((failure_success_data,cur_xs.reshape(1,-1)))
                        else:
                            failure_failure_data = np.vstack((failure_failure_data,cur_xs.reshape(1,-1)))


        if (suffix_idx1 == suffix_idx2):
            np.savez('same_blocks/contingency_data/' + suffix1 + '_contingency_data.npz', 
                    success_success_data=success_success_data,
                    success_failure_data=success_failure_data,
                    failure_success_data=failure_success_data,
                    failure_failure_data=failure_failure_data)
        else:
            np.savez('same_blocks/contingency_data/' + suffix1 + '_' + suffix2 + '_contingency_data.npz', 
                    success_success_data=success_success_data,
                    success_failure_data=success_failure_data,
                    failure_success_data=failure_success_data,
                    failure_failure_data=failure_failure_data)
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