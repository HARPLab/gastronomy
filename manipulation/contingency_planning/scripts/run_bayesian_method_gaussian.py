import argparse
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import multivariate_normal, gaussian_kde

import joblib

from sampling_methods import pick_random_skill_from_top_n, pick_most_uncertain, pick_top_skill

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', '-n', type=int, default=10)
    parser.add_argument('--contingency_gp_dir', type=str, default='same_blocks/contingency_data/')
    parser.add_argument('--visualize_block_num', '-vbn', type=int, default=-1)
    parser.add_argument('--sampling_method', '-s', type=int, default=2)
    parser.add_argument('--joint_cont_off', '-j', action='store_true')
    parser.add_argument('--num_skills', '-k', type=int, default=-1)
    args = parser.parse_args()

    data = np.load('baseline/pick_up/complete_data.npy')

    num_blocks = data.shape[1]
    num_total_skills = data.shape[0]

    if args.num_skills == -1:
        num_total_skills = 500
    else:
        num_total_skills = args.num_skills
    
    suffices = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_flat', 'spatula_tilted']
    skill_input_nums = [3, 3, 5, 4, 5]
    num_skills = len(suffices)

    contingency_nns = {}

    for suffix_idx1 in range(num_skills):
        for suffix_idx2 in range(num_skills):
            if (suffix_idx1 == suffix_idx2):
                key = suffices[suffix_idx1]
            else:
                suffix1 = suffices[suffix_idx1]
                suffix2 = suffices[suffix_idx2]
                key = suffix1 + '_' + suffix2
            contingency_nns[key+'_success_success'] = joblib.load(args.contingency_gp_dir + key + '_success_success_contingency_gkde_params.pkl')
            contingency_nns[key+'_success_failure'] = joblib.load(args.contingency_gp_dir + key + '_success_failure_contingency_gkde_params.pkl')
            contingency_nns[key+'_failure_success'] = joblib.load(args.contingency_gp_dir + key + '_failure_success_contingency_gkde_params.pkl')
            contingency_nns[key+'_failure_failure'] = joblib.load(args.contingency_gp_dir + key + '_failure_failure_contingency_gkde_params.pkl')
            
    sorted_x_y_thetas = np.load('data/franka_fingers_inputs.npy')
    franka_x_y_thetas = np.hstack((sorted_x_y_thetas, np.zeros((500,2))))
    #franka_fingers_probs = franka_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_thetas = np.load('data/tongs_overhead_inputs.npy')
    tong_overhead_x_y_thetas = np.hstack((sorted_x_y_thetas, np.zeros((500,2))))
    #tong_overhead_probs = tong_overhead_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_theta_dist_tilts = np.load('data/tongs_side_inputs.npy')
    tong_side_x_y_theta_dist_tilts = sorted_x_y_theta_dist_tilts
    #tong_side_probs = tong_side_nn_model.predict(sorted_x_y_theta_dist_tilts)

    sorted_x_y_theta_dists = np.load('data/spatula_flat_inputs.npy')
    spatula_flat_x_y_theta_dists = np.hstack((sorted_x_y_theta_dists, np.zeros((500,1))))
    #spatula_flat_probs = spatula_flat_nn_model.predict(sorted_x_y_theta_dists)

    sorted_x_y_theta_dist_tilts = np.load('data/spatula_tilted_inputs.npy')
    spatula_tilted_x_y_theta_dist_tilts = sorted_x_y_theta_dist_tilts
    #spatula_tilted_probs = spatula_tilted_nn_model.predict(sorted_x_y_theta_dist_tilts)

    inputs = np.vstack((franka_x_y_thetas[:num_total_skills,:], tong_overhead_x_y_thetas[:num_total_skills,:], tong_side_x_y_theta_dist_tilts[:num_total_skills,:], spatula_flat_x_y_theta_dists[:num_total_skills,:], spatula_tilted_x_y_theta_dist_tilts[:num_total_skills,:]))

    new_data = np.zeros((0,num_blocks))
    for skill_id in range(num_skills):
        new_data = np.vstack((new_data,data[skill_id*500:skill_id*500+num_total_skills,:num_blocks]))

    num_successes_each_block = np.zeros(num_blocks)
    first_success_each_block = np.ones(num_blocks) * (args.num_trials + 1)
    f1_scores = np.zeros(num_blocks)
    accuracies = np.zeros(num_blocks)

    for current_block_id in range(num_blocks):
        if args.visualize_block_num > 0:
            current_block_id = args.visualize_block_num
        print(current_block_id)

        skills_tested = []

        skill_success_probabilities = np.ones(num_total_skills*num_skills) * 0.5
        skill_failure_probabilities = np.ones(num_total_skills*num_skills) * 0.5

        block_data = new_data[:,current_block_id]

        #print(x_y_thetas[np.nonzero(block_data),:])

        if args.visualize_block_num >= 0:
            fig = plt.figure()
            ax = plt.axes()

            p = ax.scatter(inputs[:,0], inputs[:,1], c=block_data);
            ax.set_title('Ground Truth')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            #ax.set_zlabel('Num Samples')
            #ax.view_init(azim=0, elev=90)
            fig.colorbar(p)
            plt.show()

        for i in range(args.num_trials):

            if args.sampling_method == 0:
                skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_total_skills * num_skills * 0.03))
            elif args.sampling_method == 1:
                skill_num = pick_top_skill(skill_success_probabilities)
            elif args.sampling_method == 2:
                if i < 3:
                    skill_num = pick_most_uncertain(skill_success_probabilities)
                else:
                    skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_total_skills * num_skills * 0.03))

            skills_tested.append(skill_num)

            current_input = inputs[skill_num]
            skill_id = int(current_input[0])

            xs = inputs[:,:2] - current_input[:2]
            thetas = np.arctan2(np.sin(inputs[:,2] - current_input[2]), np.cos(inputs[:,2] - current_input[2]))
            dists = inputs[:,3] - current_input[3]
            tilts = np.arctan2(np.sin(inputs[:,4] - current_input[4]), np.cos(inputs[:,4] - current_input[4]))
            relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

            new_skill_success_probabilities = np.ones(skill_success_probabilities.shape) * 0.5
            new_skill_failure_probabilities = np.ones(skill_success_probabilities.shape) * 0.5

            if block_data[skill_num] == 1:
                if args.joint_cont_off:
                    key = suffices[skill_id]
                    num_inputs = skill_input_nums[skill_id]

                    new_skill_success_probabilities[num_total_skills*skill_id:num_total_skills*(skill_id+1)] = contingency_nns[key+'_success_success'].pdf(np.transpose(relative_transforms[num_total_skills*skill_id:num_total_skills*(skill_id+1),:num_inputs])).reshape(-1)
                    new_skill_failure_probabilities[num_total_skills*skill_id:num_total_skills*(skill_id+1)] = contingency_nns[key+'_success_failure'].pdf(np.transpose(relative_transforms[num_total_skills*skill_id:num_total_skills*(skill_id+1),:num_inputs])).reshape(-1)
                
                else:
                    for suffix_idx in range(num_skills):
                        if suffix_idx == skill_id:
                            key = suffices[suffix_idx]
                            num_inputs = skill_input_nums[skill_id]
                        else:
                            suffix1 = suffices[skill_id]
                            suffix2 = suffices[suffix_idx]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])

                        new_skill_success_probabilities[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1)] = contingency_nns[key+'_success_success'].pdf(np.transpose(relative_transforms[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1),:num_inputs])).reshape(-1)
                        new_skill_failure_probabilities[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1)] = contingency_nns[key+'_success_failure'].pdf(np.transpose(relative_transforms[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1),:num_inputs])).reshape(-1)
                

                if args.visualize_block_num >= 0: 
                    fig = plt.figure()
                    ax = plt.axes()

                    ax.scatter(relative_transforms[:,0], relative_transforms[:,1], c=new_skill_success_probabilities);
                    ax.scatter(0, 0, c='red');
                    ax.set_title('Success plot')
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    #ax.set_zlabel('Num Samples')
                    #ax.view_init(azim=0, elev=90)
                    plt.show()

                num_successes_each_block[current_block_id] += 1
                if first_success_each_block[current_block_id] == (args.num_trials + 1):
                    first_success_each_block[current_block_id] = i + 1
            else:
                if args.joint_cont_off:
                    key = suffices[skill_id]
                    num_inputs = skill_input_nums[skill_id]

                    new_skill_success_probabilities[num_total_skills*skill_id:num_total_skills*(skill_id+1)] = contingency_nns[key+'_failure_success'].pdf(np.transpose(relative_transforms[num_total_skills*skill_id:num_total_skills*(skill_id+1),:num_inputs])).reshape(-1)
                    new_skill_failure_probabilities[num_total_skills*skill_id:num_total_skills*(skill_id+1)] = contingency_nns[key+'_failure_failure'].pdf(np.transpose(relative_transforms[num_total_skills*skill_id:num_total_skills*(skill_id+1),:num_inputs])).reshape(-1)
                        
                else:
                    for suffix_idx in range(num_skills):
                        if suffix_idx == skill_id:
                            key = suffices[suffix_idx]
                            num_inputs = skill_input_nums[skill_id]
                        else:
                            suffix1 = suffices[skill_id]
                            suffix2 = suffices[suffix_idx]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])

                        new_skill_success_probabilities[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1)] = contingency_nns[key+'_failure_success'].pdf(np.transpose(relative_transforms[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1),:num_inputs])).reshape(-1)
                        new_skill_failure_probabilities[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1)] = contingency_nns[key+'_failure_failure'].pdf(np.transpose(relative_transforms[num_total_skills*suffix_idx:num_total_skills*(suffix_idx+1),:num_inputs])).reshape(-1)

                if args.visualize_block_num >= 0:
                    fig = plt.figure()
                    ax = plt.axes()

                    ax.scatter(relative_transforms[:,0], relative_transforms[:,1], c=new_skill_success_probabilities);
                    ax.scatter(0, 0, c='red');
                    ax.set_title('Failure plot')
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    #ax.set_zlabel('Num Samples')
                    #ax.view_init(azim=0, elev=90)
                    plt.show()

            skill_success_probabilities = skill_success_probabilities * new_skill_success_probabilities
            skill_failure_probabilities = skill_failure_probabilities * new_skill_failure_probabilities

            skill_probabilities_sum = skill_success_probabilities + skill_failure_probabilities
            skill_success_probabilities = skill_success_probabilities / skill_probabilities_sum
            skill_failure_probabilities = skill_failure_probabilities / skill_probabilities_sum

            if args.visualize_block_num >= 0:
                fig = plt.figure()
                ax = plt.axes()

                p = ax.scatter(inputs[:,0], inputs[:,1], c=skill_success_probabilities);
                ax.scatter(current_input[0], current_input[1], c='red');
                if i == 0:
                    ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skill')
                else:
                    ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skills')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                fig.colorbar(p)
                #ax.set_zlabel('Num Samples')
                #ax.view_init(azim=0, elev=90)
                plt.show()

        predicted_successes = (skill_success_probabilities) > 0.5
        accuracies[current_block_id] = np.count_nonzero(predicted_successes == block_data) / (num_total_skills * num_skills)
        f1_scores[current_block_id] = f1_score(block_data, predicted_successes)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_block), np.std(first_success_each_block)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_block),np.std(num_successes_each_block)))
    print(str(np.count_nonzero(first_success_each_block == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    print('Avg accuracy = {:.3f} +- {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))