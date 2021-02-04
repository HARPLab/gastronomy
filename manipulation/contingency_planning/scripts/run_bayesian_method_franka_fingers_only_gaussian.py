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
    parser.add_argument('--num_trials', '-n', type=int, default=5)
    parser.add_argument('--contingency_gp_dir', type=str, default='same_blocks/pick_up/')
    parser.add_argument('--visualize_block_num', '-vbn', type=int, default=-1)
    parser.add_argument('--sampling_method', '-s', type=int, default=2)
    parser.add_argument('--num_skills', '-k', type=int, default=50)
    args = parser.parse_args()

    data = np.load('baseline/pick_up/good_franka_fingers_data.npy')

    success_success_contingency_gkde = joblib.load(args.contingency_gp_dir + 'franka_fingers_success_success_contingency_gkde_params.pkl')
    success_failure_contingency_gkde = joblib.load(args.contingency_gp_dir + 'franka_fingers_success_failure_contingency_gkde_params.pkl')
    failure_success_contingency_gkde = joblib.load(args.contingency_gp_dir + 'franka_fingers_failure_success_contingency_gkde_params.pkl')
    failure_failure_contingency_gkde = joblib.load(args.contingency_gp_dir + 'franka_fingers_failure_failure_contingency_gkde_params.pkl')

    gaussian_params = np.load('franka_fingers_gaussian.npz')

    success_success_rv = multivariate_normal(gaussian_params['success_success_mean'], gaussian_params['success_success_cov'])
    success_failure_rv = multivariate_normal(gaussian_params['success_failure_mean'], gaussian_params['success_failure_cov'])
    failure_success_rv = multivariate_normal(gaussian_params['failure_success_mean'], gaussian_params['failure_success_cov'])
    failure_failure_rv = multivariate_normal(gaussian_params['failure_failure_mean'], gaussian_params['failure_failure_cov'])

    x_y_thetas = np.load('data/franka_fingers_inputs.npy')

    num_blocks = data.shape[1]
    num_skills = data.shape[0]

    if args.num_skills == -1:
        num_skills = data.shape[0]
    else:
        num_skills = args.num_skills

    x_y_thetas = x_y_thetas[:num_skills,:]

    data = data[:num_skills,:num_blocks].reshape((num_skills,num_blocks))

    num_successes_each_block = np.zeros(num_blocks)
    first_success_each_block = np.ones(num_blocks) * (args.num_trials + 1)
    f1_scores = np.zeros(num_blocks)
    accuracies = np.zeros(num_blocks)

    for current_block_id in range(num_blocks):
        if args.visualize_block_num > 0:
            current_block_id = args.visualize_block_num
        print(current_block_id)

        skills_tested = []

        skill_success_probabilities = np.ones(num_skills) * 0.5
        skill_failure_probabilities = np.ones(num_skills) * 0.5

        block_data = data[:,current_block_id]

        #print(x_y_thetas[np.nonzero(block_data),:])

        if args.visualize_block_num >= 0:
            fig = plt.figure()
            ax = plt.axes()

            p = ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=block_data);
            ax.set_title('Ground Truth')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            #ax.set_zlabel('Num Samples')
            #ax.view_init(azim=0, elev=90)
            fig.colorbar(p)
            plt.show()

        for i in range(args.num_trials):

            if args.sampling_method == 0:
                skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_skills * 0.03))
            elif args.sampling_method == 1:
                skill_num = pick_top_skill(skill_success_probabilities)
            elif args.sampling_method == 2:
                if i < 3:
                    skill_num = pick_most_uncertain(skill_success_probabilities)
                else:
                    skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_skills * 0.03))

            skills_tested.append(skill_num)

            current_input = x_y_thetas[skill_num]

            cur_x_y_thetas = np.repeat(current_input.reshape(1,-1), 500, axis=0)
            xs = x_y_thetas[:,:2] - current_input[:2]
            thetas = np.arctan2(np.sin(x_y_thetas[:,2] - current_input[2]), np.cos(x_y_thetas[:,2] - current_input[2]))
            #relative_transforms = np.hstack((np.zeros(num_skills).reshape(-1,1),xs, thetas.reshape(-1,1)))
            relative_transforms = np.hstack((xs, thetas.reshape(-1,1)))

            if block_data[skill_num] == 1:
                new_skill_success_probabilities = success_success_contingency_gkde.pdf(np.transpose(relative_transforms))
                new_skill_failure_probabilities = success_failure_contingency_gkde.pdf(np.transpose(relative_transforms))
                

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
                new_skill_success_probabilities = failure_success_contingency_gkde.pdf(np.transpose(relative_transforms))
                new_skill_failure_probabilities = failure_failure_contingency_gkde.pdf(np.transpose(relative_transforms))

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

                p = ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=skill_success_probabilities);
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
        accuracies[current_block_id] = np.count_nonzero(predicted_successes == block_data) / num_skills
        f1_scores[current_block_id] = f1_score(block_data, predicted_successes)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_block), np.std(first_success_each_block)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_block),np.std(num_successes_each_block)))
    print(str(np.count_nonzero(first_success_each_block == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    print('Avg accuracy = {:.3f} +- {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))