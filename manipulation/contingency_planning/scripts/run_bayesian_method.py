import argparse
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import load_model

from sampling_methods import pick_random_skill_from_top_n, pick_most_uncertain, pick_top_skill

def resample_points(num_samples, probabilities, x_y_thetas):
    probabilities_sum = np.sum(probabilities, 0)
    normalized_probabilities = probabilities / probabilities_sum

    random_samples = np.random.random(num_samples)

    resampled_points = np.zeros((0,x_y_thetas.shape[1]))

    for i in range(normalized_probabilities.shape[0]):
        random_samples -= normalized_probabilities[i]
        nonzero_samples = np.nonzero(random_samples < 0)
        resampled_points = np.vstack((resampled_points,np.repeat(x_y_thetas[i].reshape(1,-1),nonzero_samples[0].shape[0], axis=0)))
        random_samples[nonzero_samples[0]] += 1

    resampled_points[:,0] += np.random.normal(size=num_samples) * 0.001
    resampled_points[:,1] += np.random.normal(size=num_samples) * 0.001
    resampled_points[:,2] += np.random.normal(size=num_samples) * 0.001

    np.random.shuffle(resampled_points)
    return resampled_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', '-n', type=int, default=10)
    parser.add_argument('--contingency_nn_dir', type=str, default='same_blocks/contingency_data/')
    parser.add_argument('--joint_cont_off', '-j', action='store_true')
    parser.add_argument('--visualize_block_num', '-vbn', type=int, default=-1)
    parser.add_argument('--sampling_method', '-s', type=int, default=0)
    parser.add_argument('--use_full', '-uf', action='store_true')
    args = parser.parse_args()

    # franka_nn_model = load_model(args.franka_nn)
    # tong_overhead_nn_model = load_model(args.tong_overhead_nn)
    # tong_side_nn_model = load_model(args.tong_side_nn)
    # spatula_tilted_nn_model = load_model(args.spatula_tilted_nn)

    data = np.load('baseline/pick_up/complete_data.npy')

    suffices = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_flat', 'spatula_tilted']
    skill_input_nums = [4, 4, 6, 5, 6]
    num_skills = len(suffices)

    contingency_nns = {}

    for suffix_idx1 in range(num_skills):
        for suffix_idx2 in range(suffix_idx1, num_skills):
            if (suffix_idx1 == suffix_idx2):
                key = suffices[suffix_idx1]
            else:
                suffix1 = suffices[suffix_idx1]
                suffix2 = suffices[suffix_idx2]
                key = suffix1 + '_' + suffix2
            contingency_nns[key+'_success'] = load_model(args.contingency_nn_dir + key + '_success_contingency_model.h5')
            contingency_nns[key+'_failure'] = load_model(args.contingency_nn_dir + key + '_failure_contingency_model.h5')
            
    sorted_x_y_thetas = np.load('data/franka_fingers_inputs.npy')
    franka_x_y_thetas = np.hstack((np.zeros((500,1)), sorted_x_y_thetas, np.zeros((500,2))))
    #franka_fingers_probs = franka_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_thetas = np.load('data/tongs_overhead_inputs.npy')
    tong_overhead_x_y_thetas = np.hstack((np.ones((500,1)), sorted_x_y_thetas, np.zeros((500,2))))
    #tong_overhead_probs = tong_overhead_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_theta_dist_tilts = np.load('data/tongs_side_inputs.npy')
    tong_side_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 2, sorted_x_y_theta_dist_tilts))
    #tong_side_probs = tong_side_nn_model.predict(sorted_x_y_theta_dist_tilts)

    sorted_x_y_theta_dists = np.load('data/spatula_flat_inputs.npy')
    spatula_flat_x_y_theta_dists = np.hstack((np.ones((500,1)) * 3, sorted_x_y_theta_dists, np.zeros((500,1))))
    #spatula_flat_probs = spatula_flat_nn_model.predict(sorted_x_y_theta_dists)

    sorted_x_y_theta_dist_tilts = np.load('data/spatula_tilted_inputs.npy')
    spatula_tilted_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 4, sorted_x_y_theta_dist_tilts))
    #spatula_tilted_probs = spatula_tilted_nn_model.predict(sorted_x_y_theta_dist_tilts)

    inputs = np.vstack((franka_x_y_thetas, tong_overhead_x_y_thetas, tong_side_x_y_theta_dist_tilts, spatula_flat_x_y_theta_dists, spatula_tilted_x_y_theta_dist_tilts))

    num_blocks = data.shape[1]
    num_total_skills = data.shape[0]

    num_successes_each_block = np.zeros(num_blocks)
    first_success_each_block = np.ones(num_blocks) * (args.num_trials + 1)
    f1_scores = np.zeros(num_blocks)

    for current_block_id in range(num_blocks):
        if args.visualize_block_num > 0:
            current_block_id = args.visualize_block_num
        print(current_block_id)

        skills_tested = []

        skill_success_probabilities = np.ones(num_total_skills) * 0.5
        skill_failure_probabilities = np.ones(num_total_skills) * 0.5

        block_data = data[:,current_block_id]

        #print(x_y_thetas[np.nonzero(block_data),:])

        if args.visualize_block_num >= 0:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            p = ax.scatter3D(inputs[:,1], inputs[:,2], inputs[:,0], c=block_data[:].flatten(),vmin=0, vmax=1);
            ax.set_title('Actual Successes')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Num Samples')
            ax.view_init(azim=0, elev=90)
            fig.colorbar(p)
            plt.show()

        for i in range(args.num_trials):

            if args.sampling_method == 0:
                skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_total_skills * 0.03))
            elif args.sampling_method == 1:
                skill_num = pick_top_skill(skill_success_probabilities)
            elif args.sampling_method == 2:
                if i < 3:
                    skill_num = pick_most_uncertain(skill_success_probabilities)
                else:
                    skill_num = pick_random_skill_from_top_n(skill_success_probabilities, int(num_total_skills * 0.03))
            skills_tested.append(skill_num)

            current_input = inputs[skill_num]
            skill_id = int(current_input[0])

            xs = inputs[:,:3] - current_input[:3]
            thetas = np.arctan2(np.sin(inputs[:,3] - current_input[3]), np.cos(inputs[:,3] - current_input[3]))
            dists = inputs[:,4] - current_input[4]
            tilts = np.arctan2(np.sin(inputs[:,5] - current_input[5]), np.cos(inputs[:,5] - current_input[5]))
            relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

            new_skill_success_probabilities = np.ones(skill_success_probabilities.shape) * 0.5
            
            # if args.visualize_block_num > 0:
            #     fig = plt.figure()
            #     ax = plt.axes(projection='3d')

            #     p = ax.scatter3D(inputs[:,1], inputs[:,2], inputs[:,0], c=skill_success_probabilities.flatten(), vmin=0, vmax=1);
            #     ax.scatter3D(current_input[1], current_input[2], current_input[0], c='red');
            #     ax.set_title('Predicted skills after failure')
            #     ax.set_xlabel('x (m)')
            #     ax.set_ylabel('y (m)')
            #     ax.set_zlabel('Num Samples')
            #     ax.view_init(azim=0, elev=90)
            #     fig.colorbar(p)
            #     plt.show()

            if block_data[skill_num] == 1:
                if args.joint_cont_off:
                    key = suffices[skill_id]
                    num_inputs = skill_input_nums[skill_id]
                    new_skill_success_probabilities[500*skill_id:500*(skill_id+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*skill_id:500*(skill_id+1),:num_inputs]).reshape(-1)
                else:
                    for suffix_idx in range(num_skills):
                        if suffix_idx < skill_id:
                            suffix1 = suffices[suffix_idx]
                            suffix2 = suffices[skill_id]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                        elif suffix_idx == skill_id:
                            key = suffices[suffix_idx]
                            num_inputs = skill_input_nums[skill_id]
                        else:
                            suffix1 = suffices[skill_id]
                            suffix2 = suffices[suffix_idx]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                    
                        new_skill_success_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)
                
                if args.visualize_block_num >= 0:
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    p = ax.scatter3D(inputs[:,1], inputs[:,2], inputs[:,0], c=new_skill_success_probabilities);
                    ax.scatter3D(current_input[1], current_input[2], current_input[0], c='red');
                    ax.set_title('Predicted success skills after success')
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    ax.set_zlabel('Num Samples')
                    ax.view_init(azim=0, elev=90)
                    fig.colorbar(p)
                    plt.show()

                num_successes_each_block[current_block_id] += 1
                if first_success_each_block[current_block_id] == (args.num_trials + 1):
                    first_success_each_block[current_block_id] = i + 1
            else:
                if args.joint_cont_off:
                    key = suffices[skill_id]
                    num_inputs = skill_input_nums[skill_id]
                    new_skill_success_probabilities[500*skill_id:500*(skill_id+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*skill_id:500*(skill_id+1),:num_inputs]).reshape(-1)
                else:
                    for suffix_idx in range(num_skills):
                        if suffix_idx < skill_id:
                            suffix1 = suffices[suffix_idx]
                            suffix2 = suffices[skill_id]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                        elif suffix_idx == skill_id:
                            key = suffices[suffix_idx]
                            num_inputs = skill_input_nums[skill_id]
                        else:
                            suffix1 = suffices[skill_id]
                            suffix2 = suffices[suffix_idx]
                            key = suffix1 + '_' + suffix2
                            num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
                    
                        new_skill_success_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)
                
                if args.visualize_block_num >= 0:
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    p = ax.scatter3D(inputs[:,1], inputs[:,2], inputs[:,0], c=new_skill_success_probabilities);
                    ax.scatter3D(current_input[1], current_input[2], current_input[0], c='red');
                    ax.set_title('Predicted success skills after failure')
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    ax.set_zlabel('Num Samples')
                    ax.view_init(azim=0, elev=90)
                    fig.colorbar(p)
                    plt.show()

            skill_success_probabilities = skill_success_probabilities * new_skill_success_probabilities
            skill_failure_probabilities = skill_failure_probabilities * (1-new_skill_success_probabilities)

            skill_probabilities_sum = skill_success_probabilities + skill_failure_probabilities
            skill_success_probabilities = skill_success_probabilities / skill_probabilities_sum
            skill_failure_probabilities = skill_failure_probabilities / skill_probabilities_sum

            # print(np.max(skill_success_probabilities))
            # print(np.min(skill_success_probabilities))

            if args.visualize_block_num >= 0:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                p = ax.scatter3D(inputs[:,1], inputs[:,2], inputs[:,0], c=skill_success_probabilities.flatten());
                ax.scatter3D(current_input[1], current_input[2], current_input[0], c='red', vmin=0, vmax=1);
                ax.set_title('Skill probabilities after the network')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('Num Samples')
                ax.view_init(azim=0, elev=90)
                fig.colorbar(p)
                plt.show()

        predicted_successes = (skill_success_probabilities) > 0.5
        f1_scores[current_block_id] = f1_score(block_data, predicted_successes)


    print(num_successes_each_block)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_block), np.std(first_success_each_block)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_block),np.std(num_successes_each_block)))
    print(str(np.count_nonzero(first_success_each_block == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    