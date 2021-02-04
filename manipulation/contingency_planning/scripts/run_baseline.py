import argparse
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sampling_methods import pick_random_skill_from_top_n, pick_most_uncertain, pick_top_skill

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', '-n', type=int, default=10)
    parser.add_argument('--baseline_type', '-bt', type=str, default='baseline_same_friction')
    parser.add_argument('--sampling_method', '-s', type=int, default=0)
    parser.add_argument('--visualize_block_num', '-vbn', type=int, default=-1)
    args = parser.parse_args()

    data = np.load('baseline/pick_up/complete_data.npy')
    x_y_thetas = np.load('data/franka_fingers_inputs.npy')

    if args.baseline_type == 'baseline_same_friction':
        baseline_data = np.load('baseline_same_friction/pick_up/complete_data.npy')
    elif args.baseline_type == 'baseline_same_mass':
        baseline_data = np.load('baseline_same_mass/pick_up/complete_data.npy')

    num_blocks = data.shape[1]
    num_skills = data.shape[0]

    num_successes_each_block = np.zeros(num_blocks)
    first_success_each_block = np.ones(num_blocks) * (args.num_trials + 1)
    f1_scores = np.zeros(num_blocks)

    for current_block_id in range(num_blocks):
        if args.visualize_block_num > 0:
            current_block_id = args.visualize_block_num
        print(current_block_id)
        
        skills_tested = []

        block_data = data[:,current_block_id]

        if args.visualize_block_num >= 0:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=block_data);
            ax.set_title('Surface plot')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Num Samples')
            ax.view_init(azim=0, elev=90)
            plt.show()

        if args.baseline_type == 'baseline':
            if current_block_id == 0:
                baseline_data = data[:,1:]
            elif current_block_id == (num_blocks-1):
                baseline_data = data[:,:(num_blocks-1)]
            else:
                baseline_data = np.hstack((data[:,:current_block_id].reshape(num_skills,-1), data[:,(current_block_id+1):].reshape(num_skills,-1)))

        baseline_data_num_blocks = baseline_data.shape[1]

        block_affinities = np.ones(baseline_data_num_blocks)
        block_probabilities = np.ones(baseline_data_num_blocks)
        skill_probabilities = np.ones(num_skills) * 0.5

        for i in range(args.num_trials):

            if args.sampling_method == 0:
                skill_num = pick_random_skill_from_top_n(skill_probabilities, int(num_skills * 0.03))
            elif args.sampling_method == 1:
                skill_num = pick_top_skill(skill_probabilities)
            elif args.sampling_method == 2:
                if i < 3:
                    skill_num = pick_most_uncertain(skill_probabilities)
                else:
                    skill_num = pick_random_skill_from_top_n(skill_probabilities, int(num_skills * 0.03))
            
            skills_tested.append(skill_num)

            if block_data[skill_num] == 1:
                block_affinities += baseline_data[skill_num]
                num_successes_each_block[current_block_id] += 1
                if first_success_each_block[current_block_id] == (args.num_trials + 1):
                    first_success_each_block[current_block_id] = i + 1
            else:
                block_affinities += (1-baseline_data[skill_num])

            block_probabilities = block_affinities / (i+1)

            skill_probabilities = np.sum(np.transpose(baseline_data) * block_probabilities.reshape(-1,1), axis=0)
            skill_probabilities /= np.sum(block_probabilities)

            if args.visualize_block_num >= 0:
                current_input = x_y_thetas[skill_num]

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=skill_probabilities);
                ax.scatter(current_input[0], current_input[1], c='red');
                ax.set_title('Surface plot')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('Num Samples')
                ax.view_init(azim=0, elev=90)
                plt.show()

        predicted_successes = (skill_probabilities) > 0.5
        f1_scores[current_block_id] = f1_score(block_data, predicted_successes)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_block), np.std(first_success_each_block)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_block),np.std(num_successes_each_block)))
    print(str(np.count_nonzero(first_success_each_block == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    