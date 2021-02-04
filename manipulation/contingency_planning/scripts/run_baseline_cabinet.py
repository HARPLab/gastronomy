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
    parser.add_argument('--sampling_method', '-s', type=int, default=0)
    parser.add_argument('--visualize_cabinet_num', '-vbn', type=int, default=-1)
    args = parser.parse_args()

    data = np.load('cabinets/baseline_data.npy')
    x_y_thetas = np.load('cabinets/successful_cabinet_inputs.npy')

    num_cabinets = data.shape[1]
    num_skills = data.shape[0]

    num_successes_each_cabinet = np.zeros(num_cabinets)
    first_success_each_cabinet = np.ones(num_cabinets) * (args.num_trials + 1)
    f1_scores = np.zeros(num_cabinets)

    for current_cabinet_id in range(num_cabinets):
        if args.visualize_cabinet_num > 0:
            current_cabinet_id = args.visualize_cabinet_num
        print(current_cabinet_id)
        
        skills_tested = []

        cabinet_data = data[:,current_cabinet_id]

        if args.visualize_cabinet_num >= 0:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(x_y_thetas[:,0], x_y_thetas[:,1], c=cabinet_data);
            ax.set_title('Surface plot')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Num Samples')
            ax.view_init(azim=0, elev=90)
            plt.show()

        #baseline_data = data
        if current_cabinet_id == 0:
            baseline_data = data[:,1:]
        elif current_cabinet_id == (num_cabinets-1):
            baseline_data = data[:,:(num_cabinets-1)]
        else:
            baseline_data = np.hstack((data[:,:current_cabinet_id].reshape(num_skills,-1), data[:,(current_cabinet_id+1):].reshape(num_skills,-1)))

        baseline_data_num_cabinets = baseline_data.shape[1]

        cabinet_affinities = np.ones(baseline_data_num_cabinets)
        cabinet_probabilities = np.ones(baseline_data_num_cabinets)
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

            if cabinet_data[skill_num] == 1:
                cabinet_affinities += baseline_data[skill_num]
                num_successes_each_cabinet[current_cabinet_id] += 1
                if first_success_each_cabinet[current_cabinet_id] == (args.num_trials + 1):
                    first_success_each_cabinet[current_cabinet_id] = i + 1
            else:
                cabinet_affinities += (1-baseline_data[skill_num])

            cabinet_probabilities = cabinet_affinities / (i+1)

            skill_probabilities = np.sum(np.transpose(baseline_data) * cabinet_probabilities.reshape(-1,1), axis=0)
            skill_probabilities /= np.sum(cabinet_probabilities)

            if args.visualize_cabinet_num >= 0:
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
        f1_scores[current_cabinet_id] = f1_score(cabinet_data, predicted_successes)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_cabinet), np.std(first_success_each_cabinet)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_cabinet),np.std(num_successes_each_cabinet)))
    print(str(np.count_nonzero(first_success_each_cabinet == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    