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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', '-n', type=int, default=10)
    parser.add_argument('--contingency_nn_dir', type=str, default='cabinets/')
    parser.add_argument('--visualize_cabinet_num', '-vbn', type=int, default=-1)
    parser.add_argument('--sampling_method', '-s', type=int, default=0)
    args = parser.parse_args()

    data = np.load('cabinets/baseline_data.npy')

    success_contingency_nn = load_model(args.contingency_nn_dir + 'success_contingency_model.h5')
    failure_contingency_nn = load_model(args.contingency_nn_dir + 'failure_contingency_model.h5')

    x_y_zs = np.load('cabinets/successful_cabinet_inputs.npy')

    num_cabinets = data.shape[1]
    num_skills = data.shape[0]

    num_successes_each_cabinet = np.zeros(num_cabinets)
    first_success_each_cabinet = np.ones(num_cabinets) * (args.num_trials + 1)
    f1_scores = np.zeros(num_cabinets)

    for current_cabinet_id in range(num_cabinets):

        # success_contingency_nn = load_model(args.contingency_nn_dir + str(current_cabinet_id) + '_success_contingency_model.h5')
        # failure_contingency_nn = load_model(args.contingency_nn_dir + str(current_cabinet_id) + '_failure_contingency_model.h5')

        if args.visualize_cabinet_num > 0:
            current_cabinet_id = args.visualize_cabinet_num
        print(current_cabinet_id)

        skills_tested = []

        skill_success_probabilities = np.ones(num_skills) * 0.5
        skill_failure_probabilities = np.ones(num_skills) * 0.5

        cabinet_data = data[:,current_cabinet_id]

        #print(x_y_zs[np.nonzero(cabinet_data),:])

        if args.visualize_cabinet_num >= 0:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            p = ax.scatter(x_y_zs[:,0], x_y_zs[:,1], x_y_zs[:,2], c=cabinet_data);
            ax.set_title('Ground Truth')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
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

            current_input = x_y_zs[skill_num]

            relative_transforms = x_y_zs - current_input

            if cabinet_data[skill_num] == 1:
                new_skill_success_probabilities = success_contingency_nn.predict(relative_transforms).reshape(-1)
                   
                # if args.visualize_cabinet_num >= 0: 
                #     fig = plt.figure()
                #     ax = plt.axes()

                #     ax.scatter(relative_transforms[:,0], relative_transforms[:,1], c=new_skill_success_probabilities);
                #     #ax.scatter(current_input[0], current_input[1], c='red');
                #     ax.set_title('Surface plot')
                #     ax.set_xlabel('x (m)')
                #     ax.set_ylabel('y (m)')
                #     ax.set_zlabel('Num Samples')
                #     ax.view_init(azim=0, elev=90)
                #     plt.show()

                num_successes_each_cabinet[current_cabinet_id] += 1
                if first_success_each_cabinet[current_cabinet_id] == (args.num_trials + 1):
                    first_success_each_cabinet[current_cabinet_id] = i + 1
            else:
                new_skill_success_probabilities = failure_contingency_nn.predict(relative_transforms).reshape(-1)

                # if args.visualize_cabinet_num >= 0:
                #     fig = plt.figure()
                #     ax = plt.axes(projection='3d')

                #     ax.scatter(relative_transforms[:,0], relative_transforms[:,1], c=new_skill_success_probabilities);
                #     ax.scatter(current_input[0], current_input[1], c='red');
                #     ax.set_title('Surface plot')
                #     ax.set_xlabel('x (m)')
                #     ax.set_ylabel('y (m)')
                #     ax.set_zlabel('Num Samples')
                #     ax.view_init(azim=0, elev=90)
                #     plt.show()

            skill_success_probabilities = skill_success_probabilities * new_skill_success_probabilities
            skill_failure_probabilities = skill_failure_probabilities * (1-new_skill_success_probabilities)

            skill_probabilities_sum = skill_success_probabilities + skill_failure_probabilities
            skill_success_probabilities = skill_success_probabilities / skill_probabilities_sum
            skill_failure_probabilities = skill_failure_probabilities / skill_probabilities_sum

            if args.visualize_cabinet_num >= 0:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                p = ax.scatter(x_y_zs[:,0], x_y_zs[:,1], x_y_zs[:,2], c=skill_success_probabilities);
                ax.scatter(current_input[0], current_input[1], c='red');
                if i == 0:
                    ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skill')
                else:
                    ax.set_title('Skill Parameter Probabilities after ' + str(i+1) + ' skills')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('z (m)')
                fig.colorbar(p)
                
                #ax.view_init(azim=0, elev=90)
                plt.show()

        predicted_successes = (skill_success_probabilities) > 0.5
        f1_scores[current_cabinet_id] = f1_score(cabinet_data, predicted_successes)

    print('Avg first success = {:.3f} +- {:.3f}'.format(np.mean(first_success_each_cabinet), np.std(first_success_each_cabinet)))
    print('Avg num success out of ' + str(args.num_trials) + ' trials = {:.3f} +- {:.3f}'.format(np.mean(num_successes_each_cabinet),np.std(num_successes_each_cabinet)))
    print(str(np.count_nonzero(first_success_each_cabinet == (args.num_trials + 1))))
    print('Avg f1 score = {:.3f} +- {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))