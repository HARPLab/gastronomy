import numpy as np
from scipy.spatial import distance
import argparse

import keras
from keras.models import load_model

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns

def pick_random_skill_num(skill_probabilities):
    normalized_probabilities = skill_probabilities / np.sum(skill_probabilities)

    random_number = np.random.random(1)
    skill_num = 0

    for i in range(normalized_probabilities.shape[0]):
        if random_number < normalized_probabilities[i]:
            skill_num = i
            break
        else:
            random_number -= normalized_probabilities[i]

    return skill_num

def pick_random_skill_from_top_100(skill_probabilities):
    argsort = np.argsort(skill_probabilities)
    top_100_probabilities = skill_probabilities[argsort[-100:]]
    normalized_probabilities = top_100_probabilities / np.sum(top_100_probabilities)

    random_number = np.random.random(1)
    skill_num = 0

    for i in range(normalized_probabilities.shape[0]):
        if random_number < normalized_probabilities[i]:
            skill_num = i
            break
        else:
            random_number -= normalized_probabilities[i]

    return argsort[skill_num-100]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--contingency_nn_dir', type=str, default='same_block_data/contingency_data/')
    parser.add_argument('--joint_cont_off', '-j', action='store_true')
    parser.add_argument('--num_trials', '-n', type=int, default=100)
    parser.add_argument('--franka_nn', type=str, default='same_block_data/franka_fingers_pick_up_only_trained_model.h5')
    parser.add_argument('--tong_overhead_nn', type=str, default='same_block_data/tongs_overhead_pick_up_only_trained_model.h5')
    parser.add_argument('--tong_side_nn', type=str, default='same_block_data/tongs_side_pick_up_only_trained_model.h5')
    parser.add_argument('--spatula_tilted_nn', type=str, default='same_block_data/spatula_tilted_with_flip_pick_up_only_trained_model.h5')
    args = parser.parse_args()

    franka_nn_model = load_model(args.franka_nn)
    tong_overhead_nn_model = load_model(args.tong_overhead_nn)
    tong_side_nn_model = load_model(args.tong_side_nn)
    spatula_tilted_nn_model = load_model(args.spatula_tilted_nn)

    data = np.load('baseline/pick_up/complete_data.npy')

    suffices = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_tilted_with_flip']
    skill_input_nums = [4, 4, 6, 6]
    num_skills = len(suffices)

    contingency_nns = {}

    for suffix_idx1 in range(num_skills):
        for suffix_idx2 in range(suffix_idx1, num_skills):
            if (suffix_idx1 == suffix_idx2):
                key = suffices[suffix_idx1]
                contingency_nns[key+'_success'] = load_model(args.contingency_nn_dir + key + '_pick_up_only_success_contingency_model_100.h5')
                contingency_nns[key+'_failure'] = load_model(args.contingency_nn_dir + key + '_pick_up_only_failure_contingency_model_100.h5')
            else:
                suffix1 = suffices[suffix_idx1]
                suffix2 = suffices[suffix_idx2]
                key = suffix1 + '_' + suffix2
                contingency_nns[key+'_success'] = load_model(args.contingency_nn_dir + key + '_pick_up_only_success_contingency_model_100.h5')
                contingency_nns[key+'_failure'] = load_model(args.contingency_nn_dir + key + '_pick_up_only_failure_contingency_model_100.h5')

    sorted_x_y_thetas = np.load('baseline/franka_fingers_500.npy')
    franka_x_y_thetas = np.hstack((np.zeros((500,1)), sorted_x_y_thetas, np.zeros((500,2))))
    franka_fingers_probs = franka_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_thetas = np.load('baseline/tongs_overhead_500.npy')
    tong_overhead_x_y_thetas = np.hstack((np.ones((500,1)), sorted_x_y_thetas, np.zeros((500,2))))
    tong_overhead_probs = tong_overhead_nn_model.predict(sorted_x_y_thetas)

    sorted_x_y_theta_dist_tilts = np.load('baseline/tongs_side_500.npy')
    tong_side_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 2, sorted_x_y_theta_dist_tilts))
    tong_side_probs = tong_side_nn_model.predict(sorted_x_y_theta_dist_tilts)

    sorted_x_y_theta_dist_tilts = np.load('baseline/spatula_tilted_500.npy')
    spatula_tilted_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 3, sorted_x_y_theta_dist_tilts))
    spatula_tilted_probs = spatula_tilted_nn_model.predict(sorted_x_y_theta_dist_tilts)

    skill_probabilities = np.ones(2000)
    #skill_probabilities = np.vstack((franka_fingers_probs,tong_overhead_probs,tong_side_probs,spatula_tilted_probs)).reshape(-1)
    sorted_x_y_thetas = np.vstack((franka_x_y_thetas, tong_overhead_x_y_thetas, tong_side_x_y_theta_dist_tilts, spatula_tilted_x_y_theta_dist_tilts))

    num_successes_each_block = np.zeros(101)
    first_success_each_block = np.ones(101) * 100

    visualization_block_id = 14

    for current_block_id in range(101):
        current_block_id = visualization_block_id
        print(current_block_id)
        #if np.sum(data[:,current_block_id]) > 50:

        skills_tested = []

        initial_skill_probabilities = skill_probabilities.copy()#np.ones(2000)

        num_successes = 0

        if current_block_id == visualization_block_id:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            p = ax.scatter3D(sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], sorted_x_y_thetas[:,0], c=data[:,current_block_id].flatten());
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Skill id')
            fig.colorbar(p, ax=ax)
            plt.show()


        for i in range(args.num_trials):

            random_skill_num = pick_random_skill_num(initial_skill_probabilities)

            # while random_skill_num in skills_tested:
            #     random_skill_num = pick_random_skill_num(initial_skill_probabilities)

            # skills_tested.append(random_skill_num)

            current_input = sorted_x_y_thetas[random_skill_num]

            skill_id = int(current_input[0])

            xs = sorted_x_y_thetas[:,:3] - current_input[:3]
            thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - current_input[3]), np.cos(sorted_x_y_thetas[:,3] - current_input[3]))
            dists = sorted_x_y_thetas[:,4] - current_input[4]
            tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,5] - current_input[5]), np.cos(sorted_x_y_thetas[:,5] - current_input[5]))
            relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

            new_probabilities = np.ones(initial_skill_probabilities.shape)

            if args.joint_cont_off:
                key = suffices[skill_id]
                num_inputs = skill_input_nums[skill_id]
                if data[random_skill_num,current_block_id] == 1:
                    new_probabilities[500*skill_id:500*(skill_id+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*skill_id:500*(skill_id+1),:num_inputs]).reshape(-1)
                else:
                    new_probabilities[500*skill_id:500*(skill_id+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*skill_id:500*(skill_id+1),:num_inputs]).reshape(-1)
                new_probabilities[500*skill_id:500*(skill_id+1)] /= np.max(new_probabilities[500*skill_id:500*(skill_id+1)])
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
                
                    if data[random_skill_num,current_block_id] == 1:
                        new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)
                    else:
                        new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)
                    new_probabilities[500*skill_id:500*(skill_id+1)] /= np.max(new_probabilities[500*skill_id:500*(skill_id+1)])

            initial_skill_probabilities = initial_skill_probabilities * new_probabilities
            initial_skill_probabilities /= np.max(initial_skill_probabilities)

            if data[random_skill_num,current_block_id] == 1:
                #print('Trial ' + str(i) + ': Skill Succeeded')
                num_successes += 1
                if first_success_each_block[current_block_id] == 100:
                    first_success_each_block[current_block_id] = i
            else:
                pass
                #print('Trial ' + str(i) + ': Skill Failed')

            if current_block_id == visualization_block_id:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                p = ax.scatter3D(sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], sorted_x_y_thetas[:,0], c=initial_skill_probabilities[:].flatten());
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('Skill id')
                fig.colorbar(p, ax=ax)
                plt.show()
            
        # argsort = np.argsort(initial_skill_probabilities)    
        # for skill_num in argsort[-100:]:
        #     if data[skill_num,current_block_id]:
        #         num_successes += 1

        # for i in range(100):

        #     random_skill_num = pick_random_skill_num(initial_skill_probabilities)

        #     # while random_skill_num in skills_tested:
        #     #     random_skill_num = pick_random_skill_num(initial_skill_probabilities)

        #     skills_tested.append(random_skill_num)

        #     # current_input = sorted_x_y_thetas[random_skill_num]

        #     # skill_id = int(current_input[0])

        #     # xs = sorted_x_y_thetas[:,:3] - current_input[:3]
        #     # thetas = np.arctan2(np.sin(sorted_x_y_thetas[:,3] - current_input[3]), np.cos(sorted_x_y_thetas[:,3] - current_input[3]))
        #     # dists = sorted_x_y_thetas[:,4] - current_input[4]
        #     # tilts = np.arctan2(np.sin(sorted_x_y_thetas[:,5] - current_input[5]), np.cos(sorted_x_y_thetas[:,5] - current_input[5]))
        #     # relative_transforms = np.hstack((xs, thetas.reshape(-1,1), dists.reshape(-1,1), tilts.reshape(-1,1)))

        #     # new_probabilities = np.ones(initial_skill_probabilities.shape)

        #     # for suffix_idx in range(num_skills):
        #     #     if suffix_idx < skill_id:
        #     #         suffix1 = suffices[suffix_idx]
        #     #         suffix2 = suffices[skill_id]
        #     #         key = suffix1 + '_' + suffix2
        #     #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
        #     #     elif suffix_idx == skill_id:
        #     #         key = suffices[suffix_idx]
        #     #         num_inputs = skill_input_nums[skill_id]
        #     #     else:
        #     #         suffix1 = suffices[skill_id]
        #     #         suffix2 = suffices[suffix_idx]
        #     #         key = suffix1 + '_' + suffix2
        #     #         num_inputs = max(skill_input_nums[suffix_idx], skill_input_nums[skill_id])
            
        #     #     if data[random_skill_num,current_block_id] == 1:
        #     #         new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_success'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)
        #     #     else:
        #     #         new_probabilities[500*suffix_idx:500*(suffix_idx+1)] = contingency_nns[key+'_failure'].predict(relative_transforms[500*suffix_idx:500*(suffix_idx+1),:num_inputs]).reshape(-1)

        #     # initial_skill_probabilities = initial_skill_probabilities * new_probabilities
        #     # initial_skill_probabilities /= np.max(initial_skill_probabilities)

        #     if data[random_skill_num,current_block_id] == 1:
        #         #print('Trial ' + str(i) + ': Skill Succeeded')
        #         num_successes += 1
        #         if first_success_each_block[current_block_id] == 100:
        #             first_success_each_block[current_block_id] = i
        #     else:
        #         pass
        #         #print('Trial ' + str(i) + ': Skill Failed')

        num_successes_each_block[current_block_id] = num_successes

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # p = ax.scatter3D(sorted_x_y_thetas[:,1], sorted_x_y_thetas[:,2], sorted_x_y_thetas[:,0], c=data[:,current_block_id].flatten());
        # ax.set_xlabel('x (m)')
        # ax.set_ylabel('y (m)')
        # ax.set_zlabel('Skill id')
        # fig.colorbar(p, ax=ax)
        # plt.show()

    # We can set the number of bins with the `bins` kwarg
    plt.hist(first_success_each_block, bins=30)
    plt.show()

    #actual_successes = num_successes_each_block[np.nonzero(num_successes_each_block)[0]]
    print(str(np.mean(first_success_each_block)) + ' +- ' + str(np.std(first_success_each_block)))
    print(str(np.mean(num_successes_each_block)) + ' +- ' + str(np.std(num_successes_each_block)))
    print(str(np.mean(first_success_each_block[np.nonzero(first_success_each_block < 100)])) + ' +- ' + str(np.std(first_success_each_block[np.nonzero(first_success_each_block < 100)])))
    print(str(np.count_nonzero(first_success_each_block == 100)))