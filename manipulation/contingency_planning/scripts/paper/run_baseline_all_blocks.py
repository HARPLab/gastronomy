import argparse
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def pick_random_block_num(block_probabilities):
    normalized_probabilities = block_probabilities / np.sum(block_probabilities)

    random_number = np.random.random(1)
    block_num = 0

    for i in range(normalized_probabilities.shape[0]):
        if random_number < normalized_probabilities[i]:
            block_num = i
            break
        else:
            random_number -= normalized_probabilities[i]

    return block_num

def pick_random_block_from_top_10(block_probabilities):
    argsort = np.argsort(block_probabilities)
    top_10_probabilities = block_probabilities[argsort[-10:]]
    normalized_probabilities = top_10_probabilities / np.sum(top_10_probabilities)

    random_number = np.random.random(1)
    block_num = 0

    for i in range(normalized_probabilities.shape[0]):
        if random_number < normalized_probabilities[i]:
            block_num = i
            break
        else:
            random_number -= normalized_probabilities[i]

    return argsort[block_num-10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', '-n', type=int, default=100)
    parser.add_argument('--baseline_type', '-bt', type=str, default='baseline')
    args = parser.parse_args()

    data = np.load('baseline/pick_up/complete_data.npy')

    if args.baseline_type == 'baseline_same_friction':
        baseline_data = np.load('baseline_same_friction/pick_up/complete_data.npy')
    elif args.baseline_type == 'baseline_same_mass':
        baseline_data = np.load('baseline_same_mass/pick_up/complete_data.npy')

    num_successes_each_block_first_50 = np.zeros(101)
    num_successes_each_block_last_50 = np.zeros(101)
    f1 = np.zeros(101)
    first_success_each_block = np.ones(101) * 100

    for current_block_id in range(101):
        #print(current_block_id)
        #if np.sum(data[:,current_block_id]) > 50:
        skills_tested = []

        initial_block_affinities = np.ones(100)
        initial_block_probabilities = np.ones(100)
        division_factor = 0.0

        block_data = data[:,current_block_id]

        if args.baseline_type == 'baseline':
            if current_block_id == 0:
                baseline_data = data[:,1:]
            elif current_block_id == 100:
                baseline_data = data[:,:100]
            else:
                baseline_data = np.hstack((data[:,:current_block_id].reshape(2000,-1), data[:,(current_block_id+1):].reshape(2000,-1)))

        num_successes = 0

        for i in range(args.num_trials):
            random_block_num = pick_random_block_from_top_10(initial_block_affinities)
            random_skill_num = np.random.randint(500)

            while baseline_data[random_skill_num,random_block_num] == 0: # or random_skill_num in skills_tested:
                random_block_num = pick_random_block_from_top_10(initial_block_affinities)
                random_skill_num = np.random.randint(500)

            # random_block_num = pick_random_block_num(initial_block_probabilities)
            # successful_skills = np.nonzero(baseline_data[:,random_block_num])
            # new_skill_available = False
            # while np.sum(baseline_data[:,random_block_num]) == 0 or not new_skill_available:
            #     random_block_num = pick_random_block_num(initial_block_probabilities)
            #     successful_skills = np.nonzero(baseline_data[:,random_block_num])

            #     for skill_num in successful_skills[0]:
            #         if skill_num not in skills_tested:
            #             new_skill_available = True
            #             break
            
            # random_skill_num_idx = np.random.randint(len(successful_skills[0]))
            # random_skill_num = successful_skills[0][random_skill_num_idx]

            # while (baseline_data[random_skill_num,random_block_num] != 1) or random_skill_num in skills_tested:
            #     #random_block_num = pick_random_block_num(initial_block_probabilities)
            #     random_skill_num_idx = np.random.randint(len(successful_skills[0]))
            #     random_skill_num = successful_skills[0][random_skill_num_idx]

            skills_tested.append(random_skill_num)

            # if block_data[random_skill_num] == 1:
            #     #print('Trial ' + str(i) + ': Skill Succeeded')
            #     initial_block_affinities += baseline_data[random_skill_num]
            #     division_factor += 1
            #     num_successes += 1
            # else:
            #     #print('Trial ' + str(i) + ': Skill Failed')
            #     initial_block_affinities += (1-baseline_data[random_skill_num]) * 0.1
            #     division_factor += 0.1

            # initial_block_probabilities = initial_block_affinities / division_factor

            if block_data[random_skill_num] == 1:
                #print('Trial ' + str(i) + ': Skill Succeeded')
                current_block_data = baseline_data[random_skill_num]
                current_block_data[np.nonzero(current_block_data == 0)] = 0.3
                initial_block_affinities *= current_block_data

                if i < 50:
                    num_successes_each_block_first_50[current_block_id] += 1
                else:
                    num_successes_each_block_last_50[current_block_id] += 1

                if first_success_each_block[current_block_id] == 100:
                    first_success_each_block[current_block_id] = i
            else:
                #print('Trial ' + str(i) + ': Skill Failed')
                current_block_data = 1 - baseline_data[random_skill_num]
                current_block_data[np.nonzero(current_block_data == 0)] = 0.3
                initial_block_affinities *= current_block_data

            initial_block_affinities /= np.max(initial_block_affinities)

            # plt.bar(np.arange(100),initial_block_affinities)
            # plt.show()

            # for i in range(args.num_trials,100):

            #     random_block_num = pick_random_block_num(initial_block_affinities)
            #     random_skill_num = np.random.randint(2000)

            #     while baseline_data[random_skill_num,random_block_num] == 0 or random_skill_num in skills_tested:
            #         random_block_num = pick_random_block_num(initial_block_affinities)
            #         random_skill_num = np.random.randint(2000)
            #     # random_block_num = pick_random_block_num(initial_block_probabilities)
            #     # successful_skills = np.nonzero(baseline_data[:,random_block_num])
            #     # new_skill_available = False
            #     # while np.sum(baseline_data[:,random_block_num]) == 0 or not new_skill_available:
            #     #     random_block_num = pick_random_block_num(initial_block_probabilities)
            #     #     successful_skills = np.nonzero(baseline_data[:,random_block_num])

            #     #     for skill_num in successful_skills[0]:
            #     #         if skill_num not in skills_tested:
            #     #             new_skill_available = True
            #     #             break
            #     # random_skill_num_idx = np.random.randint(len(successful_skills[0]))
            #     # random_skill_num = successful_skills[0][random_skill_num_idx]

            #     # while (baseline_data[random_skill_num,random_block_num] != 1) or random_skill_num in skills_tested:
            #     #     #random_block_num = pick_random_block_from_top_10(initial_block_probabilities)
            #     #     random_skill_num_idx = np.random.randint(len(successful_skills[0]))
            #     #     random_skill_num = successful_skills[0][random_skill_num_idx]

            #     skills_tested.append(random_skill_num)

            #     if block_data[random_skill_num] == 1:
            #         #print('Trial ' + str(i) + ': Skill Succeeded')
            #         # initial_block_affinities += baseline_data[random_skill_num]
            #         # division_factor += 1
            #         num_successes += 1
            #         if first_success_each_block[current_block_id] == -1:
            #             first_success_each_block[current_block_id] = i - args.num_trials
            #     else:
            #         pass
            #         #print('Trial ' + str(i) + ': Skill Failed')
            #         # initial_block_affinities += (1-baseline_data[random_skill_num]) * 0.1
            #         # division_factor += 0.1

        print(num_successes_each_block_first_50[current_block_id] + num_successes_each_block_last_50[current_block_id])

        #num_successes_each_block[current_block_id] = num_successes
        skill_probabilities = np.sum(np.transpose(baseline_data) * initial_block_affinities.reshape(-1,1), axis=0)
        skill_probabilities /= np.sum(initial_block_affinities)

        predicted_successes = skill_probabilities > 0.5
        f1[current_block_id] = f1_score(data[:,current_block_id] == 1, predicted_successes)

        #print(f1[current_block_id])

    print('f1_mean = ' + str(np.mean(f1)))

    num_successes_each_block = num_successes_each_block_first_50 + num_successes_each_block_last_50

    print(str(np.mean(first_success_each_block)) + ' +- ' + str(np.std(first_success_each_block)))
    print(str(np.mean(num_successes_each_block)) + ' +- ' + str(np.std(num_successes_each_block)))
    print(str(np.mean(num_successes_each_block[np.nonzero(num_successes_each_block)])) + ' +- ' + str(np.std(num_successes_each_block[np.nonzero(num_successes_each_block)])))
    print(str(np.mean(first_success_each_block[np.nonzero(first_success_each_block < 100)])) + ' +- ' + str(np.std(first_success_each_block[np.nonzero(first_success_each_block < 100)])))
    print(str(np.count_nonzero(first_success_each_block == 100)))

    # We can set the number of bins with the `bins` kwarg
    plt.figure()
    plt.hist(num_successes_each_block_first_50, bins=50, color='red')
    plt.figure()
    plt.hist(num_successes_each_block_last_50, bins=50, color='blue')
    plt.show()

    # We can set the number of bins with the `bins` kwarg
    # plt.hist(np.sum(baseline_data, axis=0), bins=30)
    # plt.show()

        #initial_block_probabilities = initial_block_affinities / division_factor

    # for i in range(10,100):

    #     random_block_num = pick_random_block_from_top_10(initial_block_probabilities)
    #     random_skill_num = np.random.randint(2000)

    #     while (baseline_data[random_skill_num,random_block_num] != 1) or random_skill_num in skills_tested:
    #         random_block_num = pick_random_block_from_top_10(initial_block_probabilities)
    #         random_skill_num = np.random.randint(2000)

    #     skills_tested.append(random_skill_num)

    #     if block_data[random_skill_num] == 1:
    #         print('Trial ' + str(i) + ': Skill Succeeded')
    #         initial_block_affinities += baseline_data[random_skill_num]
    #         division_factor += 1
    #         num_successes += 1
    #     else:
    #         print('Trial ' + str(i) + ': Skill Failed')
    #         initial_block_affinities += (1-baseline_data[random_skill_num]) * 0.1
    #         division_factor += 0.1

    #     initial_block_probabilities = initial_block_affinities / division_factor


    # print(num_successes)

    # distances = distance.cdist(block_data.reshape(1,2000), np.transpose(baseline_data))
    # print(np.argmin(distances))
    # print(np.min(distances))

    # print(np.argmax(distances))
    # print(np.max(distances))

    # closest_block_id = np.argmax(initial_block_probabilities)

    # print(np.sum(block_data))
    # print('Closest block = ', closest_block_id)
    # print('Block Probability = ', initial_block_probabilities[closest_block_id])
    # print(initial_block_affinities[closest_block_id])
    # print('Block Affinities = ', distance.cdist(block_data.reshape(1,2000),baseline_data[:, closest_block_id].reshape(1,2000)))