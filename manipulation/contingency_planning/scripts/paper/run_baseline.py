import argparse
import numpy as np
from scipy.spatial import distance

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
    parser.add_argument('--block_num', '-b', type=int, default=100)
    parser.add_argument('--num_trials', '-n', type=int, default=10)
    parser.add_argument('--baseline_type', '-bt', type=str, default='baseline')
    args = parser.parse_args()

    data = np.load('baseline/pick_up/complete_data.npy')

    skills_tested = []

    initial_block_affinities = np.zeros(100)
    initial_block_probabilities = np.ones(100)
    division_factor = 0.0

    current_block_id = args.block_num

    block_data = data[:,current_block_id]

    if args.baseline_type == 'baseline':
        if current_block_id == 0:
            baseline_data = data[:,1:]
        elif current_block_id == 100:
            baseline_data = data[:,:100]
        else:
            baseline_data = np.hstack((data[:,:current_block_id].reshape(2000,-1), data[:,(current_block_id+1):].reshape(2000,-1)))

    elif args.baseline_type == 'baseline_same_friction':
        baseline_data = np.load('baseline_same_friction/pick_up/complete_data.npy')
    elif args.baseline_type == 'baseline_same_mass':
        baseline_data = np.load('baseline_same_mass/pick_up/complete_data.npy')
    else:
        print('Invalid baseline_type.')

    print(baseline_data.shape)


    num_successes = 0

    for i in range(args.num_trials):

        random_block_num = pick_random_block_num(initial_block_probabilities)
        random_skill_num = np.random.randint(2000)

        while (baseline_data[random_skill_num,random_block_num] != 1) or random_skill_num in skills_tested:
            random_block_num = pick_random_block_num(initial_block_probabilities)
            random_skill_num = np.random.randint(2000)

        skills_tested.append(random_skill_num)

        if block_data[random_skill_num] == 1:
            print('Trial ' + str(i) + ': Skill Succeeded')
            initial_block_affinities += baseline_data[random_skill_num]
            division_factor += 1
            num_successes += 1
        else:
            print('Trial ' + str(i) + ': Skill Failed')
            initial_block_affinities += (1-baseline_data[random_skill_num]) * 0.1
            division_factor += 0.1

        initial_block_probabilities = initial_block_affinities / division_factor

    for i in range(args.num_trials,100):

        random_block_num = pick_random_block_num(initial_block_probabilities)
        random_skill_num = np.random.randint(2000)

        while (baseline_data[random_skill_num,random_block_num] != 1) or random_skill_num in skills_tested:
            random_block_num = pick_random_block_num(initial_block_probabilities)
            random_skill_num = np.random.randint(2000)

        skills_tested.append(random_skill_num)

        if block_data[random_skill_num] == 1:
            print('Trial ' + str(i) + ': Skill Succeeded')
            # initial_block_affinities += baseline_data[random_skill_num]
            # division_factor += 1
            num_successes += 1
        else:
            print('Trial ' + str(i) + ': Skill Failed')
            # initial_block_affinities += (1-baseline_data[random_skill_num]) * 0.1
            # division_factor += 0.1

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


    print(num_successes)

    distances = distance.cdist(block_data.reshape(1,2000), np.transpose(baseline_data))
    print(np.argmin(distances))
    print(np.min(distances))

    print(np.argmax(distances))
    print(np.max(distances))

    closest_block_id = np.argmax(initial_block_probabilities)

    print(np.sum(block_data))
    print('Closest block = ', closest_block_id)
    print('Block Probability = ', initial_block_probabilities[closest_block_id])
    print(initial_block_affinities[closest_block_id])
    print('Block Affinities = ', distance.cdist(block_data.reshape(1,2000),baseline_data[:, closest_block_id].reshape(1,2000)))