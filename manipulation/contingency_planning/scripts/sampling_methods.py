import numpy as np

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

def pick_random_skill_from_top_n(skill_probabilities, top_n):
    argsort = np.argsort(skill_probabilities)
    top_n_probabilities = skill_probabilities[argsort[-top_n:]]
    normalized_probabilities = top_n_probabilities / np.sum(top_n_probabilities)

    random_number = np.random.random(1)
    skill_num = 0

    for i in range(normalized_probabilities.shape[0]):
        if random_number < normalized_probabilities[i]:
            skill_num = i
            break
        else:
            random_number -= normalized_probabilities[i]

    return argsort[skill_num-top_n]

def pick_most_uncertain(skill_probabilities):
    argsort = np.argsort(np.abs(skill_probabilities - 0.5))

    return argsort[0]

def pick_top_skill(skill_probabilities):
    argsort = np.argsort(skill_probabilities)

    return argsort[-1]

def pick_most_informative(skill_success_probabilities, success_success_probabilities, success_failure_probabilities,
                          skill_failure_probabilitites, failure_success_probabilities, failure_failure_probabilities):
    original_entropies = -(skill_success_probabilities.reshape(-1,1) * np.log(skill_success_probabilities.reshape(-1,1)) + 
                           skill_failure_probabilitites.reshape(-1,1) * np.log(skill_failure_probabilitites.reshape(-1,1)))

    new_skill_success_success_probabilities = (skill_success_probabilities.reshape(1,-1) * success_success_probabilities).reshape(-1,1)
    new_skill_success_failure_probabilities = (skill_failure_probabilities.reshape(1,-1) * success_failure_probabilities).reshape(-1,1)
    new_skill_failure_success_probabilities = (skill_success_probabilities.reshape(1,-1) * failure_success_probabilities).reshape(-1,1)
    new_skill_failure_failure_probabilities = (skill_failure_probabilities.reshape(1,-1) * failure_failure_probabilities).reshape(-1,1)

    new_skill_success_probabilities_sum = new_skill_success_success_probabilities + new_skill_success_failure_probabilities
    new_skill_failure_probabilities_sum = new_skill_failure_success_probabilities + new_skill_failure_failure_probabilities

    new_skill_success_success_probabilities = new_skill_success_success_probabilities / new_skill_success_probabilities_sum
    new_skill_success_failure_probabilities = new_skill_success_failure_probabilities / new_skill_success_probabilities_sum
    new_skill_failure_success_probabilities = new_skill_failure_success_probabilities / new_skill_failure_probabilities_sum
    new_skill_failure_failure_probabilities = new_skill_failure_failure_probabilities / new_skill_failure_probabilities_sum

    information_gain = (np.repeat(original_entropies.reshape(-1,1), 2000, axis=0) + 0.5 * #np.repeat(skill_success_probabilities.reshape(-1,1), 2000, axis=1).reshape(-1,1) * 
                       (new_skill_success_success_probabilities * np.log(new_skill_success_success_probabilities) + new_skill_success_failure_probabilities * np.log(new_skill_success_failure_probabilities))
                       + 0.5 * #np.repeat(skill_failure_probabilities.reshape(-1,1), 2000, axis=1).reshape(-1,1) * 
                       (new_skill_failure_success_probabilities * np.log(new_skill_failure_success_probabilities) + new_skill_failure_failure_probabilities * np.log(new_skill_failure_failure_probabilities)))

    best_information_gain_skill = 0
    best_information_gain = 0

    for i in range(500):
        total_information_gain = np.sum(information_gain[i*2000:(i+1) * 2000])

        if total_information_gain > best_information_gain:
            best_information_gain_skill = i
            best_information_gain = total_information_gain

    #print(best_information_gain)
    return best_information_gain_skill