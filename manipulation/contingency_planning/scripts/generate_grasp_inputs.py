import argparse
import numpy as np

import keras
from keras.models import load_model
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', '-s', type=str, default='franka_fingers')
    parser.add_argument('--num_iterations', '-ni', type=int, default=50000)
    parser.add_argument('--num_inputs', '-n', type=int, default=500)
    args = parser.parse_args()

    grasp_nn_model = load_model('same_blocks/pick_up/'+args.suffix+'_pick_up_only_trained_model.h5')

    if 'franka_fingers' in args.suffix or 'tongs_overhead' in args.suffix:
        inputs = get_random_x_y_thetas(args.num_iterations)
    elif 'tongs_side' in args.suffix or 'spatula_tilted' in args.suffix:
        inputs = get_random_x_y_theta_dist_tilts(args.num_iterations)
    elif 'spatula_flat' in args.suffix:
        inputs = get_random_x_y_theta_dists(args.num_iterations)

    probabilities = grasp_nn_model.predict(inputs)
    sorted_idx = np.argsort(-probabilities, 0)
    if 'franka_fingers' in args.suffix or 'tongs_overhead' in args.suffix:
        sorted_inputs = inputs[sorted_idx].reshape(-1,3)
    elif 'spatula_flat' in args.suffix:
        sorted_inputs = inputs[sorted_idx].reshape(-1,4)
    elif 'tongs_side' in args.suffix or 'spatula_tilted' in args.suffix:
        sorted_inputs = inputs[sorted_idx].reshape(-1,5)
    inputs = sorted_inputs[:args.num_inputs,:]
    print(np.sort(-probabilities, 0)[50])
    #np.save('data/'+args.suffix + '_inputs.npy', inputs)