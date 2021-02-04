import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import cv2
import glob
import numpy as np
import argparse

import keras
from keras.models import load_model
from random_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='urdf_data/pick_up/')
    args = parser.parse_args()

    skill_types = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_flat', 'spatula_tilted']
    skill_type_input_sizes = [3,3,5,4,5]

    convolution_models = {}
    npz_file_paths = {}

    for skill_type in skill_types:
        convolution_models[skill_type] = load_model(args.data_dir+skill_type+'_convolution_model.h5')
        npz_file_paths[skill_type] = glob.glob(args.data_dir+skill_type+'/*.npz')

    for franka_fingers_npz_file_path in npz_file_paths['franka_fingers']:
        shortened_file_name = franka_fingers_npz_file_path[franka_fingers_npz_file_path.rfind('/')+1:franka_fingers_npz_file_path.rfind('_franka_fingers.npz')]

        is_in_all_skills = np.zeros(len(skill_types)-1)
        for skill_type_idx in range(1,len(skill_types)):
            for npz_file_path in npz_file_paths[skill_types[skill_type_idx]]:
                if shortened_file_name in npz_file_path:
                    is_in_all_skills[skill_type_idx-1] = 1

        if np.sum(is_in_all_skills) == len(skill_types) - 1:
            urdf_data = {}
            for skill_type in skill_types:
                urdf_data[skill_type] = np.load(args.data_dir + skill_type + '/' + shortened_file_name + '_'+skill_type+'.npz')


            for i in range(4):
                for skill_type in skill_types:
                    if skill_type == 'franka_fingers' or skill_type == 'tongs_overhead':
                        inputs = urdf_data[skill_type]['x_y_thetas']
                    elif skill_type == 'tongs_side' or skill_type == 'spatula_tilted':
                        inputs = urdf_data[skill_type]['x_y_theta_dist_tilts']
                    elif skill_type == 'spatula_flat':
                        inputs = urdf_data[skill_type]['x_y_theta_dists']

                    cropped_image_file = args.data_dir + skill_type + '/' + shortened_file_name + '_' + str(i) + '_cropped.png'
                    image = cv2.imread(cropped_image_file)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    batch_images = np.repeat(gray.reshape(1,100,100,1), 100, axis=0)
                    current_inputs = inputs[100*i:100*(i+1)]
                    if skill_type == 'tongs_side':
                        print(np.count_nonzero(convolution_models[skill_type].predict((current_inputs, batch_images)) > 0.2))


    # print(len(cropped_image_file_paths))


    # images = np.zeros((0,100,100,1))
    # actions = np.zeros((0,4))
    # Y = np.zeros((0,1))

    # i = 0
    # j = 0
    # for cropped_image_file in cropped_image_file_paths:
    #     print(cropped_image_file)
    #     i += 1
    #     print(i)
    #     # load the image and show it
    #     image = 
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     file_name = cropped_image_file[cropped_image_file.rfind('/')+1:-14]
    #     trial_num = int(cropped_image_file[-13])
    #     try:
    #         print(cropped_image_file[:-14]+'_'+args.suffix+'.npz')
    #         urdf_data = np.load(cropped_image_file[:-14]+'_'+args.suffix+'.npz')

            

    #         initial_urdf_pose = urdf_data['initial_urdf_pose']
    #         post_release_urdf_pose = urdf_data['post_release_urdf_pose']
    #         incorrect_height_dif = post_release_urdf_pose[:,1] - initial_urdf_pose[:,1]

    #         if 'spatula' in args.suffix:
    #             post_grasp_urdf_pose = urdf_data['post_pick_up_urdf_pose']
    #         else:
    #             post_grasp_urdf_pose = urdf_data['post_grasp_urdf_pose']

    #         height_dif = post_grasp_urdf_pose[:,1] - initial_urdf_pose[:,1]

    #         height_thresh = 0.1

    #         successful_trials = height_dif > height_thresh
    #         incorrect_trials = incorrect_height_dif > height_thresh
    #         correct_trials = np.logical_and(successful_trials, np.logical_not(incorrect_trials))
    #         reshaped_gray = gray.reshape(1,100,100,1)
    #         actions = np.vstack((actions,inputs[trial_num*100:(trial_num+1)*100,:]))
    #         Y = np.vstack((Y,correct_trials[trial_num*100:(trial_num+1)*100].reshape(-1,1)))
    #         images = np.vstack((images,gray.reshape(1,100,100,1)))
    #     except:
    #         pass

    # np.savez(args.data_dir + args.suffix + '_convolution_data.npz', actions=actions, images=images, Y=Y)