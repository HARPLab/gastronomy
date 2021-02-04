# import the necessary packages
import numpy as np
import cv2
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='urdf_data/pick_up/')
    parser.add_argument('--suffix', '-s', type=str, default='franka_fingers')
    args = parser.parse_args()

    cropped_image_file_paths = glob.glob(args.data_dir + args.suffix + '/*_cropped.png')
    print(len(cropped_image_file_paths))

    images = np.zeros((0,100,100,1))
    actions = np.zeros((0,4))
    Y = np.zeros((0,1))

    i = 0
    j = 0
    for cropped_image_file in cropped_image_file_paths:
        print(cropped_image_file)
        i += 1
        print(i)
        # load the image and show it
        image = cv2.imread(cropped_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        file_name = cropped_image_file[cropped_image_file.rfind('/')+1:-14]
        trial_num = int(cropped_image_file[-13])
        try:
            urdf_data = np.load(cropped_image_file[:-14]+'_'+args.suffix+'.npz')

            if 'franka_fingers' in args.suffix or 'tongs_overhead' in args.suffix:
                inputs = urdf_data['x_y_thetas']
            elif 'tongs_side' in args.suffix or 'spatula_tilted' in args.suffix:
                inputs = urdf_data['x_y_theta_dist_tilts']
            elif 'spatula_flat' in args.suffix:
                inputs = urdf_data['x_y_theta_dists']

            initial_urdf_pose = urdf_data['initial_urdf_pose']
            post_release_urdf_pose = urdf_data['post_release_urdf_pose']
            incorrect_height_dif = post_release_urdf_pose[:,1] - initial_urdf_pose[:,1]

            if 'spatula' in args.suffix:
                post_grasp_urdf_pose = urdf_data['post_pick_up_urdf_pose']
            else:
                post_grasp_urdf_pose = urdf_data['post_grasp_urdf_pose']

            height_dif = post_grasp_urdf_pose[:,1] - initial_urdf_pose[:,1]

            height_thresh = 0.1

            successful_trials = height_dif > height_thresh
            incorrect_trials = incorrect_height_dif > height_thresh
            correct_trials = np.logical_and(successful_trials, np.logical_not(incorrect_trials))

            reshaped_gray = gray.reshape(1,100,100,1)
            actions = np.vstack((actions,inputs[trial_num*100:(trial_num+1)*100,:]))
            Y = np.vstack((Y,correct_trials[trial_num*100:(trial_num+1)*100].reshape(-1,1)))
            images = np.vstack((images,gray.reshape(1,100,100,1)))
        except:
            pass

    np.savez(args.data_dir + args.suffix + '_convolution_data.npz', actions=actions, images=images, Y=Y)