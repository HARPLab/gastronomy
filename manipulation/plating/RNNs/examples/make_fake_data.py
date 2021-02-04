import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

import rnns.fake_data as fake_data
import rnns.image_utils as image_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_sequences', type=int, default=970)
    parser.add_argument('-l', '--sequence_length', type=int, default=13)
    parser.add_argument('-p', '--sequence_pattern_types', type=list, default=[6],
                        help='''A list of indexes specifying which patterns to use. See
                             fake_data.PlacingPatterns.place_pattern. Format is: -p 1,2,4''')
    parser.add_argument('-s', '--save_path', type=str, default='./')
    parser.add_argument('-r', '--reference_data_path', default='./data/reference_images')
    parser.add_argument('-i', '--image_resolution', type=int, default=(224,224))#416))
    parser.add_argument('-t', '--label_type', type=str, default='yolo',
                        help="Format to return the labels in. Can be 'mask', 'center', or 'yolo'")
    parser.add_argument('-a', '--placement_area', type=list, default=[520,135,800,355], #[173, 138, 256, 204],
                        help='pixel locations of the min and max corners of the bounding box' \
                             'around the placement area. Formatted as [xmin, ymin, xmax, ymax]')
    args = parser.parse_args()
    """
    The intrinsics for the camera (These might be a bit off, didn't recorded these values exactly when images were taken):
    {"_frame": "azure_kinect_overhead", "_fx": 972.31787109375, "_fy": 971.8189086914062, "_cx": 1022.3043212890625, "_cy": 777.7421875, "_skew": 0.0, "_height": 1536, "_width": 2048, "_K": 0}

    Extrinsics:
    azure_kinect_overhead
    world
    0.801322 -0.025752 0.851278
    -0.025885 0.999502 -0.018068
    0.998471 0.026733 0.048394
    0.048853 -0.016787 -0.998665
    #TODO I don't think I have any depth images though...
    """

    num_sequences = args.number_sequences
    sequence_length = args.sequence_length
    save_path  = args.save_path
    
    ref_path = args.reference_data_path
    resolution = args.image_resolution
    label_type = args.label_type
    if args.sequence_pattern_types is not None:
        patterns = args.sequence_pattern_types
        patterns[:] = [int(c) for c in patterns if c != ',']
    else:
        patterns = args.sequence_pattern_types
    
    placement_area = args.placement_area #location of cutting board
    if placement_area is None:
        placement_area = np.array([0, 0, resolution[0], resolution[1]])

    foreground_dim_mean = 50
    foreground_dim_std = 4
    # TODO might want to put the below inside the run loop and check dynamically
    max_obj_size = foreground_dim_mean + 5* foreground_dim_std

    # mean_placement_offset = 20
    # std_placement_offset = 5


    # Collect the foreground and background images
    foregrounds, backgrounds = fake_data.collect_ref_images(ref_path, resize=False)
    # Crop the background images so they only contain the cutting board
    backgrounds = np.array(backgrounds)[:, placement_area[1]:placement_area[3], placement_area[0]:placement_area[2], :]
    # update the placement area
    placement_area = np.array([0,0,backgrounds.shape[2],backgrounds.shape[1]])
    
    # Add an alpga layer to each of the foreground images
    for key in foregrounds.keys():
        for i in range(len(foregrounds[key])):
            foregrounds[key][i] = image_utils.rgb_to_rgba_mask(foregrounds[key][i], mask_value=0, axis=2)

    # makes sure not to place objects outside of the specified location
    work_area = np.array([placement_area[0]+max_obj_size//2, placement_area[1]+max_obj_size//2, 
                        placement_area[2]-max_obj_size//2, placement_area[3]-max_obj_size//2])

    # generate the training data
    placements = fake_data.PlacingPatterns(backgrounds=backgrounds, 
                                           foregrounds=foregrounds,
                                           work_area=work_area,
                                           image_shape=resolution)

    # change the placing patterns to use here
    training_sequences, training_labels = placements.run(num_samples=num_sequences,
                                                sequence_length=sequence_length,
                                                patterns=patterns,
                                                foreground_dim_mean=foreground_dim_mean,
                                                foreground_dim_std=foreground_dim_std,
                                                label_type=label_type
    )

    # save data
    if save_path[-4:] == '.npz':
        print(f'Saving the data as: {save_path}')
        #TODO plate_bbox is hard coded right now, but you should upate PlacingPatterns to do this.
        #TODO Need to change this if you change the image size
        np.savez(save_path, train_data=training_sequences,
                train_labels=training_labels, plate_bbox=np.array([3,29,215,198]))
    else:
        print(f'Saving the data as pickle files at: {save_path}')
        # save as pickle instead
        with open(f'{save_path}/fake_train_data.pickle', 'wb') as f:
            pickle.dump(training_sequences, f)
        with open(f'{save_path}/fake_train_labels.pickle', 'wb') as f:
            pickle.dump(training_labels, f)   
    
    print('Finished')
