import numpy as np
import argparse
import pickle

import rnns.fake_data as fake_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_sequences', '-n', type=int, default=970)
    parser.add_argument('--sequence_length', '-l', type=int, default=12)
    parser.add_argument('--sequence_pattern_types', '-p', type=list, default=[6])
    parser.add_argument('--save_path', '-s', type=str, default='../')
    parser.add_argument('--reference_data_path', '-r',
                        default='../../Darknet/images/test04/images')
    parser.add_argument('--image_resolution', '-i', type=tuple, default=(416,416))
    parser.add_argument('--label_type', '-t', type=str, default='center')
    parser.add_argument('--label_file_names', '-f', type=str, default='food_boxes*')
    parser.add_argument('--placement_area', '-a', type=list, default=[173, 138, 256, 204])
    args = parser.parse_args()

    num_sequences = args.number_sequences
    sequence_length = args.sequence_length
    save_path  = args.save_path
    
    ref_path = args.reference_data_path
    resolution = args.image_resolution
    label_names = args.label_file_names
    label_type = args.label_type
    patterns = args.sequence_pattern_types
    work_area = args.placement_area #location of cutting board
    
    foreground, background, foreground_id, alpha_masks = fake_data.reference_images(
        ref_path, resolution, label_names)
    # NOTE: might want to use the below line and pick out specific reference images to use. Not all of them will turn out good
    # foreground = [foreground[0],foreground[22],foreground[33],]
    
    training_data, training_labels = fake_data.make_train_data(resolution, num_sequences, background, 
                    foreground, foreground_id, sequence_length,
                    location=work_area, masks=alpha_masks, 
                    label_type=label_type, patterns=patterns)

    if save_path[-4:] == '.npz':
        print(f'Saving the data as: {save_path}')
        np.savez(save_path, train_data=training_data,
                train_labels=training_labels)

    else:
        print(f'Saving the data as pickle files at: {save_path}')
        # save as pickle instead
        with open(f'{save_path}/fake_train_data.pickle', 'wb') as f:
            pickle.dump(training_data, f)
        with open(f'{save_path}/fake_train_labels.pickle', 'wb') as f:
            pickle.dump(training_labels, f)   
    
    print('Finished')

    # # save all the reference images produced
    # im_num = 0
    # for stuff in background:
    #     utils.save_as_img(stuff, f'{save_path}/background{im_num}')
    #     im_num += 1
    # im_num = 0
    # for stuff in foreground:
    #     utils.save_as_img(stuff, f'{save_path}/foreground{im_num}')
    #     im_num += 1