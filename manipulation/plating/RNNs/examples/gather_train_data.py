import numpy as np
import argparse

import rnns.preprocess as preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', '-d', type=str,
                        default='../../Darknet/images/rnn_training/straight_lines')
    parser.add_argument('--image_resolution', '-r', type=tuple, default=(416,416))
    parser.add_argument('--sequence_length', '-l', type=int, default=8)
    parser.add_argument('--label_type', '-t', type=str, default='center')
    parser.add_argument('--label_file_names', '-n', type=str, default='food_boxes*')
    parser.add_argument('--save_path', '-s', type=str, default='../)
    parser.add_argument('--fake_train_data_path', '-f', type=str, default=None)
    args = parser.parse_args()

    train_path = args.train_data_path
    resolution = args.image_resolution
    training_length = args.sequence_length
    label_type = args.label_type
    label_names = args.label_file_names
    save_path =  args.save_path
    fake_data = args.fake_train_data_path

    print('Collecting the training data')
    test = preprocess.PrepTrainData(train_path, resolution, label_names)
    sequences = test.collect_images()
    labels, _ = test.create_labels(label_type=label_type)
    print('Finished')
    print('Sorting and processing the training data')
    train_data, train_labels = preprocess.make_train_batches(sequences, labels, training_length)

    if fake_data is not None:
        # use augmented data set for training if provided
        print('Gathering the artificial training data')
        fake_stuff = np.load(fake_data)
        more_data = fake_stuff['train_data']
        more_labels = fake_stuff['train_labels']

        data = np.concatenate((train_data, more_data), axis=0)
        labels = np.concatenate((train_labels, more_labels), axis=0)

        print(f'Saving the data to: {save_path}')
        import ipdb; ipdb.set_trace()
        np.savez(f'{save_path}/train_data.npz', train_data=data,
                train_labels=labels)

    else:
        # for saving just the real data files
        np.savez(f'{save_path}/train_data_OG.npz', train_data=train_data,
                train_labels=train_labels)

    print('Finished')