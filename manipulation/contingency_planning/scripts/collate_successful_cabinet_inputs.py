import argparse
import numpy as np
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-f', type=str, default='cabinets/')
    args = parser.parse_args()

    files = glob.glob(args.data_dir+'successful_cabinet_*.npy')

    successful_data = np.zeros((500,0))

    for file in files:
        data = np.load(file)

        successful_data = np.hstack((successful_data,data.reshape(-1,1)))

    successful_data_sum = np.sum(successful_data, axis=1)
    successful_input_indices = np.nonzero(successful_data_sum)

    print(np.count_nonzero(successful_data_sum))

    baseline_data = successful_data[successful_input_indices[0]]

    inputs = np.load(args.data_dir + 'cabinet_inputs.npy')
    successful_inputs = inputs[successful_input_indices[0]]

    np.save(args.data_dir + 'baseline_data.npy', baseline_data)
    np.save(args.data_dir + 'successful_cabinet_inputs.npy', successful_inputs)