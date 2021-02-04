import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='same_blocks/pick_up/franka_fingers_contingency_data.npz')
    args = parser.parse_args()
    
    file_path = args.file_path
    data_idx = file_path.rfind('contingency_data')
    prefix = file_path[:data_idx]

    success_success_model_file_path = prefix + 'success_success_contingency_model.h5'
    success_failure_model_file_path = prefix + 'success_failure_contingency_model.h5'
    failure_success_model_file_path = prefix + 'failure_success_contingency_model.h5'
    failure_failure_model_file_path = prefix + 'failure_failure_contingency_model.h5'

    data = np.load(args.file_path)
    success_success_data = data['success_success_data']
    success_failure_data = data['success_failure_data']
    failure_success_data = data['failure_success_data']
    failure_failure_data = data['failure_failure_data']   
    input_dim = success_success_data.shape[1] 

    X = np.vstack((success_success_data,success_failure_data,failure_success_data,failure_failure_data))

    success_success_Y = np.concatenate((np.ones(success_success_data.shape[0]),np.zeros(success_failure_data.shape[0]),
                                        np.zeros(failure_success_data.shape[0]),np.zeros(failure_failure_data.shape[0])))
    success_failure_Y = np.concatenate((np.zeros(success_success_data.shape[0]),np.ones(success_failure_data.shape[0]),
                                        np.zeros(failure_success_data.shape[0]),np.zeros(failure_failure_data.shape[0])))
    failure_success_Y = np.concatenate((np.zeros(success_success_data.shape[0]),np.zeros(success_failure_data.shape[0]),
                                        np.ones(failure_success_data.shape[0]),np.zeros(failure_failure_data.shape[0])))
    failure_failure_Y = np.concatenate((np.zeros(success_success_data.shape[0]),np.zeros(success_failure_data.shape[0]),
                                        np.zeros(failure_success_data.shape[0]),np.ones(failure_failure_data.shape[0])))

    success_success_model = Sequential()
    success_success_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    success_success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    success_success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile success_model
    success_success_model.compile(loss='binary_crossentropy', optimizer='adam')

    success_success_model.fit(X,success_success_Y, epochs=5, batch_size=10, verbose=True, shuffle=True)

    success_success_model.save(success_success_model_file_path)

    success_failure_model = Sequential()
    success_failure_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    success_failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    success_failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile failure_model
    success_failure_model.compile(loss='binary_crossentropy', optimizer='adam')

    success_failure_model.fit(X,success_failure_Y, epochs=5, batch_size=10, verbose=True, shuffle=True)

    success_failure_model.save(success_failure_model_file_path)

    failure_success_model = Sequential()
    failure_success_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    failure_success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    failure_success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile success_model
    failure_success_model.compile(loss='binary_crossentropy', optimizer='adam')

    failure_success_model.fit(X,failure_success_Y, epochs=5, batch_size=10, verbose=True, shuffle=True)

    failure_success_model.save(failure_success_model_file_path)

    failure_failure_model = Sequential()
    failure_failure_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    failure_failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    failure_failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile failure_model
    failure_failure_model.compile(loss='binary_crossentropy', optimizer='adam')

    failure_failure_model.fit(X,failure_failure_Y, epochs=5, batch_size=10, verbose=True, shuffle=True)

    failure_failure_model.save(failure_failure_model_file_path)