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

    success_model_file_path = prefix + 'success_contingency_model.h5'
    failure_model_file_path = prefix + 'failure_contingency_model.h5'

    data = np.load(args.file_path)
    X = data['X']
    input_dim = X.shape[1]
    success_Y = data['success_Y']
    failure_Y = data['failure_Y']
    # failure_success_Y = 1 - data['success_Y']
    # failure_failure_Y = 1 - data['failure_Y']

    success_model = Sequential()
    success_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile success_model
    success_model.compile(loss='mean_squared_error', optimizer='adam')

    success_model.fit(X,success_Y, epochs=30, batch_size=10, verbose=True, shuffle=True)

    success_model.save(success_model_file_path)

    failure_model = Sequential()
    failure_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile failure_model
    failure_model.compile(loss='mean_squared_error', optimizer='adam')

    failure_model.fit(X,failure_Y, epochs=30, batch_size=10, verbose=True, shuffle=True)

    failure_model.save(failure_model_file_path)

    # success_success_model = Sequential()
    # success_success_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    # success_success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    # success_success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # # Compile success_model
    # success_success_model.compile(loss='mean_squared_error', optimizer='adam')

    # success_success_model.fit(X,success_success_Y, epochs=3, batch_size=100, verbose=True, shuffle=True)

    # success_success_model.save(success_success_model_file_path)

    # success_failure_model = Sequential()
    # success_failure_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    # success_failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    # success_failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # # Compile failure_model
    # success_failure_model.compile(loss='mean_squared_error', optimizer='adam')

    # success_failure_model.fit(X,success_failure_Y, epochs=3, batch_size=100, verbose=True, shuffle=True)

    # success_failure_model.save(success_failure_model_file_path)

    # failure_success_model = Sequential()
    # failure_success_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    # failure_success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    # failure_success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # # Compile success_model
    # failure_success_model.compile(loss='mean_squared_error', optimizer='adam')

    # failure_success_model.fit(X,failure_success_Y, epochs=3, batch_size=100, verbose=True, shuffle=True)

    # failure_success_model.save(failure_success_model_file_path)

    # failure_failure_model = Sequential()
    # failure_failure_model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    # failure_failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    # failure_failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # # Compile failure_model
    # failure_failure_model.compile(loss='mean_squared_error', optimizer='adam')

    # failure_failure_model.fit(X,failure_failure_Y, epochs=3, batch_size=100, verbose=True, shuffle=True)

    # failure_failure_model.save(failure_failure_model_file_path)