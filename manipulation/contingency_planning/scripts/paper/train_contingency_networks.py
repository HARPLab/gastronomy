import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='same_block_data/tongs_overhead_contingency_data_pick_up_only.npz')
    args = parser.parse_args()
    
    data = np.load(args.file_path)
    X = data['X']
    success_Y = data['success_Y']
    failure_Y = data['failure_Y']

    success_model = Sequential()
    success_model.add(Dense(100, input_dim=3, kernel_initializer='normal', activation='relu'))
    success_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    success_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile success_model
    success_model.compile(loss='mean_squared_error', optimizer='adam')

    success_model.fit(X,success_Y, epochs=10, batch_size=100, verbose=True, shuffle=True)

    success_model.save('same_block_data/tongs_overhead_pick_up_only_success_contingency_model.h5')

    failure_model = Sequential()
    failure_model.add(Dense(100, input_dim=3, kernel_initializer='normal', activation='relu'))
    failure_model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    failure_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile failure_model
    failure_model.compile(loss='mean_squared_error', optimizer='adam')

    failure_model.fit(X,failure_Y, epochs=10, batch_size=100, verbose=True, shuffle=True)

    failure_model.save('same_block_data/tongs_overhead_pick_up_only_failure_contingency_model.h5')