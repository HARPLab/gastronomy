import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', '-s', type=str)
    args = parser.parse_args()
    
    input_file_path = 'same_block_data/' + args.suffix + '_pick_up_only_data.npz'

    data = np.load(input_file_path)
    X = data['X']
    Y = data['y']
    input_dim = X.shape[1]
    data_array = np.hstack((X,Y))
    np.random.shuffle(data_array)
    trainX = data_array[:-1000,:-1]
    trainY = data_array[:-1000,-1]
    sample_weights = np.ones(trainY.shape[0]) 
    trainY_greater_than_0 = trainY > 0
    sample_weights[trainY_greater_than_0] = 100
    testX = data_array[-1000:,:-1]
    testY = data_array[-1000:,-1]

    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(trainX,trainY, epochs=100, batch_size=100, verbose=True, shuffle=True, sample_weight=sample_weights)
    predY = model.predict(testX)
    error = np.square(predY - testY.reshape(-1,1))
    print(np.mean(error,0))

    output_file_path = 'same_block_data/' + args.suffix + '_pick_up_only_trained_model.h5'

    model.save(output_file_path)