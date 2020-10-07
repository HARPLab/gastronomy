import argparse
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, actions, images, Y, batch_size) :
    self.actions = actions
    self.images = images
    self.Y = Y
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.Y) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_actions = self.actions[idx * self.batch_size : (idx+1) * self.batch_size]
    current_batch_image = self.images[idx].reshape(1,100,100,1)
    batch_images = np.repeat(current_batch_image, 100, axis=0)

    batch_y = self.Y[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return [batch_actions, batch_images], batch_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', type=str, default='urdf_data/franka_fingers/franka_fingers_convolution_data.npz')
    args = parser.parse_args()
    
    data = np.load(args.file_path)

    batch_size = 100

    actions = data['actions']
    images = data['images']
    Y = data['Y']

    actions_train = actions[:690000,:]
    actions_test = actions[690000:,:]
    images_train = images[:6900,:]
    images_test = images[6900:,:]
    Y_train = Y[:690000,:]
    Y_test = Y[690000:,:]

    my_training_batch_generator = My_Custom_Generator(actions_train, images_train, Y_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(actions_test, images_test, Y_test, batch_size)
    
    action_input = Input(shape=(3, ))
    depth_input = Input(shape=(100,100,1))
    conv1 = Conv2D(64, (3, 3), activation='relu')(depth_input)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    flatten = Flatten()(conv2)
    concatenated = concatenate([action_input, flatten])
    dense1 = Dense(100, kernel_initializer='normal', activation='relu')(concatenated)
    dense2 = Dense(100, kernel_initializer='normal', activation='relu')(dense1)
    out = Dense(1, kernel_initializer='normal', activation='sigmoid')(dense2)
    # Compile model
    model = Model([action_input,depth_input], out)
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator, epochs=10, verbose=True, shuffle=True)

    model.save('franka_fingers_convolution_model.h5')
