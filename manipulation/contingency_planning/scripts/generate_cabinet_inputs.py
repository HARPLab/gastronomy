import argparse
import numpy as np

if __name__ == "__main__":

    inputs = []

    for x in np.linspace(-0.2,0,5):
        for y in np.linspace(-0.2,0.2,10):
            for z in np.linspace(-0.2,0.2,10):
                inputs.append([x,y,z])

    input_np_array = np.array(inputs)
    np.save('cabinets/cabinet_inputs.npy', input_np_array)
