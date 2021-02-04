import numpy as np

successful_lift_data = np.load('baseline/pick_up/franka_fingers_data.npy')
print(successful_lift_data.shape)
sums = np.sum(successful_lift_data, axis=0)
print(sums)

good_data = np.transpose(np.transpose(successful_lift_data)[np.nonzero(sums > 5),:]).reshape(-1, np.count_nonzero(sums > 5))
print(good_data.shape)

np.save('baseline/pick_up/good_franka_fingers_data.npy', good_data)