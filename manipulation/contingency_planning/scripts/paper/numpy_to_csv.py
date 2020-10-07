# save numpy array as csv file
import numpy as np
from numpy import asarray
from numpy import savetxt

data = np.load('baseline/pick_up/complete_data.npy')

sorted_x_y_thetas = np.load('baseline/franka_fingers_500.npy')
franka_x_y_thetas = np.hstack((np.zeros((500,1)), sorted_x_y_thetas, np.zeros((500,2))))

sorted_x_y_thetas = np.load('baseline/tongs_overhead_500.npy')
tong_overhead_x_y_thetas = np.hstack((np.ones((500,1)), sorted_x_y_thetas, np.zeros((500,2))))

sorted_x_y_theta_dist_tilts = np.load('baseline/tongs_side_500.npy')
tong_side_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 2, sorted_x_y_theta_dist_tilts))

sorted_x_y_theta_dist_tilts = np.load('baseline/spatula_tilted_500.npy')
spatula_tilted_x_y_theta_dist_tilts = np.hstack((np.ones((500,1)) * 3, sorted_x_y_theta_dist_tilts))

sorted_x_y_thetas = np.vstack((franka_x_y_thetas, tong_overhead_x_y_thetas, tong_side_x_y_theta_dist_tilts, spatula_tilted_x_y_theta_dist_tilts))

# save to csv file
savetxt('data.csv', data, delimiter=',')
savetxt('poses.csv', sorted_x_y_thetas, delimiter=',')