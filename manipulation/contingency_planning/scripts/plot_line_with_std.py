import numpy as np
import matplotlib.pyplot as plt

num_blocks = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]

# Accuracy

baseline_acc_mean = np.array([0.607, 0.678, 0.710, 0.787, 0.796, 0.786, 0.755, 0.864, 0.813, 0.854, 0.861, 0.804, 0.818, 0.77])
baseline_acc_std = np.array([0.391, 0.306, 0.315, 0.148, 0.121, 0.157, 0.215, 0.169, 0.251, 0.166, 0.185, 0.234, 0.225, 0.27])

baseline_same_friction_acc_mean = np.array([0.607, 0.731, 0.799, 0.799, 0.799, 0.799, 0.799, 0.776, 0.776, 0.776, 0.75, 0.766, 0.764, 0.766])
baseline_same_friction_acc_std = np.array([0.35, 0.296, 0.236, 0.236, 0.236, 0.236, 0.236, 0.224, 0.224, 0.224, 0.243, 0.265, 0.271, 0.265])

baseline_same_mass_acc_mean = np.array([0.387, 0.574, 0.809, 0.859, 0.779, 0.83, 0.825, 0.825, 0.83, 0.776, 0.738, 0.743, 0.79, 0.787])
baseline_same_mass_acc_std = np.array([0.394, 0.246, 0.231, 0.175, 0.286, 0.207, 0.213, 0.213, 0.207, 0.251, 0.26, 0.265, 0.275, 0.277])

gaussian_acc_mean = np.array([0.625, 0.623, 0.794, 0.825, 0.816, 0.835, 0.836, 0.882, 0.891, 0.894, 0.886, 0.847, 0.901, 0.785])
gaussian_acc_std = np.array([0.217, 0.256, 0.278, 0.211, 0.225, 0.192, 0.173, 0.132, 0.132, 0.136, 0.134, 0.208, 0.118, 0.14])

new_NN_acc_mean = np.array([0.387, 0.646, 0.742, 0.726, 0.761, 0.784, 0.778, 0.781, 0.794, 0.84, 0.859, 0.813, 0.841, 0.865])
new_NN_acc_std = np.array([0.394, 0.276, 0.335, 0.314, 0.311, 0.288, 0.296, 0.276, 0.284, 0.211, 0.194, 0.263, 0.238, 0.201])

old_NN_acc_mean = np.array([0.404, 0.399, 0.837, 0.865, 0.857, 0.845, 0.822, 0.822, 0.868, 0.821, 0.814, 0.856, 0.84, 0.821])
old_NN_acc_std = np.array([0.389, 0.388, 0.211, 0.194, 0.191, 0.207, 0.254, 0.253, 0.17, 0.257, 0.272, 0.200, 0.237, 0.266])

# F1 Score
baseline_f1_mean = np.array([0.666, 0.653, 0.675, 0.637, 0.64, 0.675, 0.619, 0.687, 0.646, 0.719, 0.703, 0.646, 0.681, 0.673])
baseline_f1_std = np.array([0.371, 0.388, 0.384, 0.341, 0.354, 0.313, 0.363, 0.372, 0.387, 0.339, 0.361, 0.398, 0.376, 0.378])

baseline_same_friction_f1_mean = np.array([0.65, 0.534, 0.458, 0.458, 0.458, 0.458, 0.458, 0.445, 0.445, 0.445, 0.502, 0.512, 0.508, 0.512])
baseline_same_friction_f1_std = np.array([0.372, 0.432, 0.473, 0.473, 0.473, 0.473, 0.473, 0.459, 0.459, 0.459, 0.431, 0.444, 0.448, 0.444])

baseline_same_mass_f1_mean = np.array([0.0, 0.327, 0.567, 0.555, 0.628, 0.567, 0.564, 0.564, 0.567, 0.554, 0.661, 0.653, 0.658, 0.661])
baseline_same_mass_f1_std = np.array([0.0 ,0.257, 0.436, 0.448, 0.386, 0.388, 0.388, 0.388, 0.388 ,0.4, 0.389, 0.399, 0.389, 0.386])

gaussian_f1_mean = np.array([0.392, 0.462, 0.647, 0.546, 0.541, 0.560, 0.565, 0.718, 0.716, 0.646, 0.642, 0.709, 0.741, 0.671])
gaussian_f1_std = np.array([0.319, 0.371, 0.399, 0.438, 0.439, 0.433, 0.428, 0.35, 0.356, 0.423, 0.421, 0.358, 0.344, 0.313])

new_NN_f1_mean = np.array([0.0, 0.402, 0.548, 0.534, 0.576, 0.621, 0.639, 0.648, 0.569, 0.566, 0.607, 0.676, 0.679, 0.657])
new_NN_f1_std = np.array([0.0, 0.391, 0.441, 0.429, 0.433, 0.414, 0.397, 0.404, 0.451, 0.445, 0.434, 0.376, 0.390, 0.414])

old_NN_f1_mean = np.array([0.074, 0.043, 0.564, 0.623, 0.607, 0.565, 0.536, 0.687, 0.630, 0.683, 0.679, 0.603, 0.688, 0.659])
old_NN_f1_std = np.array([0.096, 0.073, 0.444, 0.433, 0.44, 0.451, 0.456, 0.376, 0.429, 0.373, 0.376, 0.437, 0.389, 0.395])

fig = plt.figure(1)
ax = plt.axes()
plt.plot(num_blocks, baseline_acc_mean, color='b', label='baseline')
plt.fill_between(num_blocks,baseline_acc_mean-baseline_acc_std,baseline_acc_mean+baseline_acc_std,color='b',alpha=.1)

plt.plot(num_blocks, baseline_same_friction_acc_mean, color='g', label='baseline_same_friction')
plt.fill_between(num_blocks,baseline_same_friction_acc_mean-baseline_same_friction_acc_std,baseline_same_friction_acc_mean+baseline_same_friction_acc_std,color='g',alpha=.1)

plt.plot(num_blocks, baseline_same_mass_acc_mean, color='r', label='baseline_same_mass')
plt.fill_between(num_blocks,baseline_same_mass_acc_mean-baseline_same_mass_acc_std,baseline_same_mass_acc_mean+baseline_same_mass_acc_std,color='r',alpha=.1)

plt.plot(num_blocks, gaussian_acc_mean, color='c', label='gaussian')
plt.fill_between(num_blocks,gaussian_acc_mean-gaussian_acc_std,gaussian_acc_mean+gaussian_acc_std,color='c',alpha=.1)

plt.plot(num_blocks, new_NN_acc_mean, color='m', label='new_NN')
plt.fill_between(num_blocks,new_NN_acc_mean-new_NN_acc_std,new_NN_acc_mean+new_NN_acc_std,color='m',alpha=.1)

plt.plot(num_blocks, old_NN_acc_mean, color='orange', label='old_NN')
plt.fill_between(num_blocks,old_NN_acc_mean-old_NN_acc_std,old_NN_acc_mean+old_NN_acc_std,color='orange',alpha=.1)

ax.legend(loc=4)
ax.set_xlabel('Num Blocks')
ax.set_ylabel('Accuracy')
ax.set_title('Num Blocks vs Accuracy for 50 skills')
plt.show()

fig = plt.figure(2)
ax = plt.axes()
plt.plot(num_blocks, baseline_f1_mean, color='b', label='baseline')
plt.fill_between(num_blocks,baseline_f1_mean-baseline_f1_std,baseline_f1_mean+baseline_f1_std,color='b',alpha=.1)

plt.plot(num_blocks, baseline_same_friction_f1_mean, color='g', label='baseline_same_friction')
plt.fill_between(num_blocks,baseline_same_friction_f1_mean-baseline_same_friction_f1_std,baseline_same_friction_f1_mean+baseline_same_friction_f1_std,color='g',alpha=.1)

plt.plot(num_blocks, baseline_same_mass_f1_mean, color='r', label='baseline_same_mass')
plt.fill_between(num_blocks,baseline_same_mass_f1_mean-baseline_same_mass_f1_std,baseline_same_mass_f1_mean+baseline_same_mass_f1_std,color='r',alpha=.1)

plt.plot(num_blocks, gaussian_f1_mean, color='c', label='gaussian')
plt.fill_between(num_blocks,gaussian_f1_mean-gaussian_f1_std,gaussian_f1_mean+gaussian_f1_std,color='c',alpha=.1)

plt.plot(num_blocks, new_NN_f1_mean, color='m', label='new_NN')
plt.fill_between(num_blocks,new_NN_f1_mean-new_NN_f1_std,new_NN_f1_mean+new_NN_f1_std,color='m',alpha=.1)

plt.plot(num_blocks, old_NN_f1_mean, color='orange', label='old_NN')
plt.fill_between(num_blocks,old_NN_f1_mean-old_NN_f1_std,old_NN_f1_mean+old_NN_f1_std,color='orange',alpha=.1)

ax.legend(loc=4)
ax.set_xlabel('Num Blocks')
ax.set_ylabel('F1 score')
ax.set_title('Num Blocks vs F1 Score for 50 skills')
plt.show()