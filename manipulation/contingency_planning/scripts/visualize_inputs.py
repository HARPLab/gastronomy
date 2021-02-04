from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('data/tongs_overhead_inputs.npy')

fig = plt.figure()
ax = plt.axes(projection='3d')
p = ax.scatter(data[:500,0], data[:500,1])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.show()
