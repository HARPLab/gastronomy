import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str)
    parser.add_argument('--video_path', '-v', type=str)
    args = parser.parse_args()


def update_line(num, data, line):
    line.set_data(data[:,:num])
    return line,

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)


fig1 = plt.figure()

data = np.load(args.filename)
force_data = data['push_down_robot_forces']
z_force_data = force_data[:,2].reshape(1,-1)
enumerated_data = np.arange(z_force_data.shape[1]).reshape(1,-1) * 0.01
combined_data = np.vstack((enumerated_data,z_force_data))

print(z_force_data.shape[1])

l, = plt.plot([], [], 'r-')
plt.xlim(0, z_force_data.shape[1] * 0.01)
plt.ylim(-15, 5)
plt.xlabel('Time(s)')
plt.ylabel('Force (N)')
plt.title('Push Z Forces')
line_ani = animation.FuncAnimation(fig1, update_line, z_force_data.shape[1], fargs=(combined_data, l),
                                   interval=10, blit=True)
line_ani.save(args.video_path, writer=writer)