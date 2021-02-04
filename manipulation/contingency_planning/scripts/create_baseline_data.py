import glob
import numpy as np

suffices = ['franka_fingers'] #, 'tongs_overhead', 'tongs_side'] #, 'spatula_flat', 'spatula_tilted']

successful_lift_data = {}

for suffix in suffices:
    file_paths = glob.glob('baseline_same_friction/pick_up/'+suffix+'/*.npy')
    successful_lift_data[suffix] = np.zeros((500,100))

    for i in range(0,101):
        for file_path in file_paths:
            if str(i) in file_path:
                file_data = np.load(file_path)
                successful_lift_data[suffix][:,i] = file_data

# complete_data = np.vstack((successful_lift_data['franka_fingers'],
#                            successful_lift_data['tongs_overhead'],
#                            successful_lift_data['tongs_side'],
#                            successful_lift_data['spatula_flat'],
#                            successful_lift_data['spatula_tilted']))

# complete_data = np.vstack((successful_lift_data['franka_fingers'],
#                            successful_lift_data['tongs_overhead'],
#                            successful_lift_data['tongs_side']))

#np.save('baseline_same_friction/pick_up/complete_data.npy', complete_data)
#np.save('baseline_same_friction/pick_up/grasp_data.npy', complete_data)
np.save('baseline_same_friction/pick_up/franka_fingers_data.npy', successful_lift_data['franka_fingers'])