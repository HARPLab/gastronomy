import glob
import subprocess

#file_paths = glob.glob('baseline_same_friction/pick_up/franka_fingers/*.npz')
#file_paths = glob.glob('same_block_data/contingency_data/*.npz')

file_paths = glob.glob('same_block_data/pick_up_only/*.npz')

for file_path in file_paths:
	print(file_path)
	#command = "mv " + file_path + " " + file_path[:-9] + file_path[-4:]
	#command = "python scripts/paper/process_successful_lift.py -f " + file_path 
	#command = "python scripts/paper/train_joint_contingency_networks.py -f " + file_path
	command = "python scripts/paper/process_change_in_block_pose.py -f " + file_path 
	subprocess.run(command, shell=True)