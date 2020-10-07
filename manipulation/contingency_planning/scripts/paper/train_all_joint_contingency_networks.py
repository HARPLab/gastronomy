import glob
import subprocess

file_paths = glob.glob('same_block_data/contingency_data/*pick_up_only_with_flip_200.npz')

for file_path in file_paths:
	print(file_path)
	#command = "mv " + file_path + " " + file_path[:-9] + file_path[-4:]
	command = "python scripts/paper/train_failure_joint_contingency_networks.py -f " + file_path
	subprocess.run(command, shell=True)