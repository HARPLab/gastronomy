import glob
import subprocess

# for i in range(0,17):
# 	command = "python scripts/pick_up_block_with_tongs_overhead.py -b " + str(i) + " -fc multi_gpu_cfg/franka_tongs_2.yaml"
# 	subprocess.run(command, shell=True)
	# command = "python scripts/pick_up_block_with_spatula_tilted.py -b " + str(i) + " -fc multi_gpu_cfg/franka_spatula_2.yaml"
	# subprocess.run(command, shell=True)

file_paths = glob.glob('baseline/pick_up/tongs_overhead/block*_pick_up_block_with_tongs_overhead.npz')

for file_path in file_paths:
	print(file_path)

	command = "python scripts/process_successful_lifts.py -f " + file_path 
	subprocess.run(command, shell=True)