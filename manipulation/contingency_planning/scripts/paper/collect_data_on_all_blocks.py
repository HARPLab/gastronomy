import subprocess

suffix = ' -bcd cfg/paper/baseline_same_mass/ -bud assets/baseline_same_mass/ -dd baseline_same_mass/pick_up/'

for i in range(0,33):
	command = "python scripts/paper/pick_up/pick_up_block_with_franka_fingers_only.py -b " + str(i) + suffix + 'franka_fingers/'
	subprocess.run(command, shell=True)
	command = "python scripts/paper/pick_up/pick_up_block_with_tongs_overhead_only.py -b " + str(i) + suffix + 'tongs_overhead/'
	subprocess.run(command, shell=True)
	command = "python scripts/paper/pick_up/pick_up_block_with_tongs_side_only.py -b " + str(i) + suffix + 'tongs_side/'
	subprocess.run(command, shell=True)
	command = "python scripts/paper/pick_up/pick_up_block_with_spatula_tilted_with_flip.py -b " + str(i) + suffix + 'spatula_tilted/'
	subprocess.run(command, shell=True)