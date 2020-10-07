import subprocess

for i in range(0,100):
	command = "python scripts/paper/create_same_random_block_urdf.py -b " + str(i) 
	subprocess.run(command, shell=True)
