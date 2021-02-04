import subprocess

for i in range(0,100):
	command = "python scripts/create_random_block.py -b " + str(i) 
	subprocess.run(command, shell=True)
