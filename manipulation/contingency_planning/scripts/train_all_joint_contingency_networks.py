import glob
import subprocess

file_paths = glob.glob('same_blocks/contingency_data/*contingency_data.npz')

for file_path in file_paths:
	print(file_path)
	command = "python scripts/fit_gaussian.py -f " + file_path
	subprocess.run(command, shell=True)