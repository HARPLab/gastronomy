import glob
import subprocess
import random

urdf_file_paths = glob.glob('../dexnet_meshes/*/*.urdf')

for urdf_file_path in urdf_file_paths:
	urdf_file_path = urdf_file_path[17:]
	randint = random.randint(0,1000)
	urdf_file_path = urdf_file_paths[randint][17:]
	print(urdf_file_path)
	urdf_name = urdf_file_path[urdf_file_path.rfind('/')+1:-9] + '_successful_lift_data.npz'
	print(urdf_name)
	command = "python scripts/paper/pick_up/pick_up_urdf_with_tongs_overhead.py -uf " + urdf_file_path #+ ' -i urdf_data/tongs_overhead/'+urdf_name 
	subprocess.run(command, shell=True)