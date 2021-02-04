import glob
import subprocess

urdf_file_paths = glob.glob('../dexnet_meshes/*/*.urdf')
files_to_go = []
num_files_to_divide_into = 30

for urdf_file_path in urdf_file_paths:
    urdf_file_path = urdf_file_path[17:]
    files_to_go.append(urdf_file_path)

number_of_files = len(files_to_go)
number_of_files_in_each_file = int(number_of_files/num_files_to_divide_into)
for i in range(num_files_to_divide_into):
    if(i+1 == num_files_to_divide_into):
        files = files_to_go[number_of_files_in_each_file*i:]
    else:
        files = files_to_go[number_of_files_in_each_file*i:number_of_files_in_each_file*(i+1)]

    with open('multi_gpu_cfg/urdf_files_'+str(i)+'.txt', 'w') as f:
        for file_path in files:
            f.write("%s\n" % file_path)

print(len(files_to_go))

    # command = "python scripts/paper/pick_up/pick_up_urdf_with_tongs_overhead.py -uf " + urdf_file_path 
    # subprocess.run(command, shell=True)