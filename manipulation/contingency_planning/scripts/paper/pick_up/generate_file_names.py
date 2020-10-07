import glob
import subprocess

urdf_file_paths = glob.glob('../dexnet_meshes/*/*.urdf')
#data_file_paths = glob.glob('urdf_data/franka_fingers/*.npz')
data_file_paths = []
files_to_go = []

for urdf_file_path in urdf_file_paths:
    urdf_file_path = urdf_file_path[17:]
    #print(urdf_file_path)
    urdf_name = urdf_file_path[urdf_file_path.find('/')+1:-5]

    collected_data = False

    for names in data_file_paths:
        if urdf_name in names:
            collected_data = True
            break

    if not collected_data:
        files_to_go.append(urdf_file_path)

number_of_files = len(files_to_go)
number_of_files_divided_by_4 = int(number_of_files/4)
files1 = files_to_go[:number_of_files_divided_by_4]
files2 = files_to_go[number_of_files_divided_by_4:number_of_files_divided_by_4*2]
files3 = files_to_go[number_of_files_divided_by_4*2:number_of_files_divided_by_4*3]
files4 = files_to_go[number_of_files_divided_by_4*3:]

with open('isaac_cfg/franka_tongs_0.txt', 'w') as f:
    for file_path in files1:
        f.write("%s\n" % file_path)
with open('isaac_cfg/franka_tongs_1.txt', 'w') as f:
    for file_path in files2:
        f.write("%s\n" % file_path)
with open('isaac_cfg/franka_tongs_2.txt', 'w') as f:
    for file_path in files3:
        f.write("%s\n" % file_path)
with open('isaac_cfg/franka_tongs_3.txt', 'w') as f:
    for file_path in files4:
        f.write("%s\n" % file_path)

print(len(files_to_go))

    # command = "python scripts/paper/pick_up/pick_up_urdf_with_tongs_overhead.py -uf " + urdf_file_path 
    # subprocess.run(command, shell=True)