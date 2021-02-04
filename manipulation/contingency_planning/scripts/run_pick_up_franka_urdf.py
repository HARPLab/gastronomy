import argparse
import subprocess
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', '-g', type=int, default=0)
    parser.add_argument('--urdf_file_num', '-u', type=int, default=0)
    args = parser.parse_args()

    skill_types = ['franka_fingers', 'tongs_overhead', 'tongs_side', 'spatula_flat', 'spatula_tilted']

    file = open('multi_gpu_cfg/urdf_files_'+str(args.urdf_file_num)+'.txt')
    file_contents = file.read()
    urdf_file_paths = file_contents.splitlines()
    file.close()

    existing_skill_type_files = {}
    for skill_type in skill_types:
        existing_skill_type_files[skill_type] = glob.glob('urdf_data/pick_up/'+skill_type+'/*.npz')

    for urdf_file_path in urdf_file_paths:
        for skill_type in skill_types:
            if skill_type == 'franka_fingers':
                cfg_file = 'franka_fingers'
            elif 'tongs' in skill_type:
                cfg_file = 'franka_tongs'
            elif 'spatula' in skill_type:
                cfg_file = 'franka_spatula'
            already_collected = False
            urdf_npz_file_name = urdf_file_path[urdf_file_path.rfind('/')+1:-5]+'_'+skill_type+'.npz'
            for existing_skill_file in existing_skill_type_files[skill_type]:
                if urdf_npz_file_name in existing_skill_file:
                    print(urdf_npz_file_name)
                    already_collected = True
                    break
            if not already_collected:
                command = "python scripts/pick_up_urdf_with_"+skill_type + ".py -uf " + urdf_file_path + " -fc multi_gpu_cfg/"+cfg_file+"_"+str(args.gpu_num)+".yaml"
                subprocess.run(command, shell=True)