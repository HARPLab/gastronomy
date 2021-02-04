import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_urdf_template_filename', '-uf', type=str, default='assets/block_template.urdf')
    parser.add_argument('--block_yaml_template_filename', '-yf', type=str, default='cfg/block_template.yaml')
    parser.add_argument('--urdf_dir_path', '-ud', type=str, default='assets/baseline/')
    parser.add_argument('--yaml_dir_path', '-yd', type=str, default='cfg/paper/baseline/')
    parser.add_argument('--block_num', '-b', type=int)
    
    args = parser.parse_args()

    f = open(args.block_urdf_template_filename, "r")
    filedata = f.read()
    f.close()

    random_dim_number = np.random.random()
    random_mass_number = np.random.random()
    random_friction_number = np.random.random()

    block1_width = 0.05
    block1_height = 0.05
    block2_width = block1_width
    block2_height = block1_height
    block3_width = block1_width
    block3_height = block1_height

    if(random_dim_number < 0.25):
        block1_length = np.random.random() * 0.075
        block2_length = block1_length
        block3_length = 0.15 - 2 * block1_length
    elif (random_dim_number < 0.5):
        block1_length = np.random.random() * 0.075
        block3_length = block1_length
        block2_length = 0.15 - 2 * block1_length
    elif (random_dim_number < 0.75):
        block2_length = np.random.random() * 0.075
        block3_length = block2_length
        block1_length = 0.15 - 2 * block2_length
    else:
        block1_length = np.random.random() * 0.15
        block2_length = np.random.random() * (0.15 - block1_length)
        block3_length = 0.15 - block1_length - block2_length
    
    if(random_mass_number < 0.2):
        block1_mass = np.random.random()
        block2_mass = block1_mass
        block3_mass = block1_mass
    elif (random_mass_number < 0.4):
        block1_mass = np.random.random()
        block2_mass = block1_mass
        block3_mass = np.random.random() * (3 - block1_mass - block2_mass)
    elif (random_mass_number < 0.6):
        block1_mass = np.random.random()
        block3_mass = block1_mass
        block2_mass = np.random.random() * (3 - block1_mass - block3_mass)
    elif (random_mass_number < 0.8):
        block2_mass = np.random.random()
        block3_mass = block2_mass
        block1_mass = np.random.random() * (3 - block2_mass - block3_mass)
    else:
        block1_mass = np.random.random() * 3.0
        block2_mass = np.random.random() * (3 - block1_mass)
        block3_mass = np.random.random() * (3 - block1_mass - block2_mass)

    if(random_friction_number < 0.2):
        block1_friction = np.random.random()
        block1_rolling_friction = np.random.random()
        block1_torsion_friction = np.random.random()
        block2_friction = block1_friction
        block2_rolling_friction = block1_rolling_friction
        block2_torsion_friction = block1_torsion_friction
        block3_friction = block1_friction
        block3_rolling_friction = block1_rolling_friction
        block3_torsion_friction = block1_torsion_friction
    elif (random_friction_number < 0.4):
        block1_friction = np.random.random()
        block1_rolling_friction = np.random.random()
        block1_torsion_friction = np.random.random()
        block2_friction = block1_friction
        block2_rolling_friction = block1_rolling_friction
        block2_torsion_friction = block1_torsion_friction
        block3_friction = np.random.random()
        block3_rolling_friction = np.random.random()
        block3_torsion_friction = np.random.random()
    elif (random_friction_number < 0.6):
        block1_friction = np.random.random()
        block1_rolling_friction = np.random.random()
        block1_torsion_friction = np.random.random()
        block2_friction = np.random.random()
        block2_rolling_friction = np.random.random()
        block2_torsion_friction = np.random.random()
        block3_friction = block1_friction
        block3_rolling_friction = block1_rolling_friction
        block3_torsion_friction = block1_torsion_friction
    elif (random_friction_number < 0.8):
        block1_friction = np.random.random()
        block1_rolling_friction = np.random.random()
        block1_torsion_friction = np.random.random()
        block2_friction = np.random.random()
        block2_rolling_friction = np.random.random()
        block2_torsion_friction = np.random.random()
        block3_friction = block2_friction
        block3_rolling_friction = block2_rolling_friction
        block3_torsion_friction = block2_torsion_friction
    else:
        block1_friction = np.random.random()
        block1_rolling_friction = np.random.random()
        block1_torsion_friction = np.random.random()
        block2_friction = np.random.random()
        block2_rolling_friction = np.random.random()
        block2_torsion_friction = np.random.random()
        block3_friction = np.random.random()
        block3_rolling_friction = np.random.random()
        block3_torsion_friction = np.random.random()

    new_block1_length = filedata.replace("{block1_length}", str(block1_length))
    new_block1_width = new_block1_length.replace("{block1_width}", str(block1_width))
    new_block1_height = new_block1_width.replace("{block1_height}", str(block1_height))
    new_block1_mass = new_block1_height.replace("{block1_mass}", str(block1_mass))
    new_block1_ixx = new_block1_mass.replace("{block1_ixx}", str(block1_mass*(block1_height*block1_height+block1_width*block1_width)/12))
    new_block1_iyy = new_block1_ixx.replace("{block1_iyy}", str(block1_mass*(block1_length*block1_length+block1_width*block1_width)/12))
    new_block1_izz = new_block1_iyy.replace("{block1_izz}", str(block1_mass*(block1_length*block1_length+block1_height*block1_height)/12))

    new_block2_length = new_block1_izz.replace("{block2_length}", str(block2_length))
    new_block2_width = new_block2_length.replace("{block2_width}", str(block2_width))
    new_block2_height = new_block2_width.replace("{block2_height}", str(block2_height))
    new_block2_mass = new_block2_height.replace("{block2_mass}", str(block2_mass))
    new_block2_ixx = new_block2_mass.replace("{block2_ixx}", str(block2_mass*(block2_height*block2_height+block2_width*block2_width)/12))
    new_block2_iyy = new_block2_ixx.replace("{block2_iyy}", str(block2_mass*(block2_length*block2_length+block2_width*block2_width)/12))
    new_block2_izz = new_block2_iyy.replace("{block2_izz}", str(block2_mass*(block2_length*block2_length+block2_height*block2_height)/12))

    new_block3_length = new_block2_izz.replace("{block3_length}", str(block3_length))
    new_block3_width = new_block3_length.replace("{block3_width}", str(block3_width))
    new_block3_height = new_block3_width.replace("{block3_height}", str(block3_height))
    new_block3_mass = new_block3_height.replace("{block3_mass}", str(block3_mass))
    new_block3_ixx = new_block3_mass.replace("{block3_ixx}", str(block3_mass*(block3_height*block3_height+block3_width*block3_width)/12))
    new_block3_iyy = new_block3_ixx.replace("{block3_iyy}", str(block3_mass*(block3_length*block3_length+block3_width*block3_width)/12))
    new_block3_izz = new_block3_iyy.replace("{block3_izz}", str(block3_mass*(block3_length*block3_length+block3_height*block3_height)/12))

    new_link1_x = new_block3_izz.replace("{link1_x}", str((block1_length+block2_length)/2))
    new_link2_x = new_link1_x.replace("{link2_x}", str((block2_length+block3_length)/2))

    f = open(args.urdf_dir_path+'block'+str(args.block_num)+'.urdf', "w")
    f.write(new_link2_x)
    f.close()

    f = open(args.block_yaml_template_filename, "r")
    filedata = f.read()
    f.close()

    new_block_urdf_path = filedata.replace("{block_urdf_path}", 'block'+str(args.block_num)+'.urdf')

    new_block1_friction = new_block_urdf_path.replace("{block1_friction}", str(block1_friction))
    new_block1_rolling_friction = new_block1_friction.replace("{block1_rolling_friction}", str(block1_rolling_friction))
    new_block1_torsion_friction = new_block1_rolling_friction.replace("{block1_torsion_friction}", str(block1_torsion_friction))

    new_block2_friction = new_block1_torsion_friction.replace("{block2_friction}", str(block2_friction))
    new_block2_rolling_friction = new_block2_friction.replace("{block2_rolling_friction}", str(block2_rolling_friction))
    new_block2_torsion_friction = new_block2_rolling_friction.replace("{block2_torsion_friction}", str(block2_torsion_friction))

    new_block3_friction = new_block2_torsion_friction.replace("{block3_friction}", str(block3_friction))
    new_block3_rolling_friction = new_block3_friction.replace("{block3_rolling_friction}", str(block3_rolling_friction))
    new_block3_torsion_friction = new_block3_rolling_friction.replace("{block3_torsion_friction}", str(block3_torsion_friction))

    f = open(args.yaml_dir_path+'block'+str(args.block_num)+'.yaml', "w")
    f.write(new_block3_torsion_friction)
    f.close()