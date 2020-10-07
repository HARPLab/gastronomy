import argparse

from autolab_core import YamlConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_filename', '-tf', type=str, default='assets/block_template.urdf')
    parser.add_argument('--filename', '-f', type=str, default='assets/block.urdf')
    parser.add_argument('--cfg', '-c', type=str, default='cfg/block.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    f = open(args.template_filename, "r")
    filedata = f.read()
    f.close()

    block1_length = cfg['block']['block1']['length']
    block1_width = cfg['block']['block1']['width']
    block1_height = cfg['block']['block1']['height']
    block1_mass = cfg['block']['block1']['mass']

    block2_length = cfg['block']['block2']['length']
    block2_width = cfg['block']['block2']['width']
    block2_height = cfg['block']['block2']['height']
    block2_mass = cfg['block']['block2']['mass']

    block3_length = cfg['block']['block3']['length']
    block3_width = cfg['block']['block3']['width']
    block3_height = cfg['block']['block3']['height']
    block3_mass = cfg['block']['block3']['mass']

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

    f = open(args.filename, "w")
    f.write(new_link2_x)
    f.close()