import numpy as np
import os
import shutil


def main(path_list, dest_path):
    img_count = 1
    main_dest_dir = os.path.join(dest_path, 'main')
    secondary_dest_dir = os.path.join(dest_path, 'secondary')

    for p in [main_dest_dir, secondary_dest_dir]:
        if not os.path.exists(p):
            os.makedirs(p)

    for p in path_list:
        main_img_dir = os.path.join(p, 'main')
        secondary_img_dir = os.path.join(p, 'secondary')

        for img_idx in range(1, 1000):
            img_path = '{}_main_img.png'.format(img_idx)
            if not os.path.exists(os.path.join(main_img_dir, img_path)):
                break

            if 'png' not in img_path:
                continue
            main_img_path = os.path.join(main_img_dir, img_path)
            new_main_img_path = os.path.join(
                main_dest_dir, '{}_main_img.png'.format(img_count))
            shutil.copy2(main_img_path, new_main_img_path)


            curr_img_count = int(img_path.split('_')[0])
            secondary_img_path = os.path.join(
                secondary_img_dir, '{}_secondary_img.png'.format(curr_img_count))
            assert os.path.exists(secondary_img_path)
            new_secondary_img_path = os.path.join(
                secondary_dest_dir, '{}_secondary_img.png'.format(img_count)
            )
            shutil.copy2(secondary_img_path, new_secondary_img_path)
            img_count = img_count + 1
        
        print("Copied images \t     from dir:   {}\n"
              "              \t       to dir:   {}\n"
              "              \t        count:   {}".format(
                  p, dest_path, img_idx
              ))


if __name__ == '__main__':
    path_list = [
        '/home/klz/good_calib_data/main_overhead_sec_front_left/Nov_23_try_1/calib_data_1',
        '/home/klz/good_calib_data/main_overhead_sec_front_left/Nov_23_try_2',

        # '/home/klz/good_calib_data/main_front_right_sec_overhead/combined_data/Nov_20_7_30/org/calib_data_Nov_19_11_20_PM',
        # '/home/klz/good_calib_data/main_front_right_sec_overhead/combined_data/Nov_20_7_30/org/calib_data_Nov_21_12_30_PM_try_5'    
    ]
    # dest_path = '/home/klz/good_calib_data/main_front_right_sec_overhead/combined_data/Nov_20_7_30/combined'
    dest_path = '/home/klz/good_calib_data/main_overhead_sec_front_left/combined/Nov_23_try_1_2/'

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    main(path_list, dest_path)