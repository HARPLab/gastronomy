# import the necessary packages
import numpy as np
import cv2
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='urdf_data/franka_fingers/')
    parser.add_argument('--image_width', '-iw', type=int, default=100)
    parser.add_argument('--image_height', '-ih', type=int, default=100)
    args = parser.parse_args()

    seg_image_file_paths = glob.glob(args.data_dir + '*_seg.png')
    print(len(seg_image_file_paths))

    i = 0
    for seg_image_file in seg_image_file_paths:
        # print(seg_image_file)
        i += 1
        print(i)
        # load the image and show it
        image = cv2.imread(seg_image_file)
        object_image = image[:,:,1] > 200
        object_mask = np.nonzero(object_image)
        if len(object_mask) == 2:
            min_y = np.min(object_mask[0])
            max_y = np.max(object_mask[0])
            min_x = np.min(object_mask[1])
            max_x = np.max(object_mask[1])
            # print(object_mask)
            # cv2.imshow("original", image)
            # cv2.waitKey(0)

            image_width = max_x - min_x
            image_height = max_y - min_y
            if image_width <= 100 and image_height <= 100:
                image_center_x = int((max_x + min_x) / 2)
                image_center_y = int((max_y + min_y) / 2)
                depth_image_file = seg_image_file[:-7] + 'depth.png'
                depth_image = cv2.imread(depth_image_file)
                cropped_depth_image = depth_image[(image_center_y-50):(image_center_y+50), (image_center_x-50):(image_center_x+50),:]
                cropped_depth_image_file = seg_image_file[:-7] + 'cropped.png'
                cv2.imwrite(cropped_depth_image_file, cropped_depth_image)
                # cv2.imshow("cropped", cropped_depth_image)
                # cv2.waitKey(0)
    # print(image_widths)
    # print(image_heights)

    # print(np.max(np.array(image_widths)))
    # print(np.max(np.array(image_heights)))

    # image[:,:,0] = image[:,:,1]
    # image[:,:,2] = image[:,:,1]
    # image *= 255
    # cv2.imshow("original", image)
    # cv2.waitKey(0)

    # print(image)

    # cropped = image[70:170, 440:540]
    # cv2.imshow("cropped", cropped)
    # cv2.waitKey(0)