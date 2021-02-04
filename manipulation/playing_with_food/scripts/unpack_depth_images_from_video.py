import cv2
import numpy as np
import argparse
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file_path', '-v', type=str)
    parser.add_argument('--data_dir', '-d', type=str)
    args = parser.parse_args()

    dir_path = args.data_dir
    createFolder(dir_path)

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(args.video_file_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video file")

    i = 0

    boundaries = np.load(args.video_file_path[:-4] + '_image_boundaries.npy')

    #depth_images = np.zeros((0,480,640), dtype=np.uint16)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            new_depth_image = np.zeros((480,640))

            for j in range(3):
                frame_color = frame[:,:,j]
                nonzero_indices = np.nonzero(frame_color)
                alpha = (boundaries[i,j,1] - boundaries[i,j,0]) / 255.0
                new_depth_image[nonzero_indices] = (frame_color[nonzero_indices] * alpha) + boundaries[i,j,0]

            depth_image = new_depth_image.astype(np.uint16)

            cv2.imwrite(dir_path + str(i) + '.png', depth_image)
            #depth_images = np.vstack((depth_images, depth_image.reshape(1,480,640)))
            i += 1

            # Display the resulting frame
            cv2.imshow('Frame',depth_image)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    #np.save(dir_path+'_depth_images.npy', depth_images)

