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

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            cv2.imwrite(dir_path + str(i) + '.png', frame)
            i += 1

            # # Display the resulting frame
            # cv2.imshow('Frame',frame)

            # # Press Q on keyboard to exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

