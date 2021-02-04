# playing_with_food
This repository was used to collect [this dataset](https://sites.google.com/view/playing-with-food/home) involving playing and cutting food items.

## Instructions

1. Start a rosmaster in a terminal.
`
roscore
`
2. If you are using the fingervision, run the following command and look for the USB 2.0 Camera ports.
`
v4l2-ctl --list-devices
`
3. Change the finger_vision.launch file to match the first FingerVision /dev/video port of each USB 2.0 Camera.
4. Start the fingervision by running the following command from the playing_with_food directory.
`
cdplay
./bash_scripts/start_finger_vision.sh
`
5. Open the sound settings in Ubuntu and check that the default input is Multichannel Input - UMC something.
6. Start the sound publisher using the following command from the playing_with_food_directory depending on which setup you are using.
`
cdplay
sfranka
./bash_scripts/start_sound_publishing_cutting.sh
./bash_scripts/start_sound_publishing_playing.sh
`
7. Start the Microsoft Azure Kinect Publisher using the following command.
`
cdplay
./bash_scripts/start_azure_kinect_ros_driver.sh
`
8. Start the Realsense cameras using the following command from the playing_with_food_directory depending on which setup you are using.
`
cdplay
./bash_scripts/start_realsense_cutting.sh
./bash_scripts/start_realsense_playing.sh
`
9. Start the franka-interface on the desired robot with the following commands.
`
cdfranka
start iam-space
start iam-mind
`
10. Run the scripts from the playing_with_food directory.
`
python ./data_collection_scripts/cutting_scripts/cut_vegetable_data_collection.py -w ./dmp_wts/normal_cut_position_wts.pkl -f potato -s 1 -t 1 -ht 0.15
`
`
python ./data_collection_scripts/playing_scripts/collect_data.py -f potato -s 1 -t 1
`

NOTE: don't forget to check audio input
take before and after pics of all 10 slices w/ save_images.py

NOTE: FV1 faces tv, FV2 faces us
