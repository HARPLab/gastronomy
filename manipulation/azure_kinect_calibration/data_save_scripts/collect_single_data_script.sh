#!/bin/bash

# Add path to the directory to save it.
data_save_dir='/home/klz/datasets/data_in_line/try_2/false/objects_5/'
# raw_data_save_dir='/home/klz/datasets/data_in_line/try_2/true/objects_3/try_5_Dec_01_2019_11_12_PM'
# raw_data_save_dir='/home/klz/datasets/data_in_line/try_2/true/objects_4/try_5_Dec_10_2019_01_28_AM'

python ./k4a_data_save_open3d.py --save_dir ${data_save_dir}
wait
sleep 2
echo "Collected raw data."

raw_data_save_dir=$(ls -td ${data_save_dir}/*/ | head -1)

h5_path="${raw_data_save_dir}/input_data.h5"
python ./create_pcd_from_saved_data.py --h5_path ${h5_path} --output_dir_prefix extracted_pcd_data

wait
sleep 2
echo "Did create initially segmented pcd data."
echo "===="

extracted_data="${raw_data_save_dir}/extracted_pcd_data/final_segmented_pcl.pcd"
./pcl/build/pcl_segmentation --pcd "${extracted_data}"

wait
sleep 2
echo "Did use PCL to segment objects."
echo "===="

extracted_data_dir="${raw_data_save_dir}/extracted_pcd_data/"
python ./visualize_cloud_clusters.py --pcd_dir ${extracted_data_dir}
wait
echo "Did visualize object clusters"
echo "===="