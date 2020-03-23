args=(
    # --anchor_pcd_path ~/datasets/robot_data/action_based_relation/data_in_line/try_2/true/objects_3/try_0_Dec_01_2019_11_00_PM/extracted_pcd_data/cloud_cluster_0.pcd
    # --other_pcd_path ~/datasets/robot_data/action_based_relation/data_in_line/try_2/true/objects_3/try_0_Dec_01_2019_11_00_PM/extracted_pcd_data/cloud_cluster_1.pcd

    # --pcd_dir ~/datasets/robot_data/action_based_relation/data_in_line/try_2/true/objects_3/try_2_Dec_01_2019_11_07_PM/extracted_pcd_data
    --pcd_dir ~/datasets/robot_data/action_based_relation/data_in_line/try_2/false/objects_3/try_0_Dec_02_2019_12_49_PM/extracted_pcd_data 
    --vis_type 'all_obj_pair'
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
python -m pdb ./action_relation/dataloader/robot_octree_data_vis_helper.py "${args[@]}"

