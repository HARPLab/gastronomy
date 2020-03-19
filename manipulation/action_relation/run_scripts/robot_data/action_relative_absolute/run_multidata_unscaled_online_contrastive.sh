BATCH_SIZE=64

args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

   --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/small_simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_50.0_triplet_1._orient_1.0_Dec_28_2_25_PM

  # --checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Dec_21_8_55_PM/checkpoint/cp_16000.pth


   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/4_sample_types/train
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/1_edge_1_in_1_out/train
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/sample_2_edge_1_out/train/
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/4_sample_types/train/
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_0/train
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_1_min_th_0.01_max_th_0.10/train
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_3_min_th_0.01_max_th_0.04/train


   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/4_sample_types/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/1_edge_1_in_1_out/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_in_air/sample_2_edge_1_out/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/4_sample_types/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_0/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_1_min_th_0.01_max_th_0.10/test
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/small_objects_dense_voxel_Dec_18_temp/anchor_on_ground/sample_2_edge_1_out/try_3_min_th_0.01_max_th_0.04/test

  # --cp_prefix 'test_cp_7000_spring_damper_4_5_6_7'
  --cp_prefix 'test'

  # graph model args
  # --z_dim 61  # CML
  # --z_dim 509 # CML_2
  --z_dim 128 # CML_4
  # --z_dim 512 # ResNet-18
  --batch_size 128
  --num_epochs 400
  --save_freq_iters 500
  --log_freq_iters 5 
  --print_freq_iters 5
  --test_freq_iters 400

  --dataloader 'orient_change_sampler'
  --model 'small_simple_model'
  # --model 'resnet18'
  --lr 0.0003
  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --scaled_mse_loss 0
  --octree_0_multi_thread 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_pos 1.0
  --weight_angle 50.0
  --weight_inv_model 0.0
  --weight_contrastive_loss 1.0
  --weight_orient_contrastive_loss 1.0
  --weight_ft_force 0.0
  --weight_ft_torque 0.0

  --loss_type 'regr'
  --use_l1_loss 1
  --classif_num_classes 5
  --use_spatial_softmax 0
  --use_contrastive_loss 1
  --contrastive_margin 1.0
  --contrastive_gt_pose_margin 0.2
  --use_orient_contrastive_loss 1
  --orient_contrastive_margin 1.0
  --orient_contrastive_gt_pose_margin 0.005

  --use_ft_sensor 0
  --use_contact_preds 0
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
python -m ipdb ./action_relation/trainer/train_voxels_online_contrastive.py "${args[@]}"

