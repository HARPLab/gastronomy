BATCH_SIZE=64

args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/only_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_2.0_gt_margin_0.04/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_0.0_triplet_1.0_Oct_14_8_15_PM

   # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/resnet_no_bb_input_no_contact_no_ft/orient_classif_5/use_triplet_loss_margin_gt_pos_0.005_orient_0.005_emb_pos_1.0_orient_1.0/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Nov_6_11_05_PM
   # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/resnet_no_bb_input_no_contact_no_ft/orient_classif_5/nouse_triplet_loss/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_use_final_orient_change_data_no_use_Oct_30_data_eval_Nov_7_8_20_PM

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/simple_model_no_bb_input_no_contact_no_ft/orient_classif_3/use_orient_triplet_loss_gt_pose_diff_0.01/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_1.0_Nov_noeval_on_Oct_30_data_Oct_8_1_40_AM

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/simple_model_use_bb_input_no_contact_no_ft/orient_classif_3/no_triplet/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Nov_noeval_on_Oct_30_data_Oct_8_4_55_AM
   # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/simple_model_no_bb_input_no_contact_no_ft/orient_classif_3/use_orient_triplet_loss_gt_pose_diff_0.01/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_1.0_Nov_noeval_on_Oct_30_data_Oct_8_1_40_AM
   --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Oct_31_scene_4_orient_change/resnet10_no_bn_at_all_no_bb_input_no_contact_no_ft/orient_classif_3_inc_loss_wt/nouse_triplet_loss/use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_use_final_orient_change_data_no_use_Oct_30_data_eval_Nov_8_4_20_PM


   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_0
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_1
   ## --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_2
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_0
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_1
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1/try_0
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1/try_1
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1/try_2
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1/try_3
   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1/try_4

   # --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_4_200_scenes
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_2
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1_init/try_5

   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_4
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_5
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_6
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_7
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_2
   #--train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_3
#--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_4
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_5
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_6
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_30_spring_damper_try_0/try_7
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_2
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1/try_3
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1_init/try_4
   #--test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_orient_change/Nov_2_try_1_init/try_5


  # --cp_prefix 'test_cp_7000_spring_damper_4_5_6_7'
  --cp_prefix 'test_cp_7000_orient_change_4_5'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 512 # CML_2
  # --z_dim 512  # ResNet-18
  --batch_size 128
  --num_epochs 400
  --save_freq_iters 500
  --log_freq_iters 5 
  --print_freq_iters 5
  --test_freq_iters 400

  --model 'resnet10'
  --lr 0.0003
  --add_xy_channels 1
  --use_bb_in_input 1
  --voxel_datatype 0
  --scaled_mse_loss 0
  --octree_0_multi_thread 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_pos 1.0
  --weight_angle 1.0
  --weight_inv_model 0.0
  --weight_contrastive_loss 0.0
  --weight_orient_contrastive_loss 0.0
  --weight_ft_force 0.0
  --weight_ft_torque 0.0

  --pos_loss_type 'regr'
  --use_l1_loss 1
  --orient_loss_type 'classif'
  --orient_classif_num_classes 3
  --use_spatial_softmax 0
  --use_contrastive_loss 1
  --contrastive_margin 1.0
  --contrastive_gt_pose_margin 0.005
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
python -m ipdb ./action_relation/trainer/train_voxels_online_contrastive_3.py "${args[@]}"

