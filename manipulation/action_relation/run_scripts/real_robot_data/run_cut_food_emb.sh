BATCH_SIZE=64
args=(
  --cuda 1

  # --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/cut_food/unfactored_scene_resnet18/train_0_1_test_3_Dec_21_4_30_PM
  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/cut_food/factored_scene/no_contrastive_loss/lr_0.0003_train_0_1_2_test_4_Dec_23_7_40_PM
  --save_embedding_only 0

  #--emb_checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_21_8_55_PM/checkpoint/cp_16000.pth
  #--emb_save_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_21_8_55_PM/emb_data/cp_16000

   #--emb_checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Dec_21_8_55_PM/checkpoint/cp_16000.pth
   #--emb_save_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Dec_21_8_55_PM/real_robot_emb/data_cut_food/cp_16000kkkkkkkk

   #--emb_checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Dec_22_1_40_PM/checkpoint/cp_12000.pth
   #--emb_save_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_1.0_orient_1.0/l2_loss_no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.1_orient_0.1_Dec_22_1_40_PM/real_robot_emb/data_cut_food

   # NO CONTRASTIVE LOSS
   --emb_checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_22_1_40_PM/checkpoint/cp_12000.pth
   --emb_save_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_22_1_40_PM/real_robot_emb/data_cut_food
  
  # --emb_checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/small_simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_21_8_55_PM/checkpoint/cp_16000.pth
  # --emb_save_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/small_simple_model_nouse_bb_input_no_contact_no_ft/no_contrastive_loss/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_1.0_triplet_0.0_orient_0.0_Dec_21_8_55_PM/real_robot_data/cut_food/cp_16000

  #--train_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/train/
  #--test_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/test/

  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/no_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/one_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/no_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/one_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/two_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/two_obstacle
  # --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/true/2_obstacle/
  #--train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/false/two_obstacle

  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/Dec_2/test
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/true/objects_4
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/false/objects_4
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/true/two_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/false/two_obstacle
  
  #--test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/true/2_obstacle/
  #--test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/two_obstacle/split_1/test
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/three_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/true/3_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/three_obstacle
  --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/true/4_obstacle
  --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/false/4_obstacle

  # --train_type 'unfactored_scene'
  # --train_type 'unfactored_scene_resnet18'

  --train_type 'all_object_pairs_g_f_ij_cut_food'

  --cp_prefix 'test'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 512  # CML_2
  # --z_dim 128  # small simple model
  --batch_size 8
  --num_epochs 1000
  --save_freq_iters 200
  --log_freq_iters 5
  --print_freq_iters 1
  --test_freq_iters 25

  --lr 0.0003   # For simple conv model
  --emb_lr 0.0

  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_precond 1.0

  --loss_type 'classif'
  --classif_num_classes 2
  --use_spatial_softmax 0
  --max_train_data_size 1000
  --max_test_data_size 10000
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
python -m pdb ./action_relation/trainer/multi_object_precond/train_multi_object_precond_e2e.py "${args[@]}"

