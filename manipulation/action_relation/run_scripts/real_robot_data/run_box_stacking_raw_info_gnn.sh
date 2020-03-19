BATCH_SIZE=64
args=(
  --cuda 1

  # --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/cut_food/unfactored_scene_resnet18/train_0_1_test_3_Dec_21_4_30_PM
  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/box_stacking/factored_scene_gnn/loss_pred_contrastive/raw_info/weight_bce_batch_16_no_stable_loss_lr_0.0003_train_3_4_test_6_Dec_29_8_10_PM
  --save_embedding_only 0


  # resnet model
  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/resnet_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_2.0_orient_2.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_20.0_triplet_5._orient_5.0_Dec_23_10_10_PM/checkpoint/cp_24000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/resnet_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_2.0_orient_2.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_20.0_triplet_5._orient_5.0_Dec_23_10_10_PM/real_robot_data/data_box_stacking

   --emb_checkpoint_path ~/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/small_simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_2.0_orient_2.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_20.0_triplet_5._orient_5.0_Dec_23_10_10_PM/checkpoint/cp_24000.pth
   --emb_save_path ~/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/Dec_18_scene_4_orient_change_relative_action/orient_change_dataloader/small_simple_model_nouse_bb_input_no_contact_no_ft/use_contrastive_loss_gt_pos_0.2_orient_0.005_margin_pos_2.0_orient_2.0/l2_loss_use_add_xy_channels_batch_128_lr_0.0003_pos_1.0_orient_20.0_triplet_5._orient_5.0_Dec_23_10_10_PM/real_robot_data/data_box_stacking

  #--train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/true/3_obj
  #--test_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/true/4_obj

  --train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/3_obj
  --train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/4_obj/
  --test_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/6_obj_fixed_labels

  --train_type 'all_object_pairs_gnn_raw_obj_info'

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
  --test_freq_iters 50

  --lr 0.0003   # For simple conv model
  --emb_lr 0.0

  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_precond 1.0
  --use_dynamic_bce_loss 1

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

