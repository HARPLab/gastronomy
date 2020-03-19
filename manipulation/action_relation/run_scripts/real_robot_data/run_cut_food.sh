BATCH_SIZE=64
args=(
  --cuda 1

  --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/cut_food/unfactored_scene_resnet18/max_data_dir_20_train_0_1_2_test_4_Dec_23_6_50_PM

  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/checkpoint/cp_20000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/emb_data/cp_20000

  # --emb_checkpoint_path  /tmp/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/new_push_data_Dec_5_temp_voxel_0.01_dense/z_512_cml_2_no_relu_on_emb_action_index_0_2_4_6_nouse_bb_input_no_emb/margin_1.0_gt_margin_0.04/no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_triplet_0.1_Dec_6_5_55_PM/checkpoint/cp_24000.pth
  # --emb_save_path  /tmp/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/new_push_data_Dec_5_temp_voxel_0.01_dense/z_512_cml_2_no_relu_on_emb_action_index_0_2_4_6_nouse_bb_input_no_emb/margin_1.0_gt_margin_0.04/no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_triplet_0.1_Dec_6_5_55_PM/emb_data/all_true_cp_24000/
  --save_embedding_only 0


  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/checkpoint/cp_13000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/emb_data/no_relu_cp_13000

  #--train_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/train/
  #--test_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/test/

  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/no_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/one_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/no_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/one_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/true/two_obstacle
  --train_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17_knife_3/false/two_obstacle
  --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/true/4_obstacle
  --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_22/false/4_obstacle

  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/Dec_2/test
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/true/objects_4
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/false/objects_4
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/true/two_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/false/two_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/true/three_obstacle
  # --test_dir ~/datasets/robot_data/action_based_relation/data_cut_food/try_1_Dec_17/false/three_obstacle

  # --train_type 'unfactored_scene'
  --train_type 'unfactored_scene_resnet18'

  # --train_type 'all_object_pairs_g_f_ij'

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  # --z_dim 512  # CML_2
  --z_dim 128  # Unfactored scene
  --batch_size 4
  --num_epochs 400
  --save_freq_iters 200
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 25

  --lr 0.0003   # For simple conv model
  --emb_lr 0.0003

  # --lr 0.0001  # For resnet based models that classify directly from the entire scene
  # --emb_lr 0.0001 # For resnet based models that classify directly from the entire scene

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

