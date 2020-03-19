BATCH_SIZE=64
args=(
  --cuda 1

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/precond_classif/classify_from_emb_e2e/Oct_17_data_1/cml_2_action_index_0_2_4_6_input_100/l1_loss_margin_1.0_gt_0.04_no_xy_l2_loss_100.0_triplet_0.1_Oct_20_6_15_PM/use_init_emb_lr_1e-5_lr_decay_0.9_no_relu_cp_13000_batch_64_lr_0.0003_loss_1.0_Oct_23_1_45_PM
  #--result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/factored_scene/cp_24000_type_all_object_pairs_g_f_ij_Dec_10_6_10_PM
  --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/factored_scene/cp_24000_type_all_object_pairs_g_f_ij_train_2_3_test_5_Dec_10_6_15_PM

  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/checkpoint/cp_20000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/emb_data/cp_20000

  --emb_checkpoint_path  /tmp/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/new_push_data_Dec_5_temp_voxel_0.01_dense/z_512_cml_2_no_relu_on_emb_action_index_0_2_4_6_nouse_bb_input_no_emb/margin_1.0_gt_margin_0.04/no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_triplet_0.1_Dec_6_5_55_PM/checkpoint/cp_24000.pth
  --emb_save_path  /tmp/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/new_push_data_Dec_5_temp_voxel_0.01_dense/z_512_cml_2_no_relu_on_emb_action_index_0_2_4_6_nouse_bb_input_no_emb/margin_1.0_gt_margin_0.04/no_add_xy_channels_batch_128_lr_0.0003_pos_1.0_triplet_0.1_Dec_6_5_55_PM/emb_data/all_true_cp_24000/
  --save_embedding_only 0

  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/checkpoint/cp_13000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/emb_data/no_relu_cp_13000

  #--train_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/train/
  #--test_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/test/

  --train_dir ~/datasets/robot_data/action_based_relation/data_in_line/Dec_2/train/
  --train_dir ~/datasets/robot_data/action_based_relation/data_in_line/true/objects_2/
  --train_dir ~/datasets/robot_data/action_based_relation/data_in_line/false/objects_2/

  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/Dec_2/test
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/true/objects_4
  # --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/false/objects_4
  --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/true/objects_5
  --test_dir ~/datasets/robot_data/action_based_relation/data_in_line/false/objects_5

  # --train_type 'unfactored_scene'
  --train_type 'all_object_pairs_g_f_ij'

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 512  # CML_2
  # --z_dim 128  # Unfactored scene
  --batch_size 4
  --num_epochs 400
  --save_freq_iters 200
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 10

  --lr 0.001
  --emb_lr 0.001
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

