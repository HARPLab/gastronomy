BATCH_SIZE=64
args=(
  --cuda 1

  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/precond_classif/classify_from_voxels/Oct_17_data_1/cml_2_action_index_0_2_4_6_input_100/l1_loss_margin_1.0_gt_0.04_no_xy_l2_loss_100.0_triplet_0.1_Oct_20_6_15_PM/no_relu_cp_13000_batch_64_lr_0.0003_loss_1.0_Oct_21_10_35_PM

  --use_embeddings 1
  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/checkpoint/cp_20000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_14_11_55_PM/emb_data/cp_20000

  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_40_PM/checkpoint/cp_13000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_1.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_40_PM/emb_data/use_relu_cp_13000

  #--emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/checkpoint/cp_13000.pth
  #--emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_19_11_55_PM/emb_data/no_relu_cp_13000

  --emb_checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_1.0_gt_margin_0.04_fixed_den_l1_loss/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_20_6_15_PM/checkpoint/cp_13000.pth
  --emb_save_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/regr_with_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0_2_4_6/margin_1.0_gt_margin_0.04_fixed_den_l1_loss/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_100.0_triplet_0.1_Oct_20_6_15_PM/emb_data/no_relu_cp_13000


  --train_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/train/
  --test_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/test/

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 512  # CML_2
  --batch_size 32
  --num_epochs 400
  --save_freq_iters 200
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 100

  --lr 0.0003
  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_precond 1.0

  --loss_type 'classif'
  --classif_num_classes 2
  --use_spatial_softmax 0
  --max_train_data_size 100
  --max_test_data_size 10000
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
python -m ipdb ./action_relation/trainer/train_voxel_precond.py "${args[@]}"
