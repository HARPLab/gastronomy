BATCH_SIZE=64
args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/only_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_2.0_gt_margin_0.04/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_0.0_triplet_1.0_Oct_14_8_15_PM
  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/only_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_2.0_gt_margin_0.04_fixed_den/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_0.0_triplet_1.0_Oct_14_10_05_PM/

  #--train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0_all/train
  #--test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0_all/test

  --train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/train
  --test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/test

  #--train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0/test
  #--test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0/test

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 509  # CML_2
  --batch_size 128
  --num_epochs 2000
  --save_freq_iters 500
  --log_freq_iters 4
  --print_freq_iters 2
  --test_freq_iters 100

  --lr 0.0003
  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --scaled_mse_loss 0
  --octree_0_multi_thread 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_pos 0.0
  --weight_angle 0.0
  --weight_inv_model 0.0
  --weight_contrastive_loss 1.0

  --loss_type 'regr'
  --classif_num_classes 5
  --use_spatial_softmax 0
  --use_contrastive_loss 0
  --use_online_contrastive_loss 1
  --contrastive_margin 2.0
  --contrastive_gt_pose_margin 0.04


  # contrastive loss args
  --online_contrastive_batch_size 128
  --online_contrastive_test_freq 5
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
# python -m ipdb ./action_relation/trainer/train_voxels_online_contrastive.py "${args[@]}"
python -m ipdb ./action_relation/trainer/train_voxels_online_contrastive_2.py "${args[@]}"
