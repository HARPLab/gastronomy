uATCH_SIZE=64
args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  # --result_dir /tmp/mohit/experiment_results/robot_interface_utils/action_relation/sept_8_try_1/add_xy_channels_1_z_1024/loss_img_1.0_bb_100.0_local_perturb_10.0_Sept_9_11_59_PM
  # --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/sept_26_train_all_nopos_in_input/lr_0.0001_loss_10.0_Sept_27_5_45_PM
  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/scaled_voxels/sept_26_train_push_q_0_all_pos_only_other_noq/loss_classif_batch_32_lr_0.0001_loss_10.0_z_1069_Oct_1_10_45_AM/
  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/scaled_voxels_classif/Oct_3_voxel_0.250/use_non_orient_bb_input_add_xy_channels_batch_32_lr_0.0001_loss_1.0_Oct_6_11_10_PM/

  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/noscaled_voxels_expanded/Oct_3_voxel_0.25/use_non_orient_bb_input_add_xy_channels_batch_32_lr_0.0001_loss_1.0_Oct_7_11_25_PM

  # --checkpoint_path /tmp/experiment_results/robot_interface_utils/action_relation_voxel/scaled_voxels/sept_26_train_all/batch_32_lr_0.0001_loss_1.0_Oct_1_3_10_PM/checkpoint/cp_6500.pth
  # --checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/scaled_voxels/sept_26_train_all/batch_32_lr_0.0001_loss_10.0_Sept_28_7_00_PM/checkpoint/cp_4500.pth

  # --checkpoint_path /home/mohit/experiment_results/robot_interface_utils/action_relation_voxel/scaled_voxels/sept_26_train_push_q_0_pos_only/batch_32_lr_0.0001_loss_10.0_z_1069_Sept_29_8_40_PM/checkpoint/cp_24000.pth

  # --checkpoint_path ~/experiment_results/robot_interface_utils/action_relation_voxel/sept_26_train_all/lr_0.0001_loss_1.0_Sept_27_11_20_AM/checkpoint/cp_3000.pth

  #--train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0_all/test
  #--test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0_all/test

  --train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/train
  --test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/test

  #--train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0/test
  #--test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_push_q_0/test
  --cp_prefix 'test'

  # graph model args
  --z_dim 557  # for regression
  # --z_dim 278  # for classif

  --batch_size 32
  --num_epochs 1000
  --save_freq_iters 500
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 500

  --lr 0.0001
  --add_xy_channels 1
  --use_bb_in_input 1
  --voxel_datatype 0
  --scaled_mse_loss 1

  --weight_pos 1.0
  --weight_angle 0.0
  --weight_inv_model 0.0
  --loss_type 'classif'
  --classif_num_classes 5
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
python -m ipdb ./action_relation/trainer/train_voxels.py "${args[@]}"
