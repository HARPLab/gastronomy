BATCH_SIZE=64
args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  # --result_dir /tmp/mohit/experiment_results/robot_interface_utils/action_relation/sept_8_try_1/add_xy_channels_1_z_1024/loss_img_1.0_bb_100.0_local_perturb_10.0_Sept_9_11_59_PM
  --result_dir ~/experiment_results/robot_interface_utils/action_relation/sept_11/add_xy_channels_1_spatial_softmax_z_256_try_8_no_scene_1/loss_img_1.0_bb_10.0_Sept_13_2_30_PM
  # --checkpoint_path ~/experiment_results/robot_interface_utils/action_relation/sept_11/add_xy_channels_1_spatial_softmax_z_256_try_8_no_scene_1/loss_img_1.0_bb_10.0_Sept_13_1_30_PM/checkpoint/cp_3000.pth
  # --checkpoint_path /tmp/mohit/experiment_results/robot_interface_utils/action_relation/sept_8_try_1/add_xy_channels_1/loss_img_1.0_bb_100.0_local_perturb_10.0_Sept_9_11_59_PM/checkpoint/cp_23000.pth
  # --checkpoint_path /tmp/mohit/experiment_results/robot_interface_utils/action_relation/sept_8_try_1/add_xy_channels_1/loss_img_1.0_bb_100.0_local_perturb_10.0_Sept_9_11_25_PM/checkpoint/cp_6000.pth

  --train_dir ~/datasets/action_based_relation/sept_8/try_8_no_scene_1/train
  --test_dir ~/datasets/action_based_relation/sept_8/try_8_no_scene_1/test

  # graph model args
  --z_dim 256
  --batch_size 32
  --num_epochs 2000
  --save_freq_iters 1000
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 100

  --lr 0.0001
  --add_xy_channels 1
  --use_bb_in_input 0

  --weight_bb 10.0
  --weight_inv_model 1.0
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
python -m ipdb ./action_relation/trainer/train.py "${args[@]}"
