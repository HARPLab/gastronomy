BATCH_SIZE=64
args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/scaled_vs_unscaled_voxels/noscaled_loss_no_spatial_softmax_cml_2/no_bb_add_xy_batch_size_16_lr_0.0001_Oct_11_3_30_PM

  # --train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_all/test/
  # --test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/train_all/test/
  --train_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/train
  --test_dir ~/datasets/vrep_data/relation_on_drop/Sept_26_final/cuboid_only_data/Oct_3_voxel_0.250/test
  --cp_prefix 'test'

  # graph model args
  --z_dim 557
  --batch_size 32
  --num_epochs 2000
  --save_freq_iters 500
  --log_freq_iters 10
  --print_freq_iters 5
  --test_freq_iters 100

  --lr 0.0001
  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 1

  --weight_bb 10.0
  --weight_inv_model 0.0
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
python -m ipdb ./action_relation/trainer/train_voxels.py "${args[@]}"

