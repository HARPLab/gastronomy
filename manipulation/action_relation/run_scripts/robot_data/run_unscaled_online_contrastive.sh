BATCH_SIZE=64

args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  # --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
  --cuda 1

  # --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/only_contrastive_loss_2/Oct_3_voxel_0.25/cml_2_action_index_0/margin_2.0_gt_margin_0.04/no_bb_input_no_add_xy_channels_batch_128_lr_0.0003_loss_0.0_triplet_1.0_Oct_14_8_15_PM

   --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/vrep_robot_data/regr_with_contrastive_loss_2/Oct_3_voxel_0.01/cml_2_no_use_bb_input/margin_1.0_gt_margin_0.04_orient_1.0_orient_gt_margin_0.01_fixed_den_l1_loss/no_add_xy_channels_batch_128_lr_0.0003_pos_10.0_orient_100.0_triplet_0.1_orient_0.0_Nov_1_11_59_PM

   --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1_train/train/
   --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/scene_4_go_down/Oct_31_try_1_train/test/

  # --train_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_29_spring_damper_init/train
  # --test_dir ~/datasets/vrep_data/relation_on_drop/robot_data/Oct_29_spring_damper_init/test


  --cp_prefix 'test'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 509 # CML_2
  # --z_dim 512  # ResNet-18
  --batch_size 128
  --num_epochs 400
  --save_freq_iters 500
  --log_freq_iters 5 
  --print_freq_iters 5
  --test_freq_iters 400

  --model 'simple_model'
  --lr 0.0003
  --add_xy_channels 0
  --use_bb_in_input 0
  --voxel_datatype 0
  --scaled_mse_loss 0
  --octree_0_multi_thread 0
  --save_full_3d 1
  --expand_voxel_points 0

  --weight_pos 10.0
  --weight_angle 100.0
  --weight_inv_model 0.0
  --weight_contrastive_loss 0.1
  --weight_orient_contrastive_loss 0.0

  --loss_type 'regr'
  --use_l1_loss 1
  --classif_num_classes 5
  --use_spatial_softmax 0
  --use_contrastive_loss 1
  --contrastive_margin 1.0
  --contrastive_gt_pose_margin 0.005
  --use_orient_contrastive_loss 0
  --orient_contrastive_margin 1.0
  --orient_contrastive_gt_pose_margin 0.01
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
# cd /home/tanya/project/robot-interface-utils
python -m ipdb ./action_relation/trainer/train_voxels_online_contrastive.py "${args[@]}"

