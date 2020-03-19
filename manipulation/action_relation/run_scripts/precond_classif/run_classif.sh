BATCH_SIZE=64
args=(
  --cuda 1

  --result_dir /tmp/experiment_results/robot_interface_utils/action_relation_voxel/precond_classif/classify_from_voxels/Oct_17_data_1/cml_2_input_size_100/no_bb_input_batch_128_lr_0.0003_loss_1.0_Oct_21_9_30_PM

  --train_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/train/
  --test_dir ~/datasets/vrep_data/relation_on_drop/precond_data/fall_if_removed/Oct_17_push_q_0_anchor_cuboid_other_cuboid_check_fall/test/

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  --z_dim 512  # CML_2
  --batch_size 16
  --num_epochs 200
  --save_freq_iters 100
  --log_freq_iters 5
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
