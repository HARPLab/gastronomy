BATCH_SIZE=64
args=(
  --cuda 1

  --result_dir ~/experiment_results/robot_interface_utils/action_relation_voxel/real_robot_data/box_stacking/unfactored_scene_resnet/train_3_4_all_test_6_Dec_29_7_10_PM

  --save_embedding_only 0

  --train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/3_obj
  --train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/4_obj/train
  --train_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/4_obj/test
  --test_dir ~/datasets/robot_data/action_based_relation/block_stacking/try_1_Dec_24/split_2/6_obj_fixed_labels

  # --train_type 'unfactored_scene'
  --train_type 'unfactored_scene_resnet18'

  --cp_prefix 'train'

  # graph model args
  # --z_dim 61  # CML
  # --z_dim 512  # CML_2
  --z_dim 128  # Unfactored scene
  --batch_size 8
  --num_epochs 400
  --save_freq_iters 200
  --log_freq_iters 10
  --print_freq_iters 1
  --test_freq_iters 50

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
  --use_dynamic_bce_loss 1

  --loss_type 'classif'
  --classif_num_classes 2
  --use_spatial_softmax 0
  --max_train_data_size 10000
  --max_test_data_size 10000
)

echo "${args[@]}"

export PYTHONPATH=/home/mohit/projects/robot-interface-utils:$PYTHONPATH
cd /home/mohit/projects/robot-interface-utils
python -m pdb ./action_relation/trainer/multi_object_precond/train_multi_object_precond_e2e.py "${args[@]}"

