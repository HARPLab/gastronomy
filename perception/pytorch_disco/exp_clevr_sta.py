from exp_base import *
import os
import ipdb
st = ipdb.set_trace
# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE
############## choose an experiment ##############

current = '{}'.format(os.environ["exp_name"])


mod = '"{}"'.format(os.environ["run_name"]) # debug

############ Final Config ##################

exps['clevr_multiview_training'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'fastest_logging',
]

exps['clevr_trainer_gen_examples_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate_test', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'create_example_dict_50',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_frozen'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout_hardneg'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'high_neg',
    'self_improve_iterate_100_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

############## define experiments ##############


exps['replica_multiview_builder'] = [
    'replica_sta', # mode
    'replica_test_sta_data', # dataset
    '200_iters',
    'lr3',
    'B2',
    # 'no_shuf',
    'train_feat',
    'train_occ',
    'train_view',
    'eval_boxes',
    'debug',
    # 'debug',
    # 'vis_clusters',
    # 'eval_recall_o_quicker',   
    # 'orient_tensors_in_eval_recall_o',    
    # 'profile_time',
    'fastest_logging',
]



exps['replica_multiview_trainer'] = [
    'replica_sta', # mode
    'replica_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    # 'no_shuf',
    'reset_iter',
    'train_feat',
    'train_occ',
    'train_view',
    # 'pretrained_feat',
    # 'pretrained_occ',    
    # 'pretrained_view',    
    'eval_boxes',
    # 'debug',
    # 'vis_clusters',
    # 'eval_recall_o_quicker',   
    # 'orient_tensors_in_eval_recall_o',    
    # 'profile_time',
    'fast_logging',
]


exps['replica_trainer_orient_viewContrast_moc'] = [
    'replica_sta', # mode
    'replica_sta_data',
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'do_moc',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]

exps['replica_trainer_orient_viewContrast'] = [
    'replica_sta', # mode
    'replica_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_emb3D_o',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]


exps['replica_det_trainer_freeze_vq_rotate_deep_det'] = [
    'replica_sta', # mode
    'replica_sta_data', # dataset
    '500k_iters',
    'eval_boxes',    
    # 'no_shuf',
    'pretrained_feat',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]




exps['replica_trainer_quantize_object_no_detach_rotate'] = [
    'replica_sta', # mode
    'replica_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    'quantize_object_no_detach_rotate_100',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
    # 'S3',    
]

exps['replica_trainer_gen_examples_multiple'] = [
    'replica_sta', # mode
    'replica_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',
    'pretrained_feat',
    'create_example_dict_50',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['replica_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_frozen'] = [
    'replica_sta', # mode
    'replica_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50_init_replica_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]



groups['replica_test_sta_data'] = [
    'dataset_name = "replica"',
    'H = %d' % 256,
    'W = %d' % 256,
    'N = %d' % 50,
    'testset = "cct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/replica_processed/npy"',
    'dataset_location = "CHANGE_ME/replica_processed/npy"',
    'root_dataset = "CHANGE_ME/replica_processed"',
    'dataset_format = "npz"',
]


groups['replica_sta_data'] = [
    'dataset_name = "replica"',
    'H = %d' % 256,
    'W = %d' % 256,
    'N = %d' % 50,
    'trainset = "cct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/replica_processed/npy"',
    'dataset_location = "CHANGE_ME/replica_processed/npy"',
    'root_dataset = "CHANGE_ME/replica_processed"',
    'dataset_format = "npz"',
]


groups['replica_sta_data_aa'] = [
    'dataset_name = "replica"',
    'H = %d' % 256,
    'W = %d' % 256,
    'N = %d' % 50,
    'trainset = "aat"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/replica_processed/npy"',
    'dataset_location = "CHANGE_ME/replica_processed/npy"',
    'root_dataset = "CHANGE_ME/replica_processed"',
    'dataset_format = "npz"',
]

########### carla specific stuff goes here ##############
exps['carla_multiview_builder'] = [
    'carla_det', # mode
    'carla_sta_data_mix_1_test', # dataset
    '200_iters',
    'lr3',
    'B2',
    # 'no_shuf',
    'train_feat',
    'train_occ',
    'train_view',
    'eval_boxes',
    'vis_clusters',
    # 'eval_recall_o_quicker',   
    # 'orient_tensors_in_eval_recall_o',    
    # 'profile_time',
    'fastest_logging',
]






##### detector
exps['carla_det_builder'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    # 'carla_sta_data_temp',
    '100_iters',
    'eval_boxes',
    'no_shuf',
    'frozen_feat',
    'debug',
    'lr3',
    'B2',
    'train_feat',
    # 'train_det_deep',
    'fastest_logging',
]

exps['carla_det_trainer_big_freeze_vq_rotate_deep_det_1'] = [
    'carla_det', # mode
    'carla_sta_data_mix_1', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['carla_det_trainer_big_freeze_vq_rotate_deep_det_pret_rgb_occ'] = [
    'carla_det', # mode
    'carla_sta_data_fc', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['carla_det_trainer_big_freeze_vq_rotate_deep_det_from2d_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'use_2d_boxes',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['carla_det_trainer_big_freeze_vq_rotate_deep_det_from2d'] = [
    'carla_det', # mode
    'carla_sta_data_mix_1', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'use_2d_boxes',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['carla_det_trainer_big_freeze_vq_rotate_deep_det_test'] = [
    'carla_det', # mode
    'carla_sta_data_mix_2_test', # dataset
    '500k_iters',
    'eval_boxes',
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'break_constraint',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]


exps['carla_det_trainer_big_freeze_vq_rotate_filterboxes_deep_test_cs'] = [
    'carla_det', # mode
    'carla_sta_data_mix_2_test',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_100',
    'only_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]


exps['carla_det_trainer_big_freeze_vq_rotate_filterboxes_deep_test_q_cs'] = [
    'carla_det', # mode
    'carla_sta_data_mix_2_test',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_100',
    'q_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]

exps['carla_det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout_hardneg'] = [
    'carla_det', # mode
    'carla_sta_data_mix_2',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'high_neg',
    'self_improve_iterate_100_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['carla_det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout'] = [
    'carla_det', # mode
    'carla_sta_data_mix_2',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'maskout',
    'self_improve_iterate_100_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

##### detector



#view pred

exps['carla_test_eval_recall_hard_vis'] = [
    'carla_det', 
    'carla_test_sta_data', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
    'do_hard_eval',
    'hard_vis',
    # 'debug',
]


exps['carla_trainer_quantize_object_no_detach_rotate_combvis'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_100',
    'rotate_combinations',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_combvis_multi'] = [
    'carla_det', # mode
    'carla_sta_data_mix_1_test', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_100',
    'eval_boxes',
    'rotate_combinations',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]


exps['carla_test_eval_recall_rotation_check'] = [
    'carla_det', 
    'carla_test_sta_data',
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
]




exps['carla_trainer_rgb_occ_orient'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ', 
    'eval_boxes',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]



exps['carla_trainer_rgb_orient'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    # 'train_occ', 
    'eval_boxes',
    # 'eval_recall_o',   
    # 'orient_tensors_in_eval_recall_o',
    # 'pretrained_feat',
    # 'pretrained_occ',
    # 'pretrained_view',
    'fast_logging',
]


exps['carla_trainer_occ'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    # 'train_view',
    'train_occ', 
    'eval_boxes',
    # 'eval_recall_o',   
    # 'orient_tensors_in_eval_recall_o',
    # 'pretrained_feat',
    # 'pretrained_occ',
    # 'pretrained_view',
    'fast_logging',
]



exps['carla_trainer_rgb_occ_orient_low_res'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ', 
    'eval_boxes',
    'low_res',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]

exps['bigbird_tester_rgb_occ_orient_gg'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg_test', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ', 
    'break_constraint',
    # 'eval_recall_o',
    'eval_recall_o_vbig_pool',   
    'cpu',
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]

exps['carla_trainer_rgb_occ_orient_low_res_viewContrast'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'do_moc',
    'eval_boxes',
    'low_res',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]


#view pred





#vqvae

exps['carla_trainer_gen_examples'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    'shuf',
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'create_example_dict_52',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


#prototype reconstruction
exps['carla_trainer_quantize_object_no_detach_rotate_50'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    'resume',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

# ?no detach
exps['carla_trainer_quantize_object_no_detach_50'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    # 'resume',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_50',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

#prototype reconstruction
#rotate frozen


exps['carla_trainer_quantize_object_no_detach_rotate_100_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_100_init_examples_hard_only_embed_frozen_tester'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_quantized',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['carla_trainer_quantize_object_no_detach_rotate_70_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_70_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_40_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_10_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_10_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

#no rotate frozen
exps['carla_trainer_quantize_object_no_detach_100_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_100_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_70_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_70_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_40_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_10_init_examples_hard_only_embed_frozen'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_10_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]







exps['carla_trainer_quantize_object_no_detach_rotate_52_init_examples_hard_only_embed_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_52_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]




exps['carla_trainer_quantize_object_no_detach_rotate_70_init_examples_hard_only_embed_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_70_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_40_init_examples_hard_only_embed_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['carla_trainer_quantize_object_no_detach_rotate_10_init_examples_hard_only_embed_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_10_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


# exps['carla_trainer_quantize_object_no_detach_rotate_combvis'] = [
#     'clevr_sta', # mode
#     'carla_sta_data_mix_1', # dataset
#     '200k_iters',
#     'lr3',
#     'B1',
#     'no_bn',        
#     'pretrained_feat',
#     'pretrained_view',
#     'pretrained_quantized',
#     'quantize_object_no_detach_rotate',
#     'rotate_combinations',
#     'eval_recall_o',
#     'train_feat',
#     'train_view',
#     'break_constraint',    
#     'fastest_logging',
# ]


# vqvae and view pred


groups['carla_test_sta_data'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "bbt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_test_sta_data_1'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "bb_at"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_test_sta_data_2'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "bb_bt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_test_sta_data_fc'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "fct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_fc'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "fct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_sta_data'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bbt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_temp'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "tv_updatedt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_single'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "mct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_single_test'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "mct"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_sta_data_single_small'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "mc_smallt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_sta_data_single_small_test'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "mc_smallt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_mix_1'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bb_tv_at"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]



groups['carla_sta_data_mix_2'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bb_tv_bt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]




groups['carla_sta_data_mix_1_test'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "tv_updatedt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_mix_2_test'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
    'testset = "bb_tv_bt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]


groups['carla_sta_data_1'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bb_at"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

groups['carla_sta_data_2'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bb_bt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

########## Carla specific stuff ends here ###############




########### bird specific stuff goes here ##############

exps['bigbird_multiview_builder'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg_test', # dataset
    '100_iters',
    'lr3',
    'B2',
    'train_feat',
    # 'train_occ',
    # 'train_view',
    # 'profile_time',
    'eval_boxes',
    'fastest_logging',
]




















# VQVAE AND VIEW PRED

# exps['bigbird_tester_rgb_occ_orient'] = [
#     'bigbird_sta', # mode
#     'bigbird_multiview_data_ee_test', # dataset
#     '500k_iters',
#     'lr3',
#     'B2',
#     'no_bn',
#     'train_feat',
#     'train_view',
#     'train_occ', 
#     'break_constraint',
#     'eval_recall_o',   
#     'cpu',
#     'orient_tensors_in_eval_recall_o',
#     'pretrained_feat',
#     'pretrained_occ',
#     'pretrained_view',
#     'fast_logging',
# ]

# vqvae

exps['bigbird_trainer_rgb_occ_orient_lr0'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr0',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ',
    'train_occ',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    # 'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]







exps['bigbird_trainer_orient_viewContrast_moc'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg',
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'do_moc',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]

exps['bigbird_trainer_orient_viewContrast'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_emb3D_o',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]

exps['bigbird_trainer_rgb_orient'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'fast_logging',
]


exps['bigbird_trainer_rgb_orient_tester'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg_test', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'break_constraint',
    'pretrained_feat',
    'train_view',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'fast_logging',
]

exps['bigbird_trainer_occ_orient'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_occ', 
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'fast_logging',
]

# exps['bigbird_trainer_gen_examples_multiple'] = [
#     'bigbird_sta', # mode
#     'bigbird_multiview_data_gg_test', # dataset
#     '500k_iters',
#     'lr4',
#     'B2',
#     'no_bn',        
#     'pretrained_feat',
#     'create_example_dict_50',
#     'eval_boxes',
#     'train_feat',
#     'break_constraint',    
#     'fast_logging',
# ]



exps['bigbird_trainer_gen_examples_multiple'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg_test', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'create_example_dict_82',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['bigbird_trainer_quantize_object_no_detach_rotate_82_init_examples_hard_only_embed_test_multiple1'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_82_init_bigbird_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['bigbird_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_frozen'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50_init_bigbird_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

#no rotate frozen
exps['bigbird_trainer_quantize_object_no_detach_50_init_examples_hard_only_embed_frozen'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_gg', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_50_init_bigbird_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]













exps['bigbird_trainer_quantize_object_no_detach_rotate_multiple_fixed'] = [
    'bigbird_sta', # mode
    'bigbird_multiview_data_dd', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_100',
    'orient_tensors_in_eval_recall_o',
    'eval_recall_o',
    'quick_snap',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

# vqvae AND viewpred




























groups['bigbird_multiview_data_cc'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'trainset = "cct"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

groups['bigbird_multiview_data_ee'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'trainset = "eet"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

groups['bigbird_multiview_data_ff'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'trainset = "fft"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]


groups['bigbird_multiview_data_gg'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'trainset = "ggt"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

groups['bigbird_multiview_data_gg_test'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'testset = "ggt"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]


groups['bigbird_multiview_data_ee_test'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'testset = "eet"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

########## bird specific stuff ends here ###############


########## CLEVR specific stuff STARTS here ###############


# self improving detector


# -------------------------------------------------- TRAINERS ---------------------------------------------------

# detector

exps['det_builder'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '100_iters',
    'eval_boxes',    
    'no_shuf',
    'lr3',
    'B2',
    'train_feat',
    'train_det',
    'fastest_logging', 
]


exps['det_builder_det_boxes'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '100_iters',
    'eval_boxes',    
    'no_shuf',
    'lr3',
    'B2',
    'use_det_boxes',
    'train_feat',
    'train_det',
    'fastest_logging', 
]

exps['det_trainer_big_freeze_vq_rotate'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det',
    'fast_logging', 
]


exps['det_trainer_big_freeze_vq_rotate_deep_det'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '5k_iters',
    'reset_iter',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'quick_snap',    
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fastest_logging', 
]




exps['det_trainer_big_freeze_vq_rotate_deep_det_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_norotate', # dataset
    '500k_iters',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_det',
    'reset_iter',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['det_trainer_big_vq_rotate_det_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_norotate', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    # 'pretrained_det',
    # 'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det',
    'fast_logging', 
]



exps['det_trainer_big_freeze_vq_rotate_deep_det_multiple_from2d'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_norotate', # dataset
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'use_2d_boxes',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


# ?no detach
exps['clevr_trainer_quantize_object_no_detach_50_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    # 'resume',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_50',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]




exps['det_trainer_big_freeze_vq_rotate_deep_det_multiple_tester'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '500k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'break_constraint',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]
# detector

# self improve detector
# iterative

exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_single'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'self_improve_iterate_41',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_single_quick'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'self_improve_iterate_41_quick',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_single_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'self_improve_iterate_41_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_single_debug'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'self_improve_iterate_41_debug',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

# multiple

exps['clevr_det_builder'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',     
    'break_constraint',
    'no_shuf',
    'lr4',
    'B2',
    'use_2d_boxes',
    'train_feat',
    'fastest_logging',
]



exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_debug'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr4',
    'B2',
    'maskout',    
    'self_improve_iterate_100_debug',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]



exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_norotate',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'maskout',
    'self_improve_iterate_100_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


# exps['det_trainer_big_freeze_vq_rotate_selfimproveIterate_deep_multiple_debug'] = [
#     'clevr_sta', # mode
#     'clevr_veggies_sta_data_big',
#     '500k_iters',
#     'reset_iter',
#     'eval_boxes',    
#     'pretrained_feat',
#     'pretrained_quantized',
#     'pretrained_det',
#     'break_constraint',
#     'frozen_feat',
#     'lr4',
#     'B1',
#     'filter_boxes_cs_100',
#     'self_improve_iterate_debug',
#     'train_feat',
#     'train_det_deep',
#     'fast_logging', 
# ]




groups['self_improve_iterate_41'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 1000',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 41',
                                        'exp_max_iters = 100','maxm_max_iters = 1000',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']

groups['self_improve_iterate_41_quick'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 500',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 41',
                                        'exp_max_iters = 100','maxm_max_iters = 500',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']                                        

groups['self_improve_iterate_41_big'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 2000',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 41',
                                        'exp_max_iters = 1000','maxm_max_iters = 6000',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']                                        

groups['self_improve_iterate_41_debug'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 20',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 41',
                                        'exp_max_iters = 10','maxm_max_iters = 10',\
                                        'exp_log_freq = 1','maxm_log_freq = 1']                                        


groups['self_improve_iterate_100'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 1000',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 100',
                                        'exp_max_iters = 100','maxm_max_iters = 500',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']


groups['self_improve_iterate_100_big'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 2000',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 100',
                                        'exp_max_iters = 1000','maxm_max_iters = 5000',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']

groups['self_improve_iterate_50_big'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 2000',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 50',
                                        'exp_max_iters = 1000','maxm_max_iters = 5000',\
                                        'exp_log_freq = 100','maxm_log_freq = 100']



groups['self_improve_iterate_100_debug'] = ['self_improve_iterate = True',\
                                        'det_pool_size = 10',\
                                        'vq_rotate = True',
                                        'cs_filter = True',
                                        'object_quantize_dictsize = 100',
                                        'exp_max_iters = 7','maxm_max_iters = 7',\
                                        'exp_log_freq = 1','maxm_log_freq = 1']


# iterative



# ONCE
exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_rg'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr4',
    'B2',
    'filter_boxes',
    'self_improve_once',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_rg_highb'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr4',
    'B4',
    'filter_boxes',
    'self_improve_once',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]


exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_rg_maskout'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr4',
    'B2',
    'filter_boxes',
    'self_improve_once_maskout',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]



exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_rg_high_neg'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr4',
    'B2',
    'filter_boxes',
    'self_improve_once',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
    'high_neg',
]
# ONCE

# SELF IMPROVE DETECTOR



# detector testing

exps['det_trainer_big_freeze_vq_rotate_deep_det_test_rg'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'break_constraint',    
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]


# 0.37
exps['det_trainer_big_freeze_vq_rotate_deep_det_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big', # dataset
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'break_constraint',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]



exps['det_trainer_big_freeze_vq_rotate_deep_det_test_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_norotate', # dataset
    '1k_iters',
    'eval_boxes',
    'no_shuf',
    'break_constraint',
    'pretrained_feat',
    'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]

exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_test_multiple_cs'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_norotate',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_100',
    'only_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]

exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_test_multiple_q_cs'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_norotate',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_100',
    'q_cs_vis',
    'train_feat',
    'train_det_deep',
    'fastest_logging', 
    'vis_clusters_hyp',
]


groups['vis_clusters_hyp'] = ['neg_cs_thresh = 0.55','pos_cs_thresh = 0.5']


groups['bigbird_multiview_data_ff_test'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'testset = "fft"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

groups['bigbird_multiview_data_gg_test'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'testset = "ggt"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]

########## bird specific stuff ends here ###############


########## CLEVR specific stuff STARTS here ###############



exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_test_rg'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes',
    'q_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]


exps['det_trainer_big_freeze_vq_rotate_filterboxes_deep_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big', # dataset
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes',
    'only_q_vis',    
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]



exps['det_trainer_big_freeze_vq_rotate_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',    
    # 'pretrained_pixor',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det',
    'fast_logging', 
]

# detector testing

































# feature learning

#worst models

exps['clevr_test_eval_recall_orient_tester'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_multiple_rotate', 
    '1000_iters',
    'no_shuf',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
]


exps['replica_test_eval_recall_orient_tester'] = [
    'replica_sta', 
    'replica_test_sta_data', 
    '2000_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'eval_recall_o_quicker_big_pool',
    'break_constraint',
]

exps['replica_test_eval_recall_orient_tester1'] = [
    'replica_sta', 
    'replica_test_sta_data', 
    '2000_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'eval_recall_o_quicker_big_pool1',
    'break_constraint',
]

exps['replica_test_eval_recall_orient_tester2'] = [
    'replica_sta', 
    'replica_test_sta_data', 
    '2000_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'eval_recall_o_quicker_big_pool2',
    'break_constraint',
]



exps['bigbird_test_eval_recall_orient_tester'] = [
    'bigbird_sta', 
    'bigbird_multiview_data_gg_test', 
    '1000_iters',
    'no_shuf',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
]

exps['carla_test_eval_recall_orient_tester'] = [
    'carla_det',
    'carla_test_sta_data',
    '1000_iters',
    'lr3',
    'no_shuf',
    'randomly_select_views',
    'train_feat',
    'B2',
    'no_bn',
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
]


#worst models

#best models

exps['carla_test_eval_recall_hard_vis'] = [
    'carla_det', 
    'carla_test_sta_data', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
    'do_hard_eval',
    'hard_vis',
]


exps['bigbird_test_eval_recall_hard_vis'] = [
    'bigbird_sta', 
    'bigbird_multiview_data_gg_test', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
    'do_hard_eval',
    'hard_vis',
    # 'debug',
]

exps['clevr_test_eval_recall_hard_vis'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_multiple_rotate', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'summ_all',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
    'do_hard_eval',
    'hard_vis',
    # 'debug',
]


exps['replica_test_eval_recall_hard_vis'] = [
    'replica_sta', 
    'replica_test_sta_data', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'summ_all',    
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
    'do_hard_eval',
    'hard_vis',
    # 'debug',
]

#best models

#feature learning



#feature compression vis parsings


# only for prototype visualization


exps['replica_trainer_quantize_object_no_detach_rotate_multiple_fixed_test'] = [
    'replica_sta', # mode
    'replica_test_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'obj_multiview',
    'quantize_object_no_detach_rotate_100',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',
    'obj_multiview'    ,
    'fastest1_logging',
    'S3',
]


# exps['bigbird_trainer_quantize_object_no_detach_rotate_multiple_fixed_test'] = [
#     'bigbird_sta', # mode
#     'bigbird_multiview_data_dd', # dataset
#     '200k_iters',
#     'lr3',
#     'B2',
#     'no_bn',        
#     'pretrained_feat',
#     'pretrained_view',
#     'pretrained_quantized',
#     'quantize_object_no_detach_rotate_100',
#     'orient_tensors_in_eval_recall_o',
#     'eval_recall_o',
#     'quick_snap',
#     'train_feat',
#     'train_view',
#     'break_constraint',    
#     'fast_logging',
# ]


exps['clevr_trainer_quantize_object_no_detach_rotate_multiple_fixed_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'obj_multiview',
    'quantize_object_no_detach_rotate_50',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',
    'obj_multiview'    ,
    'fastest1_logging',
    'S3',
]

exps['carla_trainer_quantize_object_no_detach_rotate_1_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_50',
    'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest1_logging',
    'S3',    
]


exps['clevr_trainer_quantize_object_no_detach_rotate_multiple_fixed_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'obj_multiview',
    'quantize_object_no_detach_50',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',
    'obj_multiview'    ,
    'fastest1_logging',
    'S3',
]

exps['carla_trainer_quantize_object_no_detach_rotate_1_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_50',
    'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest1_logging',
    'S3',    
]

######################################################### training 

exps['clevr_trainer_quantize_object_no_detach_rotate_multiple_fixed_train'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    # 'obj_multiview',
    'quantize_object_no_detach_rotate_50',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',
    'obj_multiview'    ,
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_multiple_fixed_train'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    # 'obj_multiview',
    'quantize_object_no_detach_50',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'obj_multiview',
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_1_train'] = [
    'carla_det', # mode
    'carla_sta_data_single', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'reset_iter',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_50',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'fastest_logging',
]

exps['carla_trainer_quantize_object_no_detach_1_train'] = [
    'carla_det', # mode
    'carla_sta_data_single', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'reset_iter',
    'fastest_logging',
    'pretrained_quantized',
    'quantize_object_no_detach_50',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    # 'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_1_train_test'] = [
    'carla_det', # mode
    'carla_sta_data_single_small_test', # dataset
    '10_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'reset_iter',
    'break_constraint',
    'quantize_object_no_detach_rotate_50',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'fastest_logging',
    'no_shuf',
    # 'debug',
]

exps['carla_trainer_quantize_object_no_detach_1_train_test'] = [
    'carla_det', # mode
    'carla_sta_data_single_small_test', # dataset
    '10_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'reset_iter',
    'break_constraint',
    'quantize_object_no_detach_50',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'fastest_logging',
    'no_shuf',
    # 'fast_logging',
]

exps['carla_trainer_view_small_test'] = [
    'carla_det', # mode
    'carla_sta_data_single_small_test', # dataset
    '10_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'reset_iter',
    'break_constraint',
    # 'pretrained_quantized',
    'quicker_snap',
    # 'quantize_object_no_detach_50',
    # 'obj_multiview',
    'eval_boxes',
    'train_feat',
    'train_view',
    'fastest_logging',
    'no_shuf',
    # 'fast_logging',
]


exps['carla_trainer_builder1'] = [
    'carla_det', # mode
    'carla_sta_data_single', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'eval_boxes',
    'fastest_logging',
]

###################################################################




# exps['carla_trainer_quantize_object_no_detach_rotate_1'] = [
#     'carla_det', # mode
#     'carla_sta_data', # dataset
#     '200k_iters',
#     'lr3',
#     'B2',
#     'no_bn',
#     'pretrained_feat',
#     'pretrained_view',
#     'pretrained_quantized',
#     'quantize_object_no_detach_rotate_100',
#     # 'obj_multiview',
#     'eval_boxes',
#     'train_feat',
#     'train_view',
#     'break_constraint',    
#     'fast_logging',
#     # 'S3',    
# ]

# scene parsing

exps['carla_trainer_scene_parsing'] = [
    'carla_det', # mode
    'carla_sta_data_mix_1_test', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_quantized',
    'frozen_feat',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'use_gt_centers',
    'quantize_object_no_detach_rotate_40',
    'rotate_combinations',
    # 'quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fastest1_logging',
]


exps['replica_trainer_scene_parsing'] = [
    'replica_sta', # mode
    'replica_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_quantized',
    'frozen_feat',
    'eval_boxes',
    'reset_iter',
    'low_res',
    # 'use_gt_centers',
    'quantize_object_no_detach_rotate_50',
    'rotate_combinations',
    # 'quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fastest1_logging',
]

exps['clevr_trainer_scene_parsing'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_quantized',
    'frozen_feat',
    'eval_boxes',
    'reset_iter',
    'use_gt_centers',
    'low_res',
    'rotate_combinations',
    'quantize_object_no_detach_rotate_50',
    # 'quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fastest1_logging',
]
#scene parsing


# detector













#feature compression



exps['clevr_test_eval_recall_rotation_check'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_multiple_rotate', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',]


exps['clevr_trainer_rgb_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    # 'train_occ', 
    'fast_logging',
]


exps['clevr_trainer_occ_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    # 'train_view',
    'train_occ', 
    'fast_logging',
]



exps['clevr_trainer_rgb_occ_multiple_clevr_orient'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ', 
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]



exps['clevr_trainer_rgb_occ_multiple_clevr_orient_viewContrast'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'do_moc',
    'eval_recall_o',   
    'orient_tensors_in_eval_recall_o',
    'pretrained_feat',
    'fast_logging',
]



exps['clevr_trainer_quantize_object_no_detach_rotate_multiple_fixed'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_100',
    # 'eval_recall_o',
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',
    'obj_multiview'    ,
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_multiple'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'eval_boxes',
    'quantize_object_no_detach_rotate_100',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]




# clevr viewpred
exps['clevr_trainer_quantize_object_no_detach_50'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_50',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_100'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_100',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_41_eval_quantize'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'eval_quantize',
    'quantize_object_no_detach_rotate_41',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_41'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_41',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['clevr_trainer_quantize_object_no_detach_rotate_100'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    'resume',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_100',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_100_init'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'resume',
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_100',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_500'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'resume',
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_500',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_1000'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'resume',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_1000',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_100_multi'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_100',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_5000'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    # 'eval_quantize',
    'quantize_object_no_detach_rotate_5000',
    'eval_boxes',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_offline_test_clusters'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'pretrained_feat',
    'train_feat',
    'train_view',
    'B2',
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'offline_cluster_100',
    'break_constraint',    
    'eval_boxes',
]


# clevr continual


#feature compresssion
exps['clevr_trainer_gen_examples_multiple_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'use_det_boxes',
    'create_example_dict_50',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_detboxes_frozen_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    'pretrained_quantized',
    'eval_boxes',
    'use_det_boxes',
    'reset_iter',
    'low_res',
    # 'quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes',
    'quantize_object_no_detach_rotate_50',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]



#detector

exps['clevr_det_trainer_big_freeze_vq_rotate_deep_det_multiple_from2d_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'eval_boxes',    
    # 'no_shuf',
    'pretrained_feat',
    'use_2d_boxes',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B2',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
]

exps['clevr_trainer_big_freeze_vq_rotate_deep_test_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    # 'filter_boxes_50',
    # 'q_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]




exps['clevr_trainer_big_freeze_vq_rotate_filterboxes_deep_test_multiple_only_cs_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_50',
    'only_cs_vis',
    'train_feat',
    'train_det_deep',
    'faster_logging', 
]



exps['clevr_trainer_big_freeze_vq_rotate_filterboxes_deep_test_multiple_q_cs_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate',
    '1k_iters',
    'eval_boxes',    
    'no_shuf',
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'no_shuf',
    'lr3',
    'B2',
    'filter_boxes_50',
    'q_cs_vis',
    'train_feat',
    'train_det_deep',
    'fastest_logging', 
    'new_distance_thresh',
]



exps['clevr_trainer_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout_hardneg_continual_1'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'high_neg',
    'self_improve_iterate_50_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
    'new_distance_thresh',
]

exps['clevr_trainer_freeze_vq_rotate_selfimproveIterate_deep_multiple_maskout_continual_2'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate',
    '500k_iters',
    'reset_iter',
    'eval_boxes',    
    'pretrained_feat',
    'pretrained_quantized',
    'pretrained_det',
    'break_constraint',
    'frozen_feat',
    'lr4',
    'B2',
    'maskout',
    'self_improve_iterate_50_big',
    'train_feat',
    'train_det_deep',
    'fast_logging', 
    'new_distance_thresh',
]


exps['clevr_det_trainer_big_freeze_deep_det_multiple_from2d_update_box_continual'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'eval_boxes',    
    'break_constraint',
    'no_shuf',
    'pretrained_feat',
    'pretrained_det',
    'use_2d_boxes',
    # 'pretrained_det',
    'frozen_feat',
    'lr3',
    'B1',
    'train_feat',
    'train_det_deep',
    'add_det_boxes',
    'fast_logging', 
]


#evaluation metric wrt gt

#feature learning
exps['clevr_test_eval_recall_orient_tester_continual'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_multiple_rotate', 
    '1000_iters',
    'no_shuf',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'orient_tensors_in_eval_recall_o',
    'break_constraint',
    'eval_recall_o_quicker_big_pool',
]


# feature compression
exps['clevr_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_2dboxes_frozen_continual_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50',
    # 'quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_multiview_continual_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',

    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes_multiview',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

groups['new_distance_thresh'] = ['dict_distance_thresh = 750']



#continual

# clevr vqvae

exps['clevr_trainer_gen_examples_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'create_example_dict_100',
    'eval_boxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_100_init_examples_hard_only_embed_test1'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_clevr_examples_only_dict_update1',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['clevr_trainer_quantize_object_no_detach_rotate_40_init_examples_hard_only_embed_test_multiple1'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_40_init_clevr_examples_only_dict_update1_multiple1',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

exps['clevr_trainer_quantize_object_no_detach_rotate_50_init_examples_hard_only_embed_2dboxes_frozen'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'use_2d_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


#no rotate frozen
exps['clevr_trainer_quantize_object_no_detach_50_init_examples_hard_only_embed_frozen'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_multiple_rotate', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'frozen_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_50_init_clevr_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]

# clevr vqvae





############################################################################################ MULTIVIEW ##########################################################################################################################################

exps['trainer_o_rgb_occ_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',        
    'train_occ',
    'eval_recall_o',
    'do_hard_eval',    
    'fast_logging',    
]






exps['trainer_emb_o_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_emb3D_o',
    'eval_recall_o',
    'train_feat',
    'fast_logging', 
    'do_hard_eval',
]


exps['trainer_emb_moc_o_rgb_occ_emb2d_big'] = [
    'clevr_sta', 
    'clevr_veggies_sta_data_big', 
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',        
    'train_occ',    
    'eval_recall_o',
    'fast_logging',
    'no_bn',
    'do_hard_eval',    
    'do_moc',
    'do_moc2d',    
    # 'debug',    
]


exps['trainer_emb_moc_o_big'] = [
    'clevr_sta', 
    'clevr_veggies_sta_data_big', 
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'eval_recall_o',
    'fast_logging',
    'no_bn',    
    'do_moc',
    'do_hard_eval',
]


####################################### MULTIVIEW ################################################################



########## CLEVR specific stuff ends here ###############




############## net configs ##############

groups['train_det_px'] = [
    'do_pixor_det = True',    
]

groups['train_det_px_calc_mean'] = [
    'do_pixor_det = True',   
    'calculate_mean = True',
]


groups['train_det_px_calc_std'] = [
    'do_pixor_det = True',    
    'calculate_std = True',
]


groups['train_det'] = [
    'do_det = True',    
]

groups['train_det_deep'] = [
    'do_det = True',
    'deeper_det = True'
]


groups['train_det_gt_px'] = [
    'do_gt_pixor_det = True',    
]

groups['online_cluster_20']= [
    'online_cluster = True',
    'object_quantize_dictsize = 20',
    'cluster_iters = 3000',
    'online_cluster_eval = True',    
    'initial_cluster_size = 20000',

]

groups['do_moc']= [
    'moc = True',
    'moc_qsize = 100000',
]

groups['offline_cluster_100']= [
    'offline_cluster = True',
    'offline_cluster_pool_size = 120',
    'object_quantize_dictsize = 100',
]


groups['offline_cluster_eval_kmeans'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_kmeans = True',
]

groups['offline_cluster_eval_vqvae'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
]



groups['offline_cluster_eval_vqvae_rotate'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
]


groups['offline_cluster_eval_vqvae_rotate_instances'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.1',
]

groups['offline_cluster_eval_vqvae_rotate_instances_all_vbig'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 1.0',    
    'num_rand_samps = 30',

]

groups['offline_cluster_eval_vqvae_rotate_instances_all_vsmall'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.001',
]


groups['offline_cluster_eval_vqvae_rotate_instances_all_vvsmall'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.0001',
]


groups['low_dict_size']= [
    'low_dict_size = True',
]
groups['hard_vis']= [
    'hard_vis = True',
]

groups['do_moc2d']= [
    'moc_2d = True',
]
groups['reset_iter'] = [
    'reset_iter = True',
]

groups['imgnet'] = [
    'imgnet = True',

]
groups['train_preocc'] = ['do_preocc = True']
groups['no_bn'] = ['no_bn = True']

groups['do_gen_pcds'] = [
  'GENERATE_PCD_BBOX = True'
]
groups['object_specific'] = [
    'do_object_specific = True',
    'do_eval_boxes = True',
]
groups['debug'] = [
    'do_debug = True',
    'moc_qsize = 1000',
    'offline_cluster_pool_size = 50',    
    'offline_cluster_eval_iters = 101',
    'eval_compute_freq = 1',    
    'log_freq_train = 1',
    'log_freq_val = 1',
    'log_freq_test = 1',
    'log_freq = 1',
]
groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
]


groups['quantize_vox_512'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 512',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_256'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 256',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_128'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 128',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]
groups['quantize_vox_64'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 64',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_32'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 32',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]



groups['quantize_vox_1024'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 512',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

# vq_rotate
groups['quantize_object'] = [
    'object_quantize = True',         
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
]

groups['quantize_object_no_detach_50'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
]

groups['quantize_object_no_detach_100'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
]



groups['quantize_object_no_detach_rotate_50'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
]

groups['quantize_object_no_detach_rotate_41'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
]


groups['quantize_object_no_detach_rotate_1000'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 1000',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ag_lt_ns_trainer_rgb_occ_multiple_clevr_orient2_cluster_centers_100.npy"',
]

groups['quantize_object_no_detach_rotate_500'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 500',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ag_lt_ns_trainer_rgb_occ_multiple_clevr_orient2_cluster_centers_100.npy"',
]

groups['quantize_object_no_detach_rotate_5000'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5000',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ag_lt_ns_trainer_rgb_occ_multiple_clevr_orient2_cluster_centers_100.npy"',
]




groups['quantize_object_no_detach_rotate_100'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ag_lt_ns_trainer_rgb_occ_multiple_clevr_orient2_cluster_centers_100.npy"',
]

groups['quantize_object_no_detach_rotate_40'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 40',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
]



groups['quantize_object_no_detach_rotate_100_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]
groups['quantize_object_no_detach_rotate_82_init_bigbird_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-3_F32_ggt_ns_bigbird_trainer_orient_viewContrast_moc_cluster_centers_Example_82.npy"',
    'object_quantize_dictsize = 82',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_50_init_bigbird_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-3_F32_ggt_ns_bigbird_trainer_orient_viewContrast_moc_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_50_init_bigbird_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-3_F32_ggt_ns_bigbird_trainer_orient_viewContrast_moc_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_50_init_replica_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_cct_ns_replica_trainer_hard_exp5_pret_moc_orient_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]




groups['quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_V_d32_c1_mct_ns_carla_trainer_quantize_object_no_detach_rotate_1_train_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_be_lt_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient_2dboxes_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_50_init_clevr_hard_examples_only_dict_update_2dboxes_multiview'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ag_lt_ns_trainer_rgb_occ_multiple_clevr_orient2_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_50_init_clevr_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_be_lt_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 50',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_52_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_52.npy"',
    'object_quantize_dictsize = 52',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_rotate_100_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_100_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]


groups['quantize_object_no_detach_rotate_70_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_70.npy"',
    'object_quantize_dictsize = 70',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_70_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_70.npy"',
    'object_quantize_dictsize = 70',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_40_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_40.npy"',
    'object_quantize_dictsize = 40',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_40_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_40.npy"',
    'object_quantize_dictsize = 40',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_10_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_10.npy"',
    'object_quantize_dictsize = 10',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_10_init_carla_hard_examples_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_Example_10.npy"',
    'object_quantize_dictsize = 10',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'only_embed = True',
]


groups['quantize_object_no_detach_rotate_100_init_clevr_examples_only_dict_update1'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_V_d32_c1_rgt_ns_clevr_trainer_quantize_object_no_detach_rotate_100_cluster_centers_Example_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_100_init_clevr_examples_only_dict_update1_multiple1'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_be_lt_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient_cluster_centers_Example_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]

groups['quantize_object_no_detach_rotate_40_init_clevr_examples_only_dict_update1_multiple1'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_be_lt_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient_cluster_centers_Example_40.npy"',
    'object_quantize_dictsize = 40',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]


groups['quantize_object_no_detach_rotate_100_init_carla_hard_only_dict_update'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'only_embed = True',
]



groups['quantize_object_no_detach_rotate_100_init_carla'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_bbt_ns_carla_trainer_rgb_occ_orient_low_res_cluster_centers_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
]


groups['quantize_object_no_detach_rotate_100_init_carla_hard'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-4_F32_bbt_ns_orient_lowres_cluster_centers_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
]


groups['quantize_object_no_detach_100_init_carla'] = [
    'object_quantize = True',
    'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_bbt_ns_carla_trainer_rgb_occ_orient_low_res_cluster_centers_100.npy"',
    'object_quantize_dictsize = 100',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    # 'vq_rotate = True',
]



groups['filter_boxes'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
]


groups['filter_boxes_cs'] = [
    'filter_boxes = True',
    'vq_rotate = True',
    'cs_filter = True',
]


groups['filter_boxes_100'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
    'object_quantize_dictsize = 100',
]

groups['filter_boxes_50'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
    'object_quantize_dictsize = 50',
]

groups['filter_boxes_cs_100'] = [
    'filter_boxes = True',
    'vq_rotate = True',
    'object_quantize_dictsize = 100',    
    'cs_filter = True',
]



groups['quantize_object_no_detach_rotate_instances_vsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.001'
]

groups['quantize_object_no_detach_rotate_instances_vvsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.0001'
]


groups['quantize_object_no_detach_rotate_instances_big'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.1'
]


groups['quantize_object_no_detach_rotate_instances_vbig'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 1.0'
]



groups['quantize_object_no_detach_rotate_instances_all_vbig'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 1.0',    
]


groups['quantize_object_no_detach_rotate_instances_all_vsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.001',
]

groups['quantize_object_no_detach'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'detach_background = False',
]
groups['quantize_object_no_detach_ema'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'detach_background = False',
    'object_ema = True',    
]

groups['quantize_object_no_detach_no_cluster'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
]

groups['quantize_object_no_detach_no_cluster_ema'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'object_ema = True',

    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
]

groups['quantize_object_init_cluster'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',

]

groups['quantize_object_high_coef'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'quantize_loss_coef = 5.0'
]

groups['train_feat_res'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_rt = True',
    'feat_do_flip = True',
    'feat_do_resnet = True',
]
groups['train_feat_sb'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_sb = True',
    'feat_do_resnet = True',
    'feat_do_flip = True',
    'feat_do_rt = True',
]
groups['train_occ_no_coeffs'] = [
    'do_occ = True',
    'occ_do_cheap = True',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_do_cheap = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_occ_less_smooth'] = [
    'do_occ = True',
    'occ_do_cheap = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 32',
    'render_l1_coeff = 1.0',
]
groups['train_view_accu_render'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
]
groups['train_view_accu_render_unps_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_unps = True',
    'view_accu_render_gt = True',
]
groups['train_view_accu_render_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_gt = True',
]

groups['do_hard_eval'] = [
    'hard_eval = True',
]

groups['train_occ_notcheap'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_do_cheap = False',
    'occ_smooth_coeff = 0.1',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_emb3D_o'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 100',
    'emb3D_o = True',
    'do_eval_boxes = True',

]
groups['train_emb3D_moc'] = [
    'moc = True',
]
groups['eval_boxes'] = [
  'do_eval_boxes = True'
]

groups['empty_table'] = [
  'do_empty = True'
]

groups['eval_recall_summ_o'] = [
  'eval_recall_summ_o = True'
]


groups['eval_recall'] = ['do_eval_recall = True']

groups['eval_recall_o'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 500',
]
groups['eval_recall_o_vbig_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'pool_size = 5000',
    'eval_compute_freq = 500',
]

groups['eval_recall_o_slow'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1000',
]

groups['eval_recall_o_quicker_small_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 100',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 1000',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool1'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 500',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool2'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 250',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]



groups['eval_recall_o_quicker_vbig_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 5000',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['debug_eval_recall_o'] = [
    'debug_eval_recall_o = True',
]

groups['orient_tensors_in_eval_recall_o'] = [
    'do_orientation = True',
]
groups['no_eval_recall'] = ['do_eval_recall = False']

############## datasets ##############

# dims for mem
# SIZE = 32
import socket
if "Alien"  in socket.gethostname():
    SIZE = 24

else:
    SIZE = 36

# SIZE = 72

# 56
Z = SIZE*4
Y = SIZE*4
X = SIZE*4

Z2 = Z//2
Y2 = Y//2
X2 = X//2

BOX_SIZE = 16

K = 3 # how many objects to consider
S = 2
H = 256
W = 256
N = 3
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

DATA_MOD = "aa"

# groups['clevr_veggies_sta_data'] = ['dataset_name = "clevr"',
#                              'H = %d' % H,
#                              'W = %d' % W,
#                              'trainset = "aat"',
#                              'dataset_list_dir = "/projects/katefgroup/datasets/clevr_veggies/npys"',
#                              'dataset_location = "/projects/katefgroup/datasets/clevr_veggies/npys"',
#                              'dataset_format = "npz"'
# ]



# st()
groups['clevr_veggies_sta_data'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "bgt"',
                             'valset = "bgv"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_sta_multiple_norotate'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "ag_lt"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_multiple_norotate'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "ag_lt"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

    


groups['clevr_veggies_sta_multiple_rotate'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'N = %d' % 6,
                             'root_keyword = "home"',
                             'trainset = "be_lt"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_multiple_rotate'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'N = %d' % 6,
                             'trainset = "be_lt"',
                             'testset = "be_lv"',
                             'root_keyword = "home"',                             
                             'dict_distance_thresh = 1500',                             
                             f'dataset_list_dir = "./clevr_veggies/npys"',
                             f'dataset_location = "./clevr_veggies/npys"',
                             f'root_dataset = "./clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_multiple_rotate_test'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'N = %d' % 6,
                             'testset = "be_lv"',
                             'root_keyword = "home"',                             
                             'dict_distance_thresh = 1500',                             
                             f'dataset_list_dir = "./clevr_veggies/npys"',
                             f'dataset_location = "./clevr_veggies/npys"',
                             f'root_dataset = "./clevr_veggies"',
                             'dataset_format = "npz"'
]




groups['clevr_veggies_sta_single_rotated_data'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "rat"',
                             'valset = "rav"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_test_sta_data'] = [
                             'dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "bgt"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_data_big'] = [
                             'dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "cgt"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_data_big_f'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'testset = "cg_Ft"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]



groups['clevr'] =  ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'trainset = "xft"',
                             f'dataset_location = "CHANGE_ME/clevr_vqa/output/npys"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_vqa/output/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_vqa/output"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_test_sta_data_big_cylinder'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'testset = "cg_cylindert"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_test_sta_data_big_rubberCylinder'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'testset = "cg_rubberCylindert"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_test_sta_data_big_shapes'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'testset = "cg_shapest"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_test_random_shear_scale'] = [
                             'dataset_name = "clevr"',                             
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "dit"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_train_random_shear_scale'] = [
                             'dataset_name = "clevr"',                             
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "det"',
                             # 'valset = "dev"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_test_sta_data_big_rot'] = [
                             'dataset_name = "clevr"',                             
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "rgt"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_sta_data_big'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'trainset = "cgt"',
                             # 'valset = "cgv"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_sta_data_big_rot'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,                             
                             'trainset = "rgt"',
                             # 'valset = "cgv"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]
groups['clevr_veggies_sta_data_big_rot_multiview'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'S = 4',                             
                             'trainset = "rgt"',
                             # 'valset = "cgv"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

# groups['clevr_veggies_sta_data_big_rot_multiple'] = ['dataset_name = "clevr"',
#                              'H = %d' % H,
#                              'W = %d' % W,
#                              'trainset = "ma_lt"',
#                              # 'valset = "cgv"',
#                              f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
#                              f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
#                              f'root_dataset = "CHANGE_ME/clevr_veggies"',
#                              'dataset_format = "npz"'
# ]


groups['clevr_veggies_sta_test_data_big_rot_multiple'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "ma_lt"',                             
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_sta_data_big_multiple'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "bgt"',                             
                             # 'valset = "cgv"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['real_sta_data'] = ['dataset_name = "real"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "abt"',
                             'H = 240',
                             'W = 320',
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',
                             'trainset = "abt"',
                             # 'valset = "aav"',
                             f'dataset_list_dir = "CHANGE_ME/real_data_matching/npys"',
                             f'dataset_location = "CHANGE_ME/real_data_matching/npys"',
                             f'root_dataset = "CHANGE_ME/real_data_matching"',
                             'dataset_format = "npz"'
]

groups['real_sta_data_test'] = ['dataset_name = "real"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'testset = "abt"',
                             'H = 240',
                             'W = 320',
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',
                             # 'valset = "aav"',
                             f'dataset_list_dir = "CHANGE_ME/real_data_matching/npys"',
                             f'dataset_location = "CHANGE_ME/real_data_matching/npys"',
                             f'root_dataset = "CHANGE_ME/real_data_matching"',
                             'dataset_format = "npz"'
]


# groups['clevr_veggies_sta_data'] = ['dataset_name = "clevr"',
#                              'H = %d' % H,
#                              'W = %d' % W,
#                              'trainset = "bgt"',
#                              'valset = "bgv"',
#                              f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
#                              f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
#                              f'root_dataset = "CHANGE_ME/clevr_veggies"',
#                              'dataset_format = "npz"'
# ]


############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    try:
        assert varname in globals()
        assert eq == '='
        assert type(s) is type('')
    except Exception as e:
        print(e)
        st()

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    if group not in groups:
      st()
      assert False
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)
import getpass
username = getpass.getuser()
import socket
hostname = socket.gethostname()

if 'compute' in hostname:
    if root_keyword == "katefgroup":
        root_location = "/projects/katefgroup/datasets/"
        dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
        dataset_location = dataset_location.replace("CHANGE_ME",root_location)
        root_dataset = root_dataset.replace("CHANGE_ME",root_location)
    elif root_keyword == "home":
        if 'shamit' in username:
            root_location = "/home/shamitl/datasets/"
        else:
            root_location = "/home/mprabhud/dataset/"
        dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
        dataset_location = dataset_location.replace("CHANGE_ME",root_location)
        root_dataset = root_dataset.replace("CHANGE_ME",root_location)
elif 'ip-' in hostname:
    if dataset_name == "carla" or dataset_name == "carla_mix" or dataset_name == "replica":
        root_location = "/projects/katefgroup/datasets/"
    else:
        root_location = "/home/ubuntu/datasets/"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)    

elif hostname.startswith('ip'):
    root_location = "/projects/katefgroup/datasets/"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)  
