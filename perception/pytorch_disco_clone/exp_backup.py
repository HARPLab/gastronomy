
exps['builder'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '100k10_iters',
    'lr0',
    'B2',
    # 'empty_table',
    'no_shuf',
    'train_feat',
    'eval_boxes',
    # 'train_occ',
    # 'eval_recall_o',
    # 'debug_eval_recall_o',
    'train_view',
    'train_emb3D',
    'fastest_logging',
    # 'do_gen_pcds',
    # 'debug'
]

exps['clevr_multiview_builder'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200_iters',
    'lr3',
    'B2',
    'reset_iter',
    # 'no_shuf',
    'train_feat',
    'train_occ',
    'train_view',
    'eval_boxes',
    'fast_logging',
]

exps['builder_rotate'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big', # dataset
    '100k_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat',
    'eval_boxes',
    'rotate_combinations',
    'fastest_logging',
]

exps['debug_eval_recall_with_orientation'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_single_rotated_data', # dataset
    '100k10_iters',
    'lr0',
    'B2',
    # 'empty_table',
    'no_shuf',
    'train_feat',
    'eval_boxes',
    'train_occ',
    'eval_recall_o',
    'debug_eval_recall_o',
    'orient_tensors_in_eval_recall_o',
    'train_view',
    # 'train_emb2D',
    'object_specific',
    'train_emb3D',
    'fastest_logging',
    # 'do_gen_pcds',
    'debug'
]

exps['debug_eval_recall_without_orientation'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_single_rotated_data', # dataset
    '100k10_iters',
    'lr0',
    'B2',
    # 'empty_table',
    'no_shuf',
    'train_feat',
    'eval_boxes',
    'train_occ',
    'eval_recall_o',
    'debug_eval_recall_o',
    'train_view',
    # 'train_emb2D',
    'object_specific',
    'train_emb3D',
    'fastest_logging',
    # 'do_gen_pcds',
    'debug'
]

#rgb view pred

exps['builder_rgb'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '3_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_view',
    'debug',
    # 'pretrained_feat',    
    'fastest_logging',
]



exps['builder_eval_recall'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    'no_shuf',    
    '20_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'debug',
    'eval_recall_o',
    'eval_recall_summ_o',
    'fastest_logging',
]















exps['test_eval_recall'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data', # dataset
    'no_shuf',    
    '5k_iters',
    'lr3',
    'B2',
    'train_feat',
    'eval_recall_o',
    'orient_tensors_in_eval_recall_o',
    'fastest_logging',
    'pretrained_feat',
]


exps['test_eval_recall_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big', # dataset
    'no_shuf',    
    '1k_iters',
    'lr3',
    'B2',
    'break_constraint',
    'train_feat',
    'eval_recall_o',
    'faster_logging',
    'pretrained_feat',
]

exps['test_eval_recall_random_shear_scale_test'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_random_shear_scale', # dataset
    'no_shuf',    
    '500k_iters',
    'lr3',
    'B2',
    'break_constraint',
    'train_feat',
    'eval_recall_o',
    'faster_logging',
    'pretrained_feat',
]



exps['test_eval_recall_big_rot'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    'no_shuf',    
    '5k_iters',
    'lr3',
    'B2',
    'train_feat',
    'break_constraint',
    'eval_recall_o',
    'eval_recall_summ_o',    
    'faster_logging',
    'pretrained_feat',
]

exps['test_eval_recall_big_rot_orient'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    'no_shuf',    
    '5k_iters',
    'lr3',
    'break_constraint',
    'B2',
    'train_feat',
    'eval_recall_o',
    'orient_tensors_in_eval_recall_o',
    'eval_recall_summ_o',
    'faster_logging',
    'pretrained_feat',
]


exps['test_viewpred_boxes'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data', # dataset
    'no_shuf',    
    '5k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_view',
    'debug',
    # 'eval_recall_o',
    'eval_boxes',
    'fastest_logging',
    'pretrained_feat',
    'pretrained_view',
]






























exps['trainer_rgb'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    # 'eval_recall_o',
    'resume',
    'fast_logging',
]


exps['trainer_rgb_occ_emb3d'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ',
    'train_emb3D',
    # 'eval_recall_o',
    'fast_logging',
]


exps['trainer_rgb_occ_emb3d_emb2d'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ',
    'train_emb3D',
    'train_emb2D',
    # 'eval_recall_o',
    'resume',
    'fast_logging',
]














exps['trainer_rgb_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'pretrained_feat',
    'pretrained_view',
    'fast_logging',
]

exps['trainer_rgb_occ_big_no_bn'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ', 
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'no_bn',
    'fast_logging',
]


exps['trainer_rgb_occ_big_no_bn_real'] = [
    'clevr_sta', # mode
    'real_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ', 
    # 'debug',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'no_bn',
    'fast_logging',
]


exps['trainer_rgb_occ_big_no_bn_imgnet'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ', 
    'imgnet',
    'no_bn',
    'fast_logging',
]



exps['trainer_rgb_occ_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ', 
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]


exps['trainer_rgb_occ_big_clevr'] = [
    'clevr_sta', # mode
    'clevr', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'train_feat',
    'train_view',
    'train_occ', 
    'eval_recall_o',    
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
]

exps['trainer_rgb_occ_emb3d_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ',
    'train_emb3D',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',    
    'fast_logging',

]



exps['trainer_rgb_occ_emb3d_emb2d_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ',
    'train_emb2D',
    'train_emb3D',
    'fast_logging',
]




exps['trainer_rgb_occ_shear'] = [
    'clevr_sta', # mode
    'clevr_veggies_train_random_shear_scale', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',
    'train_occ', 
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',
    'fast_logging',
    # 'break_constraint',
    # 'eval_recall_o',
]










# ml only


exps['builder_emb'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '10_iters',
    'lr0',
    'B1',
    # 'empty_table',
    'no_shuf',
    'train_feat',
    'eval_boxes',
    'eval_recall',
    # 'train_emb2D',
    'train_emb3D',
    'fastest_logging',
    # 'do_gen_pcds',
]


exps['trainer_emb'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    # 'empty_table',
    'train_feat',
    # 'train_emb2D',
    'train_emb3D',
    'faster_logging',
    'eval_recall',
    # 'do_gen_pcds',
]





# shear
exps['trainer_emb_o_rgb_occ_emb2d'] = [
    'clevr_sta', # mode
    'clevr_veggies_train_random_shear_scale', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_emb3D_o',
    'train_view',        
    'train_occ',
    'train_emb2D',    
    'eval_recall_o',
    'train_feat',
    'do_hard_eval',    
    'fast_logging',    
]


exps['trainer_emb_o'] = [
    'clevr_sta', # mode
    'clevr_veggies_train_random_shear_scale', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_emb3D_o',
    'eval_recall_o',
    'train_feat',
    'fast_logging', 
    'do_hard_eval',
]



exps['trainer_emb_moc_o'] = [
    'clevr_sta', 
    'clevr_veggies_train_random_shear_scale', 
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


exps['trainer_emb_o_rgb_occ'] = [
    'clevr_sta', # mode
    'clevr_veggies_train_random_shear_scale', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_emb3D_o',
    'train_view',        
    'train_occ',
    'eval_recall_o',
    'train_feat',
    'do_hard_eval',    
    'fast_logging',    
]



exps['trainer_emb_moc_o_rgb_occ'] = [
    'clevr_sta', 
    'clevr_veggies_train_random_shear_scale', 
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
]


exps['trainer_emb_moc_o_rgb_occ_emb2d_pret'] = [
    'clevr_sta', 
    'clevr_veggies_train_random_shear_scale', 
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_view',        
    'train_occ',    
    'eval_recall_o',
    'pretrained_feat',
    'pretrained_occ',
    'pretrained_view',    
    'reset_iter',
    'fast_logging',
    'no_bn',
    'do_hard_eval',    
    'do_moc',
    'do_moc2d', 
]




exps['trainer_emb_moc_o_rgb_occ_emb2d'] = [
    'clevr_sta', 
    'clevr_veggies_train_random_shear_scale', 
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


exps['trainer_o_rgb_occ'] = [
    'clevr_sta', # mode
    'clevr_veggies_train_random_shear_scale', # dataset
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





# 0.863
# 0.883


exps['offline_test_clusters_big_eval_kmeans'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    'pretrained_view',
    'offline_cluster_eval_kmeans',
    'break_constraint',    
    'eval_boxes',
]
# 0.7647840297795923
# 0.6341371797066083

exps['offline_test_clusters_big_eval_vqvae'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rot', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    # 'pretrained_view',
    'offline_cluster_eval_vqvae',
    'break_constraint',    
    'eval_boxes',
]

exps['offline_test_clusters_big_eval_vqvae_rotate'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rot', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    # 'pretrained_view',
    'offline_cluster_eval_vqvae_rotate',
    'break_constraint',    
    'eval_boxes',
]


exps['offline_test_clusters_big_eval_vqvae_rotate_instances_all_vbig'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rot', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'offline_cluster_eval_vqvae_rotate_instances_all_vbig',
    'break_constraint',   
    'eval_boxes',
]



exps['offline_test_clusters_big_eval_vqvae_rotate_instances_all_vsmall'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rot', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'offline_cluster_eval_vqvae_rotate_instances_all_vsmall',
    'break_constraint',   
    'eval_boxes',
]


exps['offline_test_clusters_big_eval_vqvae_rotate_instances_vvsmall'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rot', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',    
    'B2',    
    'no_bn',    
    'fast_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'offline_cluster_eval_vqvae_rotate_instances_all_vvsmall',
    'break_constraint',   
    'eval_boxes',
]



exps['offline_test_clusters_big_debug'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'offline_cluster',
    'break_constraint',    
    'eval_boxes',
    'debug',
]


exps['trainer_quantize_voxels_32'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'quantize_vox_32',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]




exps['trainer_quantize_voxels_64'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'quantize_vox_64',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',
    'fast_logging',
]


exps['trainer_quantize_voxels_128'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'quantize_vox_128',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]




exps['trainer_quantize_voxels_256'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',
    'pretrained_feat',
    'pretrained_view',
    'quantize_vox_256',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['trainer_quantize_voxels_512'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_multiple', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',
    'resume',
    'quantize_vox_512',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]





















exps['trainer_quantize_object'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'quantize_object',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]





exps['trainer_quantize_object_no_detach'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'quantize_object_no_detach',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]







exps['trainer_quantize_object_no_detach_rotate'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',    
    'quantize_object_no_detach_rotate',
    'eval_recall_o',
    'eval_quantize',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]






exps['trainer_quantize_object_no_detach_rotate_real'] = [
    'clevr_sta', # mode
    'real_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',    
    'quantize_object_no_detach_rotate',
    'eval_recall_o',
    'quick_snap',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
    # 'debug'
]







exps['trainer_quantize_object_no_detach_rotate_combvis'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'quantize_object_no_detach_rotate',
    'rotate_combinations',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]













exps['trainer_quantize_object_no_detach_rotate_combvis_parse_scene'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_test_data_big_rot_multiple', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'parse_scene',
    'quantize_object_no_detach_rotate',
    'rotate_combinations',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]


exps['trainer_quantize_object_no_detach_rotate_combvis_parse_scene2'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot_multiview', # dataset
    '200k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'parse_scene',
    'quantize_object_no_detach_rotate',
    'rotate_combinations',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]

exps['trainer_quantize_object_no_detach_rotate_combvis_parse_scene2_real'] = [
    'clevr_sta', # mode
    'real_sta_data_test', # dataset
    '500k_iters',
    'lr3',
    'B1',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'pretrained_quantized',
    'parse_scene',
    'quantize_object_no_detach_rotate',
    'rotate_combinations',
    # 'eval_recall_o',  
    'eval_boxes',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fastest_logging',
]



















exps['trainer_quantize_object_no_detach_rotate_init'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',    
    # 'pretrained_quantized',
    'quantize_object_no_detach_rotate',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_no_detach_rotate_init'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',    
    # 'pretrained_quantized',
    'quantize_object_no_detach_rotate',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]













exps['trainer_quantize_object_no_detach_rotate_instances_vbig'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',    
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_instances_vbig',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['trainer_quantize_object_no_detach_rotate_instances_big'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',    
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_instances_big',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_no_detach_rotate_instances_vsmall'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',    
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_instances_vsmall',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_no_detach_rotate_instances_vvsmall'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',    
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_instances_vvsmall',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]




































exps['trainer_quantize_object_no_detach_rotate_instances_all_vsmall'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',    
    # 'pretrained_quantized',
    'resume',
    'quantize_object_no_detach_rotate_instances_all_vsmall',
    'eval_recall_o',
    'low_dict_size',    
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['trainer_quantize_object_no_detach_rotate_instances_all_vbig'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',    
    'pretrained_quantized',
    'quantize_object_no_detach_rotate_instances_all_vbig',
    'eval_recall_o',
    'low_dict_size',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

















exps['trainer_quantize_object_no_detach_ema'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'quantize_object_no_detach_ema',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_high_coef'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'quantize_object_high_coef',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_init-model'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'quantize_object_no_detach',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_init-model_init-cluster'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'quantize_object_no_detach_no_cluster',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['trainer_quantize_object_init-model_init-cluster_ema'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'quantize_object_no_detach_no_cluster_ema',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',
    'fast_logging',
]

exps['trainer_quantize_object_init-cluster'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'quantize_object_init_cluster',
    'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]






















exps['test_eval_recall'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',
    'train_occ',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_occ',
    'break_constraint',    
    'eval_recall_o_quicker',
    # 'do_hard_eval',
    # 'hard_vis',
]





exps['test_eval_recall_hard_vis'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'break_constraint',
    'eval_recall_o_quicker',
    'do_hard_eval',
    'hard_vis',
    # 'debug',
]


exps['test_eval_recall_hard'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'train_view',
    'train_occ',
    'B2',    
    'no_bn',
    'faster_logging',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_occ',
    'break_constraint',    
    'eval_recall_o_quicker',
    'do_hard_eval',
    # 'hard_vis',
]







# detector

exps['det_trainer_px_builder'] = [
    'clevr_sta', # mode
    'clevr_veggies_test_sta_data_big', # dataset
    '200k_iters',
    'eval_boxes',    
    'lr3',
    'B2',
    'train_feat',
    'train_det_px',
    'fastest_logging', 
]

exps['det_trainer_px'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big', # dataset
    '200k_iters',
    'eval_boxes',    
    'no_shuf',
    'lr3',
    'B2',
    'train_feat',
    'train_det_px',
    'fast_logging', 
]

#backbone
exps['det_trainer_gt_px'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_data_big_rot', # dataset
    '200k_iters',
    'eval_boxes',    
    'lr3',
    'B2',
    'train_feat',
    'train_det_gt_px',
    'fastest_logging', 
]






















exps['online_test_clusters_big_eval_20'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_cylinder', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',
    'faster_logging',
    'pretrained_feat',
    'online_cluster_20',
    'break_constraint',
    'eval_boxes',
]


exps['online_test_clusters_big_eval_20_shape'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_shapes', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'online_cluster_20',
    'break_constraint',    
    'eval_boxes',
]


exps['online_test_clusters_big_eval_20_rubberCylinder'] = [
    'clevr_sta', 
    'clevr_veggies_test_sta_data_big_rubberCylinder', 
    '200k_iters',
    'lr3',    
    'train_feat',
    'B2',    
    'no_bn',    
    'faster_logging',
    'pretrained_feat',
    'online_cluster_20',
    'break_constraint',    
    'eval_boxes',
]


















#vqvae


exps['carla_trainer_quantize_object_no_detach_rotate_5000'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_5000',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_1000'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_1000',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_500'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_500',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_100_init'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['carla_trainer_quantize_object_no_detach_rotate_100_init_kmeans'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_100_init_kmeans_test_hard'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    # 'shuf',
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_100_hard_only_embed'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]
exps['carla_trainer_quantize_object_no_detach_rotate_100_init_examples_hard_only_embed_lr5'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr5',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_100_init_examples_hard_only_embed_lr4'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard_examples_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_rotate_100_init_kmeans_hard_only_embed'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla_hard_only_dict_update',
    'train_feat',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_100_init_kmeans_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100_init_carla',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_100_init_kmeans_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr4',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_100_init_carla',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_100_no_rotate_test'] = [
    'carla_det', # mode
    'carla_test_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    'pretrained_feat',
    'pretrained_view',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_100',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['carla_trainer_quantize_object_no_detach_rotate_occ_100'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    # 'pretrained_occ',
    # 'pretrained_quantized',
    'resume',
    'eval_boxes',
    'reset_iter',
    'low_res',
    'quantize_object_no_detach_rotate_100',
    'train_feat',
    'train_view',
    'train_occ',
    'break_constraint',    
    'fast_logging',
]


exps['carla_trainer_quantize_object_no_detach_rotate_50'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'resume',
    # 'pretrained_quantized',
    'eval_boxes',
    'reset_iter',
    # 'pretrained_quantized',
    'quantize_object_no_detach_rotate_50',
    # 'orient_tensors_in_eval_recall_o',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_100'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'resume',
    'reset_iter',
    # 'pretrained_quantized',
    'eval_boxes',
    'quantize_object_no_detach_100',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]



exps['carla_trainer_quantize_object_no_detach_50'] = [
    'carla_det', # mode
    'carla_sta_data', # dataset
    '500k_iters',
    'lr3',
    'B2',
    'no_bn',        
    # 'pretrained_feat',
    # 'pretrained_view',
    'resume',
    # 'pretrained_quantized',
    'eval_boxes',
    # 'pretrained_quantized',
    'quantize_object_no_detach_50',
    # 'orient_tensors_in_eval_recall_o',
    # 'eval_recall_o',
    'train_feat',
    'train_view',
    'break_constraint',    
    'fast_logging',
]

exps['carla_offline_test_clusters'] = [
    'carla_det', 
    'carla_test_sta_data_fc', 
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
