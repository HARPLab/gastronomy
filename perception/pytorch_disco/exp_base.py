import pretrained_nets_carla as pret_clevr
# import pretrained_nets_clevr as pret_clevr

exps = {}
groups = {}
group_parents = {}

############## preprocessing/shuffling ##############

############## modes ##############

groups['zoom'] = ['do_zoom = True']
groups['bigbird_sta'] = ['do_clevr_sta = True'] # Hack to run bigbird dataset with clevr code
groups['carla_det'] = ['do_clevr_sta = True'] # Hack to run carla dataset with clevr code
groups['replica_sta'] = ['do_clevr_sta = True'] # Hack to run carla dataset with clevr code

groups['carla_mot'] = ['do_carla_mot = True']
groups['carla_sta'] = ['do_carla_sta = True']
groups['carla_flo'] = ['do_carla_flo = True']
groups['clevr_sta'] = ['do_clevr_sta = True']
groups['nel_sta'] = ['do_nel_sta = True']
groups['carla_obj'] = ['do_carla_obj = True']
groups['mujoco_offline'] = ['do_mujoco_offline = True']

############## extras ##############
groups['rotate_combinations'] = ['gt_rotate_combinations = True']
groups['use_gt_centers'] = ['use_gt_centers = True']
groups['add_det_boxes'] = ['add_det_boxes = True']



groups['vis_clusters'] = ['vis_clusters = True']

groups['randomly_select_views'] = ['randomly_select_views = True']

groups['obj_multiview'] = ['obj_multiview = True']
groups['S3'] = ['S = 3']


groups['decay_lr'] = ['do_decay_lr = True']
groups['clip_grad'] = ['do_clip_grad = True']
groups['quick_snap'] = ['snap_freq = 500']
groups['quicker_snap'] = ['snap_freq = 200']
groups['quickest_snap'] = ['snap_freq = 5']
groups['superquick_snap'] = ['snap_freq = 1']

groups['use_det_boxes'] = ['use_det_boxes = True']
groups['summ_all'] = ['summ_all = True']




groups['create_example_dict_100'] = ['create_example_dict = True','object_quantize_dictsize = 100']
groups['create_example_dict_82'] = ['create_example_dict = True','object_quantize_dictsize = 82']
groups['create_example_dict_70'] = ['create_example_dict = True','object_quantize_dictsize = 70']
groups['create_example_dict_52'] = ['create_example_dict = True','object_quantize_dictsize = 52']
groups['create_example_dict_50'] = ['create_example_dict = True','object_quantize_dictsize = 50']
groups['create_example_dict_40'] = ['create_example_dict = True','object_quantize_dictsize = 40']
groups['create_example_dict_10'] = ['create_example_dict = True','object_quantize_dictsize = 10']
groups['only_embed'] = ['only_embed = True']


groups['profile_time'] = ['profile_time = True']
groups['low_res'] = ['low_res = True']
groups['cpu'] = ['cpu = True']

groups['eval_quantize'] = ['eval_quantize = True']

groups['use_2d_boxes'] = ['use_2d_boxes = True']


groups['no_shuf'] = ['shuffle_train = False',
                     'shuffle_val = False',
                     'shuffle_test = False',
]



groups['shuf'] = ['shuffle_train = True',
                  'shuffle_val = True',
                    'shuffle_test = True',
]

groups['no_backprop'] = ['backprop_on_train = False',
                         'backprop_on_val = False',
                         'backprop_on_test = False',
]
groups['gt_ego'] = ['ego_use_gt = True']
groups['precomputed_ego'] = ['ego_use_precomputed = True']
groups['aug3D'] = ['do_aug3D = True']
groups['aug2D'] = ['do_aug2D = True']

groups['sparsify_pointcloud_10k'] = ['do_sparsify_pointcloud = 10000']
groups['sparsify_pointcloud_1k'] = ['do_sparsify_pointcloud = 1000']

groups['horz_flip'] = ['do_horz_flip = True']
groups['synth_rt'] = ['do_synth_rt = True']
groups['piecewise_rt'] = ['do_piecewise_rt = True']
groups['synth_nomotion'] = ['do_synth_nomotion = True']
groups['aug_color'] = ['do_aug_color = True']
groups['break_constraint'] = ['break_constraint = True']


# groups['eval'] = ['do_eval = True']
groups['random_noise'] = ['random_noise = True']
groups['eval_map'] = ['do_eval_map = True']
groups['save_embs'] = ['do_save_embs = True']
groups['save_ego'] = ['do_save_ego = True']
groups['save_vis'] = ['do_save_vis = True']

groups['profile'] = ['do_profile = True',
                     'log_freq_train = 100000000',
                     'log_freq_val = 100000000',
                     'log_freq_test = 100000000',
                     'max_iters = 20']

groups['B1'] = ['B = 1']
groups['B2'] = ['B = 2']
groups['B4'] = ['B = 4']
groups['B8'] = ['B = 8']
groups['B10'] = ['B = 10']
groups['B16'] = ['B = 16']
groups['B32'] = ['B = 32']
groups['B64'] = ['B = 64']
groups['B128'] = ['B = 128']
groups['lr0'] = ['lr = 0.0']
groups['lr2'] = ['lr = 1e-2']
groups['lr3'] = ['lr = 1e-3']
groups['2lr4'] = ['lr = 2e-4']
groups['5lr4'] = ['lr = 5e-4']
groups['lr4'] = ['lr = 1e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['lr9'] = ['lr = 1e-9']
groups['lr12'] = ['lr = 1e-12']
groups['1_iters'] = ['max_iters = 1']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['6_iters'] = ['max_iters = 6']
groups['9_iters'] = ['max_iters = 9']
groups['21_iters'] = ['max_iters = 21']
groups['10_iters'] = ['max_iters = 10']
groups['20_iters'] = ['max_iters = 20']
groups['25_iters'] = ['max_iters = 25']
groups['30_iters'] = ['max_iters = 30']
groups['50_iters'] = ['max_iters = 50']
groups['100_iters'] = ['max_iters = 100']
groups['150_iters'] = ['max_iters = 150']
groups['200_iters'] = ['max_iters = 200']
groups['250_iters'] = ['max_iters = 250']
groups['300_iters'] = ['max_iters = 300']
groups['397_iters'] = ['max_iters = 397']
groups['400_iters'] = ['max_iters = 400']
groups['447_iters'] = ['max_iters = 447']
groups['500_iters'] = ['max_iters = 500']
groups['850_iters'] = ['max_iters = 850']
groups['1000_iters'] = ['max_iters = 1000']
groups['2000_iters'] = ['max_iters = 2000']
groups['2445_iters'] = ['max_iters = 2445']
groups['3000_iters'] = ['max_iters = 3000']
groups['4000_iters'] = ['max_iters = 4000']
groups['4433_iters'] = ['max_iters = 4433']
groups['5000_iters'] = ['max_iters = 5000']
groups['10000_iters'] = ['max_iters = 10000']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['30k_iters'] = ['max_iters = 30000']
groups['40k_iters'] = ['max_iters = 40000']
groups['50k_iters'] = ['max_iters = 50000']
groups['60k_iters'] = ['max_iters = 60000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['100k10_iters'] = ['max_iters = 100010']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['400k_iters'] = ['max_iters = 400000']
groups['500k_iters'] = ['max_iters = 500000']

groups['resume'] = ['do_resume = True']
# groups['total_init'] = ['total_init = pret_carl.total_init']
groups['reset_iter'] = ['reset_iter = True']

groups['fastest_logging'] = ['log_freq_train = 1',
                             'log_freq_val = 1',
                             'log_freq_test = 1',
                             'log_freq = 1']

groups['fastest1_logging'] = ['log_freq_train = 10',
                             'log_freq_val = 10',
                             'log_freq_test = 10',
                             'log_freq = 10']

groups['faster_logging'] = ['log_freq_train = 50',
                            'log_freq_val = 50',
                            'log_freq_test = 50',
                            'log_freq = 50',
]
groups['fast_logging'] = ['log_freq_train = 250',
                          'log_freq_val = 250',
                          'log_freq_test = 250',
                          'log_freq = 250',
]
groups['slow_logging'] = ['log_freq_train = 500',
                          'log_freq_val = 500',
                          'log_freq_test = 500',
                          'log_freq = 500',                          
]
groups['slower_logging'] = ['log_freq_train = 1000',
                            'log_freq_val = 1000',
                            'log_freq_test = 1000',
                            'log_freq = 1000',                          

]
groups['no_logging'] = ['log_freq_train = 100000000000',
                        'log_freq_val = 100000000000',
                        'log_freq_test = 100000000000',
                        'log_freq = 100000000000',                        
]



groups['fastest_logging_group'] = ['log_freq = 1',

]
groups['fastest2_logging_group'] = ['log_freq = 20',
]
groups['faster_logging_group'] = ['log_freq = 50',
]
groups['fast_logging_group'] = ['log_freq = 100',
]
groups['slow_logging_group'] = ['log_freq = 500',                          
]
groups['slower_logging_group'] = ['log_freq = 1000',
]
groups['no_logging_group'] = ['log_freq = 100000000000',                        
]
# ############## pretrained nets ##############
groups['pretrained_feat'] = ['do_feat = True',
                             'feat_init = "' + pret_clevr.feat_init + '"',
                             # 'feat_do_vae = ' + str(pret_clevr.feat_do_vae),
                             # 'feat_dim = %d' % pret_clevr.feat_dim,
]
groups['pretrained_view'] = ['do_view = True',
                             'view_init = "' + pret_clevr.view_init + '"',
                             # 'view_depth = %d' %  pret_clevr.view_depth,
                             # 'view_use_halftanh = ' + str(pret_clevr.view_use_halftanh),
                             # 'view_pred_embs = ' + str(pret_clevr.view_pred_embs),
                             # 'view_pred_rgb = ' + str(pret_clevr.view_pred_rgb),
]
groups['pretrained_det'] = ['det_init = "' + pret_clevr.det_init + '"',
                             # 'view_depth = %d' %  pret_clevr.view_depth,
                             # 'view_use_halftanh = ' + str(pret_clevr.view_use_halftanh),
                             # 'view_pred_embs = ' + str(pret_clevr.view_pred_embs),
                             # 'view_pred_rgb = ' + str(pret_clevr.view_pred_rgb),
]

groups['pretrained_quantized'] = ['quant_init = "' + pret_clevr.quant_init + '"',
]


groups['pretrained_pixor'] = ['pixor_init = "' + pret_clevr.pixor_init + '"',
]

groups['pretrained_flow'] = ['do_flow = True',
                             'flow_init = "' + pret_clevr.flow_init + '"',
]
groups['pretrained_tow'] = ['do_tow = True',
                            'tow_init = "' + pret_clevr.tow_init + '"',
]
groups['pretrained_emb2D'] = ['do_emb2D = True',
                              'emb2D_init = "' + pret_clevr.emb2D_init + '"',
                              # 'emb_dim = %d' % pret_clevr.emb_dim,
]
groups['pretrained_occ'] = ['do_occ = True',
                            'occ_init = "' + pret_clevr.occ_init + '"',
]
groups['pretrained_preocc'] = [
    'do_preocc = True',
    'preocc_init = "' + pret_clevr.preocc_init + '"',
]
groups['pretrained_vis'] = ['do_vis = True',
                            'vis_init = "' + pret_clevr.vis_init + '"',
                            # 'occ_cheap = ' + str(pret_clevr.occ_cheap),
]

groups['only_cs_vis'] = ['only_cs_vis = True','cs_filter = True']
groups['replace_with_cs'] = ['replace_with_cs = True']

groups['only_q_vis'] = ['only_q_vis = True','cs_filter = False']
groups['q_cs_vis'] = ['only_q_vis = True','cs_filter = True']



groups['self_improve_once'] = ['self_improve_once = True']
groups['self_improve_once_maskout'] = ['self_improve_once = True','maskout = True']

groups['maskout'] = ['maskout = True']

groups['high_neg'] = ['alpha_pos = 1.0','beta_neg = 2.0']


groups['fast_orient'] = ['fast_orient = True']

groups['frozen_feat'] = ['do_freeze_feat = True', 'do_feat = True']
groups['frozen_view'] = ['do_freeze_view = True', 'do_view = True']
groups['frozen_vis'] = ['do_freeze_vis = True', 'do_vis = True']
groups['frozen_flow'] = ['do_freeze_flow = True', 'do_flow = True']
groups['frozen_emb2D'] = ['do_freeze_emb2D = True', 'do_emb2D = True']
groups['frozen_occ'] = ['do_freeze_occ = True', 'do_occ = True']
# groups['frozen_ego'] = ['do_freeze_ego = True', 'do_ego = True']
# groups['frozen_inp'] = ['do_freeze_inp = True', 'do_inp = True']
