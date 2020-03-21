from exp_base import *
# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE

############## choose an experiment ##############

# current = 'builder'
current = 'trainer_basic'
current = 'trainer_render'
# current = 'trainer_sb'
# current = 'builder'
# current = 'res_trainer_accu_render'
# current = 'res_trainer'
# current = 'vis_trainer'
# current = 'occvis_trainer'
# current = 'emb_trainer_sb'
# current = 'emb_trainer'
# current = 'emb_trainer_kitti'
# current = 'tow_trainer'

# (NO MODS HERE)

############## define experiments ##############

exps['builder'] = [
    'carla_sta', # mode
    'carla_sta10_data', # dataset
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'fastest_logging',
]
exps['trainer_basic'] = [
    'carla_sta', # mode
    'carla_stat_stav_data', # dataset
    '200k_iters',
    'lr3',
    'B4',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'faster_logging',
]
exps['trainer_accu'] = [
    'carla_sta', # mode
    'carla_stat_stav_data', # dataset
    '200k_iters',
    # 'carla_stat_stav_data', # dataset
    # '200k_iters',
    'lr3',
    'B4',
    'train_feat',
    'train_occ',
    'train_view_accu_render_unps_gt',
    'train_emb2D',
    'train_emb3D',
    'faster_logging',
]
exps['trainer_render'] = [
    'carla_sta', # mode
    'carla_sta1_data', # dataset
    '10k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_occ_no_coeffs',
    'train_render',
    'faster_logging',
]
exps['res_trainer'] = [
    'carla_sta', # mode
    'carla_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B4',
    'train_feat_res',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'faster_logging',
    'resume'
]
exps['res_trainer_accu_render'] = [
    'carla_sta', # mode
    'carla_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B4',
    'train_feat_res',
    'train_occ',
    'train_view_accu_render_unps_gt',
    'faster_logging',
    'resume'
]
exps['emb_trainer'] = [
    'carla_sta', # mode
    'carla_static_data', # dataset
    '300k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_occ',
    'train_emb_view',
    'faster_logging',
]
exps['trainer_sb'] = ['carla_sta', # mode
                       'carla_sta_data', # dataset
                       '300k_iters',
                       'lr3',
                       'B4',
                       'train_feat_sb',
                       'train_occ_notcheap',
                       'train_view',
                       'train_emb2D',
                       'train_emb3D',
                       'faster_logging',
                       #'fast_logging',
                       #'fastest_logging',
]
exps['emb_trainer_noocc'] = ['carla_sta', # mode
                             'carla_static_data', # dataset
                             '300k_iters',
                             'lr3',
                             'B2',
                             'train_feat',
                             'train_emb_view',
                             'resume',
                             'slow_logging',
]
exps['emb_trainer_kitti'] = ['carla_sta', # mode
                             'kitti_static_data', # dataset
                             '300k_iters',
                             'lr3',
                             'B2',
                             'train_feat',
                             'train_occ',
                             'train_emb_view',
                             'fast_logging',
                             # 'synth_rt',
                             # 'resume',
                             # 'pretrained_carl_feat',
                             # 'pretrained_carl_view',
                             # 'pretrained_carl_emb',
                             # 'pretrained_carl_occ',
]

exps['tow_trainer'] = ['carla_sta', # mode
                       'carla_static_data', # dataset
                       '100k_iters',
                       'lr4',
                       'B4',
                       'train_tow',
                       'fast_logging',
]

exps['vis_trainer'] = ['carla_sta', # mode
                       'carla_static_data', # dataset
                       '50k_iters',
                       'lr3',
                       'B2',
                       'pretrained_carl_occ',
                       'pretrained_carl_vis',
                       'frozen_occ',
                       'frozen_vis',
                       'train_feat',
                       'train_emb2D',
                       'train_emb3d',
                       'slow_logging',
]

exps['occvis_trainer'] = ['carla_sta', # mode
                          'carla_static_data', # dataset
                          '200k_iters',
                          'lr3',
                          'B4',
                          'train_occ',
                          'train_vis',
                          'slow_logging',
]


############## net configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
    # 'feat_do_rt = True',
    # 'feat_do_flip = True',
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
############## datasets ##############

# dims for mem
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4
K = 2 # how many objects to consider
S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)
groups['carla_sta1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caus2i6c1o0one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_sta10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caus2i6c1o0ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_stat_stav_data'] = ['dataset_name = "carla"',
                                  'H = %d' % H,
                                  'W = %d' % W,
                                  'trainset = "caas7i6c1o0t"',
                                  'valset = "caas7i6c1o0v"',
                                  'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                                  'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                                  'dataset_format = "npz"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)
