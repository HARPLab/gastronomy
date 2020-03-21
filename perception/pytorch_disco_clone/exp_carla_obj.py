from exp_base import *

############## choose an experiment ##############

# current = 'obj_builder'
# current = 'obj_trainer'
# current = 'obj_trainer'
# current = 'sta_builder'

# i like what i am working on here with optim and

mod = '"ion50"' # show occ_sup/ gif
mod = '"ion51"' # get proper occs sup (always X)
mod = '"ion52"' # just show gif; no tile
mod = '"ion53"' # log steps
mod = '"ion54"' # init the hole with featX0[:,-1]
mod = '"ion55"' # ones mask for viewpred
mod = '"ion56"' # add emb2D
mod = '"ion57"' # show emb2D gif; only do_feat on step0
mod = '"ion58"' # show emb2D_g gif
mod = '"ion59"' # get featXs and apply emb3D
mod = '"ion60"' # only compute loss on steps>0
mod = '"ion61"' # summ featXs
mod = '"ion62"' # only loss in valid rgb region, to focus better on the foreground
mod = '"ion63"' # init with a lazy traj (half of real t)
mod = '"ion64"' # compute and plot tlist_l2
mod = '"ion65"' # show masks ax3 too
mod = '"ion66"' # use real traj again, just to measure that tdist_l2 again
mod = '"ion67"' # use 0.5m offset

############## define experiments ##############

exps['obj_builder'] = ['carla_obj', # mode
                       'carla_obj1_data', # dataset
                       '10_iters',
                       'test_feat',
                       'pretrained_feat',
                       'frozen_feat',
                       'lr4',
                       'B1',
                       'no_shuf',
                       'no_backprop',
                       'fastest_logging',
]
exps['obj_trainer'] = ['carla_obj', # mode
                       'carla_obj1_data', # dataset
                       '10k_iters',
                       'test_feat',
                       'pretrained_feat',
                       'frozen_feat',
                       'train_view',
                       'pretrained_view',
                       'frozen_view',
                       'train_emb2D',
                       'pretrained_emb2D',
                       'frozen_emb2D',
                       'train_occ',
                       'pretrained_occ',
                       'frozen_occ',
                       'train_emb3D', # nothing to pret or freeze here
                       'lr3',
                       'B1',
                       'no_shuf',
                       'faster_logging',
]
exps['sta_builder'] = ['carla_sta', # mode
                       # 'carla_stat_stav_data', # dataset
                       'carla_sta10_data', # dataset
                       '3_iters',
                       # '20_iters',
                       'lr0',
                       'B1',
                       'no_shuf',
                       'train_feat',
                       'train_occ',
                       'train_view',
                       'train_emb2D',
                       'train_emb3D',
                       'fastest_logging',
                       # 'slow_logging',
]


############## net configs ##############

groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_rt = True',
                        'feat_do_flip = True',
                        # 'feat_do_resnet = True',
                        'feat_do_sparse_invar = True',
]
groups['test_feat'] = ['do_feat = True',
                       'feat_dim = 32',
                       # 'feat_do_resnet = True',
                       # 'feat_do_sparse_invar = True',
]
groups['train_occ'] = ['do_occ = True',
                       'occ_do_cheap = True',
                       'occ_coeff = 1.0', 
                       'occ_smooth_coeff = 2.0', 
]
groups['train_view'] = ['do_view = True',
                       'view_depth = 32',
                       'view_l1_coeff = 1.0',
]
groups['train_occ_notcheap'] = ['do_occ = True',
                                'occ_coeff = 1.0',
                                'occ_do_cheap = False',
                                'occ_smooth_coeff = 0.1',
]
groups['train_emb2D'] = ['do_emb2D = True',
                         'emb_2D_smooth_coeff = 0.01', 
                         'emb_2D_ml_coeff = 1.0', 
                         'emb_2D_l2_coeff = 0.1', 
                         'emb_2D_mindist = 32.0',
                         'emb_2D_num_samples = 2', 
]
groups['train_emb3D'] = ['do_emb3D = True',
                         'emb_3D_smooth_coeff = 0.01', 
                         'emb_3D_ml_coeff = 1.0', 
                         'emb_3D_l2_coeff = 0.1', 
                         'emb_3D_mindist = 16.0',
                         'emb_3D_num_samples = 2', 
]
groups['train_sup_flow'] = ['do_flow = True',
                            'flow_heatmap_size = 3',
                            'flow_l1_coeff = 1.0',
                            # 'do_time_flip = True',
]
groups['train_unsup_flow'] = ['do_flow = True',
                              'flow_heatmap_size = 5',
                              'flow_warp_coeff = 1.0',
                              # 'flow_hinge_coeff = 1.0',
                              # 'flow_warp_g_coeff = 1.0',
                              'flow_cycle_coeff = 0.5', # 
                              'flow_smooth_coeff = 0.1',
                              # 'flow_do_synth_rt = True',
                              # 'flow_synth_l1_coeff = 1.0',
                              # 'flow_synth_l2_coeff = 1.0',
                              # 'do_time_flip = True',
                              # 'flow_cycle_coeff = 1.0',
]


############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 6
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
                             'dataset_format = "tf"',
]
groups['carla_sta10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caus2i6c1o0ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_sta_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caus2i6c1o0t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_format = "tf"',
]
# groups['carla_stat_stav_data'] = ['dataset_name = "carla"',
#                                   'H = %d' % H,
#                                   'W = %d' % W,
#                                   'trainset = "caus2i6c1o0t"',
#                                   'valset = "caus2i6c1o0v"',
#                                   'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
#                                   'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
#                                   'dataset_format = "tf"',
# ]
groups['carla_flo1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caws2i6c0o1one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_format = "tf"',
]
groups['carla_flo10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caws2i6c0o1ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_flot_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caws2i6c0o1t"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_format = "tf"',
]
groups['carla_flot_flov_data'] = ['dataset_name = "carla"',
                                  'H = %d' % H,
                                  'W = %d' % W,
                                  'trainset = "caws2i6c0o1t"',
                                  'valset = "caws2i6c0o1v"',
                                  'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_format = "tf"',
]
groups['carla_flov_flov_data'] = ['dataset_name = "carla"',
                                  'H = %d' % H,
                                  'W = %d' % W,
                                  'trainset = "caws2i6c0o1v"',
                                  'valset = "caws2i6c0o1v"',
                                  'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_format = "tf"',
]
groups['carla_stat_stav_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0t"',
    'valset = "caas7i6c1o0v"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"'
]
groups['carla_quicktest_data'] = [
    'dataset_name = "carla"',
    'testset = "quicktest9"', # sequential
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
    'dataset_format = "tf"',
]
groups['carla_obj1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             # 'trainset = "cabs16i3c0o1one"',
                             'trainset = "picked"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                             'dataset_format = "npz"',
]
groups['carla_obj10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caws2i6c0o1ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_obj_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caws2i6c0o1t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_format = "tf"',
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
