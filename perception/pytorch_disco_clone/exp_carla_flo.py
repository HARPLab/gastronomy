from exp_base import *

# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE

############## choose an experiment ##############

current = 'flo_builder'

# (NO MODS HERE)
mod = '""'


############## define experiments ##############

exps['flo_builder'] = ['carla_flo', # mode
                       'carla_flo_data', # dataset
                       '10_iters',
                       'train_feat',
                       'train_flow',
                       'lr4',
                       'B1',
                       'no_shuf',
                       # 'no_backprop',
                       'fastest_logging',
]
exps['flo_trainer'] = ['carla_flo', # mode
                       'carla_flo_data', # dataset
                       '100k_iters',
                       'train_feat',
                       'train_flow',
                       'lr4',
                       'B4',
                       'fast_logging',
]

############## net configs ##############

groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 16',
                        # 'feat_do_rt = True',
                        'feat_do_flip = True',
]
groups['train_flow'] = ['do_flow = True',
                        # 'flow_warp_coeff = 10.0', # maybe with l2 norm i need this higher than before
                        # 'flow_cycle_coeff = 1.0', # 
                        # 'flow_smooth_coeff = 0.1',
                        'flow_l1_coeff = 1.0',
                        # 'flow_cycle_coeff = 0.1',
                        'do_time_flip = True',
]

############## datasets ##############

# DHW for mem stuff
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

groups['carla_flo1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caws2i6c0o1one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_flo10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caws2i6c0o1ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_flo_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caws2i6c0o1t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
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
