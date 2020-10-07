from exp_base import *
import os
from munch import Munch
import ipdb
st = ipdb.set_trace

# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE
############## choose an experiment ##############

# current = 'builder'
# current = 'trainer_basic'
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

current = '{}'.format(os.environ["exp_name"])
mod = '"{}"'.format(os.environ["run_name"]) # debug

# st()
####### Final sony exp ########
exps['clevr_multiple_trainer_hard_exp5_pret_moc_orient'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}


############## define experiments ##############

exps['dummy_exp'] = {
    'groups':['nel_sta','dummy_dataset','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5']}
}

exps['builder'] = {
    'groups':['nel_sta','clevr_veggies_sta_data','train_feat','B2','debug','train_emb3D','debug','pretrained_feat'],
    'group_parents':['exp','max']
}

exps['builder_big'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','B2','debug','train_occ_gt','train_view_gt','train_emb2D_gt','train_emb3D_gt','train_emb3D','debug','pretrained_feat'],
    'group_parents':['exp','max']
}


exps['builder_big_gt'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','B2','debug','train_occ_gt','train_view_gt','train_emb2D_gt','train_emb3D_gt','train_emb3D','debug','pretrained_feat','use_gt'],
    'group_parents':['exp','max']
}







exps['trainer_big_rgb'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_emb3D','pretrained_feat','pretrained_view'],
    'group_parents':['exp','max']
}


exps['trainer_big_rgb_occ'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ'],
    'group_parents':['exp','max']
}

exps['trainer_big_rgb_occ_emb3d'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ'],
    'group_parents':['exp','max']
}


exps['trainer_big_rgb_occ_emb3d_emb2d'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ','pretrained_emb2D'],
    'group_parents':['exp','max']
}








exps['trainer_big_rgb_occ_gt_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ','use_gt'],
    'group_parents':['exp','max']
}

exps['trainer_big_rgb_occ_gt_cotrain_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ','use_gt'],
    'group_parents':['exp','max']
}

exps['trainer_big_rgb_occ_emb3d_gt_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ','use_gt'],
    'group_parents':['exp','max']
}

exps['trainer_big_rgb_occ_emb3d_gt_cotrain_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D_gt','train_emb3D','pretrained_feat','pretrained_view','pretrained_occ','use_gt'],
    'group_parents':['exp','max']
}


exps['trainer_big_rgb_occ_gt_init'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','use_gt'],
    'group_parents':['exp','max']
}



exps['trainer_big_rgb_occ_gt_cotrain_init'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','use_gt'],
    'group_parents':['exp','max']
}

exps['trainer_big_rgb_occ_gt_multiview_init'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','train_emb3D','use_gt'],
    'group_parents':['exp','max']
}








exps['trainer_big_rgb_occ_gt_init_moc_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','train_view_gt','train_occ_gt','use_gt'],
    'group_parents':['exp','max','emb_moc']
}

exps['trainer_big_rgb_occ_gt_init_moc_pret_lr4'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','use_gt','pretrained_feat'],
    'group_parents':['exp','max','emb_moc']
}

exps['trainer_big_rgb_occ_gt_init_moc_no_bn_lr4'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','train_view_gt','train_occ_gt','use_gt','no_bn'],
    'group_parents':['exp','max','emb_moc']
}

exps['trainer_big_rgb_occ_gt_init_moc_no_bn_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','train_view_gt','train_occ_gt','use_gt','no_bn','pretrained_feat'],
    'group_parents':['exp','max','emb_moc']
}






#non gt curves

exps['trainer_big_moc_pret_moc_no_bn_lr4'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':['exp','max','emb_moc']
}

exps['trainer_big_rgb_occ_pret_moc_no_bn_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','no_bn','pretrained_feat'],
    'group_parents':['exp','max','emb_moc']
}

exps['trainer_big_rgb_occ_pret_moc_no_bn_lr4'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':['exp','max','emb_moc']
}


exps['trainer_big_rgb_occ_pret_no_bn_lr4'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr4','B2','no_bn','pretrained_feat','train_emb3D'],
    'group_parents':['exp','max']
}




exps['trainer_big_rgb_occ_init_moc_no_bn_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','no_bn','use_gt'],
    'group_parents':['exp','max','emb_moc']
}


exps['trainer_big_rgb_occ_init_no_bn_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','no_bn','use_gt','train_emb3D'],
    'group_parents':['exp','max']
}




# hardmining art paper



exps['trainer_big_builder_hard_exp_big'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn'],
    'group_parents':{'exp':['do','tdata','exp_custom_vbig'],
                    'max':['do','max_custom_small_hard']}
}



exps['trainer_big_builder_hard_exp_big_real'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_vbig'],
                    'max':['do','max_custom_small_hard']}
}

exps['trainer_big_builder_hard_exp_big_real_orient'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_vbig'],
                    'max':['do','max_custom_small_hard']}
}


exps['trainer_big_builder_hard_gt_init'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn'],
    'group_parents':['exp','max']
}



exps['trainer_big_builder_hard_debug'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':['exp','max']
}

exps['trainer_big_builder_hard_small'] = {
    'groups':['nel_sta','clevr_veggies_sta_data_big','train_feat','lr3','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_small_hard']}
}

# lr4



exps['trainer_big_builder_hard_exp1'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp2'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr3','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp3'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp3']}
}

exps['trainer_big_builder_hard_exp4'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp4']}
}


exps['trainer_big_builder_hard_exp5'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp6'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp7'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp3']}
}

exps['trainer_big_builder_hard_exp8'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp4']}
}





exps['trainer_big_builder_hard_exp1_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp1_pret_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr3','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp3_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp3']}
}

# best model so far



exps['trainer_big_builder_hard_exp5_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5']}
}

exps['trainer_big_builder_hard_exp5_pret_moc'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}


# self improving detector

# carla

exps['carla_trainer_big_builder_hard_exp5_pret_moc_orient'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['carla_trainer_big_builder_hard_exp5_pret_moc_orient_low_res'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['carla_trainer_big_builder_hard_exp5_pret_moc_orient_low_res_random_noise'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res','random_noise'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['carla_trainer_big_builder_hard_exp5_pret_low_res'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_except']}
}




exps['bigbird_trainer_big_builder_hard_exp5_pret_moc_orient_low_res'] = {
    'groups':['nel_sta','bigbird_multiview_data_gg','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}


exps['bigbird_trainer_big_builder_hard_exp5_pret_moc_orient_low_res_random_noise'] = {
    'groups':['nel_sta','bigbird_multiview_data_gg','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res','random_noise'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['bigbird_trainer_big_builder_hard_exp5_pret_low_res'] = {
    'groups':['nel_sta','bigbird_multiview_data_gg','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','low_res'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_except']}
}



groups['bigbird_multiview_data_gg'] = ['dataset_name = "bigbird"',
                             'H = %d' % 256,
                             'W = %d' % 256,
                             'trainset = "ggt"',
                             'dataset_list_dir = "CHANGE_ME/bigbird_processed/npy"',
                             'dataset_location = "CHANGE_ME/bigbird_processed/npy"',
                             'root_dataset = "CHANGE_ME/bigbird_processed"',
                             'dataset_format = "npz"'
]








exps['carla_trainer_big_builder_hard_exp5_pret_moc_small'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','debug'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],'max':['do','max_custom_hard_exp5_moc_small']}
}



exps['carla_trainer_big_builder_hard_exp5_pret_moc_orientcheck_small'] = {
    'groups':['nel_sta','carla_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','debug','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],'max':['do','max_custom_hard_exp5_moc_except_small']}
}








#clevr


exps['trainer_multiple_builder_hard_exp5_pret_moc_orientcheck_small'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','debug','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_except_small']}
}


exps['clevr_multiple_trainer_hard_exp5_pret_moc_orient'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}



exps['clevr_multiple_trainer_hard_exp5_pret_moc_orient_2dboxes'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check', 'use_2d_boxes'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['clevr_multiple_trainer_hard_exp5_pret_moc_orient_detboxes'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check', 'use_det_boxes'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}



exps['clevr_multiple_trainer_hard_exp5_pret_moc_orient_random_noise'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check','random_noise'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}


exps['clevr_multiple_trainer_hard_exp5_pret_orient_small_ram'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_except_small_ram']}
}


exps['clevr_multiple_trainer_hard_exp5_pret_orient_small'] = {
    'groups':['nel_sta','clevr_veggies_sta_multiple_rotate','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_small_samp']}
}








# replica


exps['replica_trainer_multiple_builder_hard_exp5_pret_moc_orientcheck_small'] = {
    'groups':['nel_sta','replica_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','debug','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_except_small']}
}


exps['replica_trainer_hard_exp5_pret_moc_orient'] = {
    'groups':['nel_sta','replica_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}


































# self improving detector


exps['trainer_big_builder_hard_exp5_pret_moc_real'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}


exps['trainer_big_builder_hard_exp5_pret_moc_real_orient'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

exps['trainer_big_builder_hard_exp5_pret_moc_imgnet'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','imgnet'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}


exps['trainer_big_builder_hard_exp5_pret_moc_imgnet_pret'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','imgnet','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}



exps['trainer_big_builder_hard_exp5_pret_moc_imgnet_v2'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','imgnet_v2'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}




exps['trainer_big_builder_hard_exp5_pret_moc_small'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_small']}
}

exps['trainer_big_builder_hard_exp5_pret_moc_real_small'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_small']}
}

exps['trainer_big_builder_hard_exp5_pret_moc_real_orient_small'] = {
    'groups':['nel_sta','real_sta_data','train_feat','lr4','B2','no_bn','pretrained_feat','orient_check'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_except_small']}
}





exps['trainer_big_builder_hard_exp5_pret_moc_imgnet_small'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','imgnet'],
    'group_parents':{'exp':['do','tdata','exp_custom_small'],
                    'max':['do','max_custom_hard_exp5_moc_except']}
}

# best model so far
exps['trainer_big_builder_hard_exp5_pret_shapes'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5']}
}

exps['trainer_big_builder_hard_exp5_pret_shapes_moc'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_moc']}
}




exps['trainer_big_builder_hard_exp1_pret_large_samp'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1_large_samp']}
}


exps['trainer_big_builder_hard_exp5_pret_large_samp'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_large_samp']}
}

exps['trainer_big_builder_hard_exp5_pret_large_samp_shapes'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale','train_feat','lr4','B2','no_bn','pretrained_feat'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_large_samp']}
}



exps['trainer_big_builder_hard_exp1_init_small_samp'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1_small_samp']}
}

exps['trainer_big_builder_hard_exp5_init_small_samp'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp5_small_samp']}
}

exps['trainer_big_builder_hard_exp1_init_lr3'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr3','B2','no_bn'],

    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp1']}
}

exps['trainer_big_builder_hard_exp3_init'] = {
    'groups':['nel_sta','clevr_veggies_sta_shear_scale_f','train_feat','lr4','B2','no_bn'],
    'group_parents':{'exp':['do','tdata','exp_custom_big'],
                    'max':['do','max_custom_hard_exp3']}
}


# exp 1:  pretrained model  normal
# exp 2:  pretrained model  normal lr3
# exp 4: pretrained model  normal randomly scale
# exp 3: pretrained model  diff permuation of  search region

# exp 5: pretrained model  diff permuation of  train valid region
# exp 6: pretrained model  diff permuation of  numRetreived and topk
# exp 7: pretrained model  diff permuation of  numRetreived and topk
# exp 8: pretrained model with gt


# old


groups['imgnet'] = [
    'imgnet = True',
    # 'Z = 64'
    # 'Y = 64'
    # 'X = 64'
    # 'Z2 = Z//2'
    # 'Y2 = Y//2'
    # 'X2 = X//2 '   
]

groups['imgnet_v2'] = [
    'imgnet = True',
    'imgnet_v1 = False',
    'Z = 96',
    'Y = 96',
    'X = 96',
    'Z2 = 48',
    'Y2 = 48',
    'X2 = 48',   
]
groups['max_custom_hard_exp1'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',    
]


groups['max_custom_hard_exp3'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'shouldResizeToRandomScale = True',
]

groups['max_custom_hard_exp4'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'searchRegion = 3',
]

groups['max_custom_hard_exp5'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
]

groups['max_custom_hard_exp5_except'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
    'exceptions = True',
]

groups['max_custom_hard_exp5_except_small_ram'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
    'exceptions = True',
    'numRetrievalsForEachQueryEmb = 15'
]


groups['max_custom_hard_exp5_moc'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'hard_moc = True',
    'trainRegion = 8',
    'hard_moc_qsize = 100000',
]


groups['max_custom_hard_exp5_moc_except'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'hard_moc = True',
    'trainRegion = 8',
    'hard_moc_qsize = 100000',
    'exceptions = True',
]


groups['max_custom_hard_exp5_moc_small'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 1',
    'nbImgEpoch = 5',    
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'hard_moc = True',
    'trainRegion = 8',
    'hard_moc_qsize = 1000',    
]
groups['max_custom_hard_exp5_moc_except_small'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 1',
    'nbImgEpoch = 5',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'hard_moc = True',
    'trainRegion = 8',
    'hard_moc_qsize = 1000',    
    'exceptions = True',

]

groups['max_custom_hard_exp5_except_small'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 1',
    'nbImgEpoch = 5',
    'max_epochs = 2',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
    'exceptions = True',

]



groups['max_custom_hard_exp5_large_samp'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 1',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
    'numRetrievalsForEachQueryEmb = 22',
    'topK = 15',        
]

groups['max_custom_hard_exp1_large_samp'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 1',    
    'hardmining = True',    
    'numRetrievalsForEachQueryEmb = 22',
    'topK = 15',
]



groups['max_custom_hard_exp5_small_samp'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 1',    
    'hardmining = True',
    'margin = 4',
    'trainRegion = 8',
    'numRetrievalsForEachQueryEmb = 25',
    'topK = 3',        
]

groups['max_custom_hard_exp1_small_samp'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 1',    
    'hardmining = True',    
    'numRetrievalsForEachQueryEmb = 25',
    'topK = 3',
]

groups['max_custom_hard_exp6'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 1',    
    'hardmining = True',    
    'numRetrievalsForEachQueryEmb = 10',
    'topK = 5',
]


groups['max_custom_hard_exp7'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',
    'numRetrievalsForEachQueryEmb = 30',
    'topK = 15',
]

groups['max_custom_hard_gt_exp8'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',    
    'hardmining_gt = True',    
]






groups['max_custom'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 200',    
]
groups['max_custom_hard'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',    
]

groups['max_custom_hard_vis'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',    
    'visualizeHardMines = True',
]

groups['max_custom_hard_gt'] = [
    'max_iters = 300',
    'B = 2',
    'log_freq = 100',
    'max_epochs = 2',    
    'hardmining = True',    
    'hardmining_gt = True',    
]

groups['max_custom_small_hard_gt'] = [
    'max_iters = 5',
    'B = 2',
    'log_freq = 1',
    'hardmining = True',
    'hardmining_gt = True',
    'nbImgEpoch = 10',
    'max_epochs = 2',
    'topK = 3',
]

groups['max_custom_small_hard'] = [
    'max_iters = 5',
    'B = 2',
    'log_freq = 1',
    'hardmining = True',
    'nbImgEpoch = 10',
    'max_epochs = 2',
    'topK = 3',
    'shouldResizeToRandomScale = True',    
]
groups['max_custom_small_hard_vis'] = [
    'max_iters = 5',
    'B = 2',
    'log_freq = 1',
    'hardmining = True',
    'nbImgEpoch = 10',
    'max_epochs = 2',
    'topK = 3',
    'visualizeHardMines = True',
]

groups['max_custom_small'] = [
    'max_iters = 10',
    'B = 2',
    'log_freq = 1',    
]



# exps




groups['exp_custom'] = [
    'max_iters = 100',
    'log_freq = 99',
]


groups['exp_custom_small'] = [
    'max_iters = 10',
    'log_freq = 1',
    'do_debug = True',
]


groups['exp_custom_small_no_update'] = [
    'max_iters = 10',
    'log_freq = 1',
    'do_debug = True',
    'no_update = True',

]

groups['exp_custom_big'] = [
    'max_iters = 100',
    'log_freq = 100',
]

groups['exp_custom_vbig'] = [
    'max_iters = 1200',
    'log_freq = 50',
]


groups['exp_custom_big_no_update'] = [
    'max_iters = 10',
    'log_freq = 100',
    'no_update = True',
]





# emb_moc

groups['emb_moc_custom_normal'] = [
    'max_iters_init = 1500',
    'max_pool_indices = 100000',
    'indexes_to_take = 128',
    'normal_queue = True',
    # 'own_data_loader = True',
]

groups['emb_moc_custom'] = [
    'max_iters_init = 1500',
    'max_pool_indices = 5000',
    'indexes_to_take = 128',    
]

groups['emb_moc_custom_small'] = [
    'max_iters_init = 500',
    'max_pool_indices = 500',
    'indexes_to_take = 128',
    'normal_queue = True',    
    # 'own_data_loader = True'
]

# group_parents['exp'] = [
#     'do',
#     'tdata',
#     '10_iters',
#     'fast_logging_group',
# ]

# group_parents['max'] = [
#     'do',
#     'fast_logging_group',
#     '10_iters',        
#     'B2',
#     'cotrain',
# ]
############## group parent configs ##############
groups['do'] = ['do = True']
groups['tdata'] = ['tdata = True']
groups['cotrain'] = ['g_max_iters = 1']
groups['multiview'] = ['p_max_iters = 0','g_max_iters = 1']
groups['no_bn'] = ['no_bn = True']

############## net configs ##############

groups['train_preocc'] = ['do_preocc = True']

groups['do_gen_pcds'] = [
  'GENERATE_PCD_BBOX = True'
]
groups['object_specific'] = [
    'do_object_specific = True',
    'do_eval_boxes = True',
]
groups['debug'] = [
    'eval_compute_freq = 9',
    'do_debug = True',
]
groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
]
groups['use_gt'] = [
    'gt = True',
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
groups['train_occ_gt'] = [
    'do_occ_gt = True',
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
groups['train_view_gt'] = [
    'do_view_gt = True',
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

groups['train_emb2D_gt'] = [
    'do_emb2D_gt = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]

groups['train_emb3D_gt'] = [
    'do_emb3D_gt = True',
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
    'emb_3D_num_samples = 2',
    'emb3D_o = True',
    'do_eval_boxes = True',

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
]
groups['no_eval_recall'] = ['do_eval_recall = False']

groups['orient_check'] = [
    'do_orientation = True',
]
############## datasets ##############

# dims for mem
# SIZE = 32
import socket
if "Alien"  in socket.gethostname():
    SIZE = 24
else:
    SIZE = 36

# SIZE = 72
NUM_VIEWS = 24
# 56
Z = SIZE*4
Y = SIZE*4
X = SIZE*4

Z2 = Z//2
Y2 = Y//2
X2 = X//2

BOX_SIZE = 16

K = 2 # how many objects to consider
S = 2
H = 240
W = 320
N = 3
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

DATA_MOD = "aa"



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



groups['clevr_veggies_sta_data_big'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "cg_Ft"',
                             'valset = "cg_Fv"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]

groups['clevr_veggies_sta_shear_scale'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "dit"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_veggies/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
]


groups['clevr_veggies_sta_shear_scale_f'] = ['dataset_name = "clevr"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "di_Ft"',
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
                             f'dataset_list_dir = "./clevr_veggies/npys"',
                             f'dataset_location = "./clevr_veggies/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_veggies"',
                             'dataset_format = "npz"'
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


groups['replica_sta_data_temp'] = [
    'dataset_name = "replica"',
    'H = %d' % 256,
    'W = %d' % 256,
    'N = %d' % 50,
    'trainset = "tempt"',
    'low_res = True',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/replica_processed/npy"',
    'dataset_location = "CHANGE_ME/replica_processed/npy"',
    'root_dataset = "CHANGE_ME/replica_processed"',
    'dataset_format = "npz"',
]


groups['real_sta_data'] = ['dataset_name = "real"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "abt"',
                             'valset = "abv"',
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




groups['carla_sta_data'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 256,
    'trainset = "bbt"',
    'root_keyword = "home"',
    'dataset_list_dir = "CHANGE_ME/carla/npy"',
    'dataset_location = "CHANGE_ME/carla/npy"',
    'root_dataset = "CHANGE_ME/carla"',
    'dataset_format = "npz"',
]

############## verify and execute ##############

def _verify_(s):
    try:
        varname, eq, val = s.split(' ')
    except Exception:
        st()
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')


def _verify_groupParent(s):
    varname, eq, val = s.split(' ')
    group_parent,attr = varname.split(".")
    try:
        assert group_parent in globals()
        exec(f'assert "{attr}" in {group_parent}')
        assert eq == '='
        assert type(s) is type('')
    except Exception as e:
        print(e)
        st()

print(current)

assert current in exps
for group in exps[current]['groups']:
    print("  " + group)
    if group not in groups:
      st()
      assert False
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

for group_parent in exps[current]['group_parents'].keys():
    print("  " + group_parent)
    for group in exps[current]['group_parents'][group_parent]:
        if group not in groups:
          st()
          assert False
        for s in groups[group]:
            print("    " + s)
            command = '{}.{}'.format(group_parent, s)
            _verify_groupParent(command)
            exec(command)

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


elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)    