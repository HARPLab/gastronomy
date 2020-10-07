from backend import readers
# import tensorflow as tf
import numpy as np
np.random.seed(seed=1)
import torch
from torch.utils.data import DataLoader
import hyperparams as hyp
import pickle
import os
import ipdb
st = ipdb.set_trace
import utils_improc
import utils_geom
from scipy.misc import imresize

class TFRecordDataset():

    def __init__(self, dataset_path, shuffle=True, val=False):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        self.nRecords = nRecords
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
            
        dataset = tf.data.TFRecordDataset(
            records,
            compression_type="GZIP"
        ).repeat()
        
        if val:
            num_threads = 1
        else:
            num_threads = 4

        if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti' or hyp.dataset_name=='clevr':
            dataset = dataset.map(readers.carla_parser,
                                  num_parallel_calls=num_threads)
        else:
            assert(False) # reader not ready yet
            
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(hyp.B)
        self.dataset = dataset
        self.iterator = None
        self.sess = tf.Session()

        self.iterator = self.dataset.make_one_shot_iterator()
        self.batch_to_run = self.iterator.get_next()

    def __getitem__(self, index):
        try:
            batch = self.sess.run(self.batch_to_run)
        except tf.errors.OutOfRangeError:
            self.iterator = self.dataset.make_one_shot_iterator()
            self.batch_to_run = self.iterator.get_next()
            batch = self.sess.run(self.batch_to_run)

        batch_torch = []
        for b in batch:
            batch_torch.append(torch.tensor(b))

        d = {}
        [d['pix_T_cams'],
         d['cam_T_velos'],
         d['origin_T_camRs'],
         d['origin_T_camXs'],
         d['rgb_camRs'],
         d['rgb_camXs'],
         d['xyz_veloXs'],
         d['boxes3D'], 
         d['tids'], 
         d['scores'], 
         ] = batch_torch

        if hyp.do_time_flip:
            d = random_time_flip_batch(d)
            
        return d

    def __len__(self):
        return 10000000000 #never end 

class NpzRecordDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        # st()
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        # st()
        self.records = records
        self.shuffle = shuffle    

    def __getitem__(self, index):
        if hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or  hyp.dataset_name=='real'  or hyp.dataset_name=="bigbird" or hyp.dataset_name=="carla" or hyp.dataset_name =="carla_mix" or hyp.dataset_name =="replica":
            filename = self.records[index]
            d = pickle.load(open(filename,"rb"))
            d = dict(d)
        # elif hyp.dataset_name=="carla":
        #     filename = self.records[index]
        #     d = np.load(filename)
        #     d = dict(d)
                
        #     d['rgb_camXs_raw'] = d['rgb_camXs']
        #     d['pix_T_cams_raw'] = d['pix_T_cams']
        #     d['tree_seq_filename'] = "dummy_tree_filename"
        #     d['origin_T_camXs_raw'] = d['origin_T_camXs']
        #     d['camR_T_origin_raw'] = utils_geom.safe_inverse(torch.from_numpy(d['origin_T_camRs'])).numpy()
        #     d['xyz_camXs_raw'] = d['xyz_camXs']

        else:
            assert(False) # reader not ready yet
        

        if hyp.do_empty:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
                'empty_rgb_camXs_raw',
                'empty_xyz_camXs_raw',            
            ]
        else:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]
        
        # if hyp.do_time_flip:
        #     d = random_time_flip_single(d,item_names)
        # if the sequence length > 2, select S frames
        # filename = d['raw_seq_filename']
        original_filename = filename
        if hyp.dataset_name =="carla_mix":
            bbox_origin_gt = d['bbox_origin']
            if 'bbox_origin_predicted' not in d:
                bbox_origin_predicted = []
            else:
                bbox_origin_predicted = d['bbox_origin_predicted']
            # st()

            classes = d['obj_name']
            if isinstance(classes,type('')):
                classes = [classes]

            d['tree_seq_filename'] = "temp"
        
        if hyp.dataset_name =="replica":
            d['tree_seq_filename'] = "temp"
            object_category = d['object_category_names']
            bbox_origin = d['bbox_origin']
        

        if hyp.dataset_name=="carla":
            camR_index = d['camR_index']
            rgb_camtop = d['rgb_camXs_raw'][camR_index:camR_index+1]
            origin_T_camXs_top = d['origin_T_camXs_raw'][camR_index:camR_index+1]
            # predicted_box  = d['bbox_origin_predicted']
            predicted_box = []    
        filename = d['tree_seq_filename']
        
        if self.shuffle or hyp.randomly_select_views:
            d,indexes = random_select_single(d, item_names, num_samples=hyp.S)
        else:
            d,indexes = non_random_select_single(d, item_names, num_samples=hyp.S)

        filename_g = "/".join([original_filename,str(indexes[0])])
        filename_e = "/".join([original_filename,str(indexes[1])])

        rgb_camXs = d['rgb_camXs_raw']
        # move channel dim inward, like pytorch wants
        # rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])

        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = rgb_camXs[:,:3]
        rgb_camXs = utils_improc.preprocess_color(rgb_camXs)

        # st()
        if hyp.dataset_name=="carla":
            rgb_camtop = np.transpose(rgb_camtop, axes=[0, 3, 1, 2])
            rgb_camtop = rgb_camtop[:,:3]
            rgb_camtop = utils_improc.preprocess_color(rgb_camtop)
            d['rgb_camtop'] = rgb_camtop
            d['origin_T_camXs_top'] = origin_T_camXs_top
            if len(predicted_box) == 0:
                predicted_box = np.zeros([hyp.N,6])
                score = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = predicted_box.shape[0]
                score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                predicted_box = np.pad(predicted_box,[[0,hyp.N-num_boxes],[0,0]])
            d['predicted_box'] = predicted_box.astype(np.float32)
            d['predicted_scores'] = score.astype(np.float32)

        if hyp.dataset_name=="replica":
            if len(bbox_origin) == 0:
                score = np.zeros([hyp.N])
                bbox_origin = np.zeros([hyp.N,6])
                object_category = ["0"]*hyp.N
                # st()
                object_category = np.array(object_category)
            else:
                num_boxes = len(bbox_origin)
                # st()
                bbox_origin = torch.stack(bbox_origin).numpy().squeeze(1).squeeze(1).reshape([num_boxes,6])
                bbox_origin = np.array(bbox_origin)
                score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0]])
                object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
            d['gt_box'] = bbox_origin.astype(np.float32)
            d['gt_scores'] = score.astype(np.float32)
            d['classes']  = list(object_category)
            # st()

        if hyp.dataset_name =="carla_mix":
            bbox_origin_predicted = bbox_origin_predicted[:3]
            if len(bbox_origin_gt.shape) ==1:
                bbox_origin_gt = np.expand_dims(bbox_origin_gt,0)
            num_boxes = bbox_origin_gt.shape[0]
            # st()
            score_gt = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin_gt = np.pad(bbox_origin_gt,[[0,hyp.N-num_boxes],[0,0]])
            classes = np.pad(classes,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
            if len(bbox_origin_predicted) == 0:
                bbox_origin_predicted = np.zeros([hyp.N,6])
                score_pred = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = bbox_origin_predicted.shape[0]
                score_pred = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin_predicted = np.pad(bbox_origin_predicted,[[0,hyp.N-num_boxes],[0,0]])
                
            d['predicted_box'] = bbox_origin_predicted.astype(np.float32)
            d['predicted_scores'] = score_pred.astype(np.float32)            
            d['gt_box'] = bbox_origin_gt.astype(np.float32)
            d['gt_scores'] = score_gt.astype(np.float32)
            # st()
            d['classes']  = list(classes)

        d['rgb_camXs_raw'] = rgb_camXs

        if hyp.dataset_name!="carla" and hyp.do_empty:
            empty_rgb_camXs = d['empty_rgb_camXs_raw']
            # move channel dim inward, like pytorch wants
            empty_rgb_camXs = np.transpose(empty_rgb_camXs, axes=[0, 3, 1, 2])
            empty_rgb_camXs = empty_rgb_camXs[:,:3]
            empty_rgb_camXs = utils_improc.preprocess_color(empty_rgb_camXs)
            d['empty_rgb_camXs_raw'] = empty_rgb_camXs
        d['tree_seq_filename'] = filename
        d['filename_e'] = filename_e
        d['filename_g'] = filename_g
        return d

    def __len__(self):
        return len(self.records)

def random_select_single(batch,item_names, num_samples=2):
    num_all = len(batch[item_names[0]]) #total number of frames
    
    batch_new = {}
    # select valid candidate
    if 'valid_pairs' in batch:
        valid_pairs = batch['valid_pairs'] #this is ? x 2
        sample_pair = np.random.randint(0, len(valid_pairs), 1).squeeze()
        sample_id = valid_pairs[sample_pair, :] #this is length-2
    else:
        sample_id = range(num_all)

    final_sample = np.random.choice(sample_id, size=num_samples, replace=False)

    if num_samples > len(sample_id):
        print('Inputs.py. Warning: S larger than valid frames number')

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    return batch_new,final_sample

def non_random_select_single(batch, item_names, num_samples=2):
    num_all = len(batch[item_names[0]]) #total number of frames
    
    batch_new = {}
    # select valid candidate
    if 'valid_pairs' in batch:
        valid_pairs = batch['valid_pairs'] #this is ? x 2
        sample_pair = -1
        sample_id = valid_pairs[sample_pair, :] #this is length-2
    else:
        sample_id = range(num_all)

    if len(sample_id) > num_samples:
        final_sample = sample_id[-num_samples:]
    else:
        final_sample = sample_id

    if num_samples > len(sample_id):
        print('Inputs.py. Warning: S larger than valid frames number')

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    return batch_new,final_sample

def random_time_flip_batch(batch, item_names):
    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)
    

    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            # flip along the seq dim
            item = item.flip(1)
        batch[item_name] = item
        
    return batch

def random_time_flip_single(batch, item_names):
    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)
    
    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            if torch.is_tensor(item):
                # flip along the seq dim
                item = item.flip(0)
            else: #numpy array
                item = np.flip(item, axis=0)
        batch[item_name] = item
        
    return batch

def specific_select_single(batch,item_names, index):
    num_all = len(batch[item_names[0]]) #total number of frames
    batch_new = {}
    for item_name in item_names:
        item = batch[item_name]
        item = item[index]
        batch_new[item_name] = item
    return batch_new


def merge_e_g(d_e,d_g,item_names):
    d = {}    
    for item_name in item_names:
        d_e_item = d_e[item_name]
        d_g_item = d_g[item_name]
        d_item = np.stack([d_g_item,d_e_item])
        d[item_name] = d_item
    return d

def get_bbox(bbox_origin,object_category):
    if len(bbox_origin) == 0:
        score = np.zeros([hyp.N])
        bbox_origin = np.zeros([hyp.N,6])
        object_category = ["0"]*hyp.N
        # st()
        object_category = np.array(object_category)
    else:
        num_boxes = len(bbox_origin)
        # st()
        bbox_origin = torch.stack(bbox_origin).numpy().squeeze(1).squeeze(1).reshape([num_boxes,6])
        bbox_origin = np.array(bbox_origin)
        score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
        bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0]])
        object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
    return bbox_origin,score,object_category

class NpzCustomRecordDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, shuffle):
        nRecords = len(filenames)
        print('found %d records' % (nRecords))
        nCheck = np.min([nRecords, 1000])
        if hyp.emb_moc.own_data_loader:
            for record in filenames[:nCheck]:
                assert os.path.isfile(record[0]), 'Record at %s was not found' % record
                assert os.path.isfile(record[1]), 'Record at %s was not found' % record
        else:
            for record in filenames[:nCheck]:
                assert os.path.isfile("/".join(record[0].split("/")[:-1])), 'Record at %s was not found' % record
                assert os.path.isfile("/".join(record[1].split("/")[:-1])), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        self.records = filenames
        self.shuffle = shuffle

    def __getitem__(self, index):
        if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or hyp.dataset_name=='real' or  hyp.dataset_name=="bigbird" or  hyp.dataset_name=="carla_mix"  or  hyp.dataset_name=="replica":
            filename = self.records[index]
            filename_e,filename_g = filename
            if hyp.emb_moc.own_data_loader:
                d_e = pickle.load(open(filename_e,"rb"))
                d_g = pickle.load(open(filename_g,"rb"))
                index_e_parts = str(np.random.randint(0,hyp.NUM_VIEWS))
                index_g_parts = str(np.random.randint(0,hyp.NUM_VIEWS))
            else:
                filename_e_parts = filename_e.split("/")
                index_e_parts = filename_e_parts[-1]
                main_filename_e =  "/".join(filename_e_parts[:-1])

                filename_g_parts = filename_g.split("/")
                index_g_parts = filename_g_parts[-1]
                main_filename_g =  "/".join(filename_g_parts[:-1])

                d_e = pickle.load(open(main_filename_e,"rb"))
                d_g = pickle.load(open(main_filename_g,"rb"))
        else:
            assert(False) # reader not ready yet
        if hyp.do_empty:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
                'empty_rgb_camXs_raw',
                'empty_xyz_camXs_raw',            
            ]
        else:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]

        d_e = dict(d_e)
        d_g = dict(d_g)
        
        if not hyp.dataset_name == "carla_mix" and not hyp.dataset_name == "replica":
            filename_de = d_e['tree_seq_filename']
            filename_dg = d_g['tree_seq_filename']
        else:
            filename_de = filename_e
            filename_dg = filename_g
        
        if hyp.dataset_name =="carla_mix":
            bbox_origin_gt = d_e['bbox_origin']
            bbox_origin_predicted = d_e['bbox_origin_predicted']
        
        if hyp.dataset_name =="replica":
            obj_cat_name_e = d_e['object_category_names']
            obj_cat_name_g = d_g['object_category_names']
            bbox_origin_gt_e = d_e['bbox_origin']
            bbox_origin_gt_g = d_g['bbox_origin']

            # d['tree_seq_filename'] = "temp"

        d_e = specific_select_single(d_e, item_names, int(index_e_parts))
        d_g = specific_select_single(d_g, item_names, int(index_g_parts))

        d = merge_e_g(d_e,d_g,item_names)
        
        # merge is g and e
        if hyp.dataset_name =="carla_mix":
            bbox_origin_predicted = bbox_origin_predicted[:3]
            if len(bbox_origin_gt.shape) ==1:
                bbox_origin_gt = np.expand_dims(bbox_origin_gt,0)
            num_boxes = bbox_origin_gt.shape[0]
            # st()
            score_gt = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin_gt = np.pad(bbox_origin_gt,[[0,hyp.N-num_boxes],[0,0]])

            if len(bbox_origin_predicted) == 0:
                bbox_origin_predicted = np.zeros([hyp.N,6])
                score_pred = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = bbox_origin_predicted.shape[0]
                score_pred = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin_predicted = np.pad(bbox_origin_predicted,[[0,hyp.N-num_boxes],[0,0]])
            d['predicted_box'] = bbox_origin_predicted.astype(np.float32)
            d['predicted_scores'] = score_pred.astype(np.float32)            
            d['gt_box'] = bbox_origin_gt.astype(np.float32)
            d['gt_scores'] = score_gt.astype(np.float32)


        d["filename_e"] = filename_de
        d["filename_g"] = filename_dg
        d["tree_seq_filename"] = filename_dg
        rgb_camXs = d['rgb_camXs_raw']

        if hyp.dataset_name =="replica":
            bbox_origin_e, score_e, object_category_e = get_bbox(bbox_origin_gt_e,obj_cat_name_e)
            bbox_origin_g, score_g, object_category_g = get_bbox(bbox_origin_gt_g,obj_cat_name_g)
            d['object_category_names_e'] = list(object_category_e)
            d['object_category_names_g'] = list(object_category_g)
            d['bbox_origin_e'] = bbox_origin_e
            d['bbox_origin_g'] = bbox_origin_g
            d['scores_e'] = score_e
            d['scores_g'] = score_g

        # move channel dim inward, like pytorch wants
        # rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = rgb_camXs[:,:3]
        # rgb_camRs = utils_improc.preprocess_color(rgb_camRs)
        rgb_camXs = utils_improc.preprocess_color(rgb_camXs)

        d['rgb_camXs_raw'] = rgb_camXs

        d['index_val'] = index

        if hyp.do_empty:
            empty_rgb_camXs = d['empty_rgb_camXs_raw']
            empty_rgb_camXs = np.transpose(empty_rgb_camXs, axes=[0, 3, 1, 2])
            empty_rgb_camXs = empty_rgb_camXs[:,:3]
            empty_rgb_camXs = utils_improc.preprocess_color(empty_rgb_camXs)
            d['empty_rgb_camXs_raw'] = empty_rgb_camXs
        return d

    def __len__(self):
        return len(self.records)

def get_inputs():
    dataset_format = hyp.dataset_format
    all_set_inputs = {}
    for set_name in hyp.set_names:
        if hyp.sets_to_run[set_name]:
            data_path = hyp.data_paths[set_name]
            shuffle = hyp.shuffles[set_name]
            if dataset_format == 'tf':
                all_set_inputs[set_name] = TFRecordDataset(dataset_path=data_path, shuffle=shuffle)
            elif dataset_format == 'npz':
                if hyp.do_debug:
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                    shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                else:
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                    shuffle=shuffle, batch_size=hyp.B, num_workers=1, pin_memory=True, drop_last=True)
            else:
                assert False #what is the data format?
    return all_set_inputs


def get_custom_inputs(filenames):
    if hyp.do_debug:
        all_set_inputs = torch.utils.data.DataLoader(dataset=NpzCustomRecordDataset(filenames=filenames, shuffle=hyp.max.shuffle), \
        shuffle=hyp.max.shuffle, batch_size=hyp.max.B, num_workers=0, pin_memory=True, drop_last=True)
    else:
        all_set_inputs = torch.utils.data.DataLoader(dataset=NpzCustomRecordDataset(filenames=filenames, shuffle=hyp.max.shuffle), \
        shuffle=hyp.max.shuffle, batch_size=hyp.max.B, num_workers=1, pin_memory=True, drop_last=True)
    return all_set_inputs
