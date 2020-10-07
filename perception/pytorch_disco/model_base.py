import time
import numpy as np
import hyperparams as hyp
import torch
import hardPositiveMiner
from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from backend import inputs as load_inputs
from torchvision import datasets, transforms
from DoublePool import DoublePool_O
from DoublePool import MOC_DICT,MOC_QUEUE_NORMAL
from DoublePool import ClusterPool
from DoublePool import DetPool
import utils_basic
import socket
import time
import torch.nn.functional as F
import pickle
import utils_eval
import utils_improc
import utils_basic
import ipdb
st = ipdb.set_trace
from collections import defaultdict
import cross_corr
np.set_printoptions(precision=2)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes


class Model(object):
    def __init__(self, checkpoint_dir, log_dir):
        print('------ CREATING NEW MODEL ------')
        print(hyp.name)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.all_inputs = inputs.get_inputs()
        print("------ Done getting inputs ------")

        if hyp.do_orientation:
            self.mbr16 = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE, hyp.BOX_SIZE, hyp.BOX_SIZE)
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(32,32,32)
            self.hpm = None
        else:
            self.mbr16 = None
            self.hpm = None

        if hyp.self_improve_iterate:
            self.pool_det = DetPool(hyp.det_pool_size)
       
        self.device = torch.device("cuda")

    def init_model_k(self, model_q, model_k):
        param_q = model_q.state_dict()
        model_k.load_state_dict(param_q)

    def infer(self):
        pass

    def ml_loss(self,emb_e,emb_g_key,pool):
        vox_emb, vox_emb_key, classes_key = utils_eval.subsample_embs_voxs_positive(emb_e,emb_g_key, classes= None)

        vox_emb_key_og = vox_emb_key

        vox_emb_key = vox_emb_key.permute(0,2,1)
        vox_emb = vox_emb.permute(0,2,1)

        B,_,_ = vox_emb.shape

        emb_q = vox_emb.reshape(-1,hyp.feat_dim)
        emb_k = vox_emb_key.reshape(-1,hyp.feat_dim)

        N = emb_q.shape[0]

        emb_k = F.normalize(emb_k,dim=1)
        emb_q = F.normalize(emb_q,dim=1)

        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))

        queue_neg = torch.stack(pool.fetch())

        K = queue_neg.shape[0]

        queue_neg = F.normalize(queue_neg,dim=1)

        l_neg = torch.mm(emb_q, queue_neg.T)
        l_pos = l_pos.view(N, 1)
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(self.device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        temp = 0.07
        emb_loss = cross_entropy_loss(logits/temp, labels)
        return emb_loss


    def go(self):
        self.start_time = time.time()
        self.infer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
        print("------ Done creating models ------")

        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        if hyp.self_improve_once or hyp.filter_boxes:
          self.model.detnet_target.load_state_dict(self.model.detnet.state_dict())

        if hyp.sets_to_run["test"]:
            self.start_iter = 0 
        print("------ Done loading weights ------")
        if hyp.self_improve_iterate:
            exp_set_name = "expectation"
            exp_writer = SummaryWriter(self.log_dir + f'/{exp_set_name}', max_queue=MAX_QUEUE, flush_secs=60)                        
            max_set_name = "maximization_det"
            max_writer = SummaryWriter(self.log_dir + f'/{max_set_name}', max_queue=MAX_QUEUE, flush_secs=60)
            
            self.eval_steps = 0
            self.total_exp_iters = 0 
            hyp.max.B = hyp.B

            inputs = self.all_inputs['train']
            exp_log_freq = hyp.exp_log_freq
            exp_loader = iter(inputs)

            self.max_steps = 0
            self.total_max_iters = 0
            while True:
                hyp.exp_do = True
                hyp.exp_done = False                
                hyp.max_do = False  
                if hyp.exp_do:
                    start_time = time.time()
                    print("EVAL MODE: ")
                    self.eval_steps += 1
                    for step in range(hyp.exp_max_iters):
                        if step % len(inputs) == 0:
                            exp_loader = iter(inputs)                        

                        self.total_exp_iters += 1
                        iter_start_time = time.time()
                        log_this = np.mod(self.eval_steps,exp_log_freq) == 0
                        total_time, read_time, iter_time = 0.0, 0.0, 0.0
                        read_start_time = time.time()
                        
                        try:
                            feed = next(exp_loader)
                        except StopIteration:
                            print("ERROR")
                            exp_loader = iter(inputs)
                            feed = next(exp_loader)

                        feed_cuda = {}

                        tree_seq_filename = feed.pop('tree_seq_filename')
                        filename_e = feed.pop('filename_e')
                        filename_g = feed.pop('filename_g')
                
                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        read_time = time.time() - read_start_time
                        feed_cuda['tree_seq_filename'] = tree_seq_filename
                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g
                        feed_cuda['writer'] = exp_writer
                        feed_cuda['global_step'] = self.total_exp_iters
                        feed_cuda['log_freq'] = exp_log_freq
                        feed_cuda['set_name'] = exp_set_name

                        self.model.eval()
                        with torch.no_grad():
                            loss, results = self.model(feed_cuda)


                        loss_vis = loss.cpu().item()
                        summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                               global_step=feed_cuda['global_step'],
                                               set_name=feed_cuda['set_name'],
                                               log_freq=feed_cuda['log_freq'],
                                               fps=8)                        
                        if results['filenames_g'] is not None:
                            filenames_g = results['filenames_g']
                            filenames_e = results['filenames_e']
                            feat_masks =  results['featR_masks']
                            boxes =  results['filtered_boxes']
                            gt_boxes = results["gt_boxes"]
                            scores =  results['scores']
                            gt_scores = results["gt_scores"]
                            filenames = np.stack([filenames_g,filenames_e],axis=1)
                            self.pool_det.update(feat_masks, boxes, gt_boxes, scores, gt_scores, filenames)
                        else:
                            print("Filenames_g is None")
                        
                        print("Expectation: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); pool_fill: [%4d/%4d] Global_Steps: %4d"%(
                                                                                            hyp.name,
                                                                                            step,
                                                                                            hyp.exp_max_iters,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            read_time,
                                                                                            self.pool_det.num,
                                                                                            self.pool_det.pool_size,
                                                                                            self.eval_steps))
                if self.pool_det.is_full():
                    hyp.exp_do = False
                    hyp.exp_done = True                
                    hyp.max_do = False
                else:
                    hyp.exp_do = True
                    hyp.exp_done = False                
                    hyp.max_do = False
                if hyp.exp_done:
                    filenames, feat_masks, boxes, gt_boxes, scores,gt_scores = self.pool_det.fetch()
                    final_filenames = np.stack(filenames)
    
                    hyp.exp_do = False
                    hyp.exp_done = False
                    hyp.max_do = True

                if hyp.max_do:
                    max_loader = load_inputs.get_custom_inputs(final_filenames)
                    max_loader_iter = iter(max_loader)
                    max_log_freq = hyp.maxm_log_freq
                    self.model.train()
                    self.max_steps += 1
                    print("MAX MODE: ")
                    boxes = torch.stack(boxes)
                    scores = torch.stack(scores)
                    feat_masks = torch.stack(feat_masks)
                    gt_boxes = torch.stack(gt_boxes)
                    gt_scores = torch.stack(gt_scores)
                    for step in range(hyp.maxm_max_iters):
                        self.total_max_iters += 1
                        iter_start_time = time.time()
                        try:
                            feed = next(max_loader_iter)
                        except StopIteration:
                            print("ERROR")
                            max_loader_iter = iter(max_loader)
                            feed = next(max_loader_iter)

                        feed_cuda = {}                        
                        filename_e = feed.pop('filename_e')
                        filename_g = feed.pop('filename_g')
                        tree_seq_filename = feed.pop('tree_seq_filename')
                        index_val = feed.pop('index_val')

                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g
                        feed_cuda['tree_seq_filename'] = tree_seq_filename
                        feed_cuda['writer'] = max_writer
                        feed_cuda['global_step'] = self.total_max_iters
                        feed_cuda['log_freq'] = max_log_freq
                        feed_cuda['set_name'] = max_set_name
                        
                        
                        feed_cuda["sudo_gt_boxes"] = boxes[index_val]
                        feed_cuda["sudo_gt_scores"] = scores[index_val]
                        feed_cuda["feat_mask"] = feat_masks[index_val]

                        feed_cuda["gt_boxes"] = gt_boxes[index_val]
                        feed_cuda["gt_scores"] = gt_scores[index_val]

                        iter_start_time = time.time()

                        final_filenames[index_val[0]][0].split("/")[-2][:-2] == filename_e[0].split("/")[-1][:-5]
                        loss, results = self.model(feed_cuda)

                        total_loss = loss
                        backprop_start_time = time.time()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        backprop_time = time.time()- backprop_start_time
                        iter_time = time.time()- iter_start_time
                        total_time = time.time()-start_time
                        print("Predicted Maximization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Global_Steps: %4d"% (hyp.name,
                                                                                            step,
                                                                                            hyp.maxm_max_iters,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            backprop_time,
                                                                                            total_loss,self.max_steps))
                
                saverloader.save(self.model, self.checkpoint_dir, self.total_max_iters, self.optimizer)

        else:
            set_nums = []
            set_names = []
            set_inputs = []
            set_writers = []
            set_log_freqs = []
            set_do_backprops = []
            set_dicts = []
            set_loaders = []



            for set_name in hyp.set_names:
                if hyp.sets_to_run[set_name]:
                    set_nums.append(hyp.set_nums[set_name])
                    set_names.append(set_name)
                    set_inputs.append(self.all_inputs[set_name])
                    set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                    set_log_freqs.append(hyp.log_freqs[set_name])
                    set_do_backprops.append(hyp.sets_to_backprop[set_name])
                    set_dicts.append({})
                    set_loaders.append(iter(set_inputs[-1]))

            for step in range(self.start_iter+1, hyp.max_iters+1):
                for i, (set_input) in enumerate(set_inputs):
                    if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                        if hyp.add_det_boxes:
                            st()
                            print("entire iteration over dataset done")
                        set_loaders[i] = iter(set_input)

                for (set_num,
                        set_name,
                        set_input,
                        set_writer,
                        set_log_freq,
                        set_do_backprop,
                        set_dict,
                        set_loader
                        ) in zip(
                        set_nums,
                        set_names,
                        set_inputs,
                        set_writers,
                        set_log_freqs,
                        set_do_backprops,
                        set_dicts,
                        set_loaders
                        ):   

                    log_this = np.mod(step, set_log_freq)==0
                    total_time, read_time, iter_time = 0.0, 0.0, 0.0
                    if log_this or set_do_backprop or hyp.break_constraint:
                    
                        read_start_time = time.time()

                        feed = next(set_loader)

                        feed_cuda = {}
                        if hyp.do_clevr_sta:
                            tree_seq_filename = feed.pop('tree_seq_filename')
                            filename_e = feed.pop('filename_e')
                            filename_g = feed.pop('filename_g')
                        if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                            classes = feed.pop('classes')

                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        read_time = time.time() - read_start_time


                        if hyp.do_clevr_sta:
                            feed_cuda['tree_seq_filename'] = tree_seq_filename
                            feed_cuda['filename_e'] = filename_e
                            feed_cuda['filename_g'] = filename_g
                        
                        if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                            classes = np.transpose(np.array(classes))
                            feed_cuda['classes'] = classes

                        feed_cuda['writer'] = set_writer
                        feed_cuda['global_step'] = step
                        feed_cuda['set_num'] = set_num
                        feed_cuda['set_name'] = set_name
                        iter_start_time = time.time()

                        if set_do_backprop:
                            start_time =  time.time()                            
                            self.model.train()
                            loss, results = self.model(feed_cuda)
                            if hyp.profile_time:
                                print("forwardpass time",time.time()-start_time)
                        else:
                            self.model.eval()
                            with torch.no_grad():
                                loss, results = self.model(feed_cuda)
                        
                        loss_vis = loss.cpu().item()

                        summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                               global_step=feed_cuda['global_step'],
                                               set_name=feed_cuda['set_name'],
                                               fps=8)
                        summ_writer.summ_scalar('loss',loss_vis)
                        
                        if set_do_backprop and hyp.sudo_backprop:
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        hyp.sudo_backprop = True

                        iter_time = time.time()-iter_start_time
                        total_time = time.time()-self.start_time

                        print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                            step,
                                                                                            hyp.max_iters,
                                                                                            total_time,
                                                                                            read_time,
                                                                                            iter_time,
                                                                                            loss_vis,
                                                                                            set_name))
                if np.mod(step, hyp.snap_freq) == 0:
                    saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

            for writer in set_writers: #close writers to flush cache into file
                writer.close()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
