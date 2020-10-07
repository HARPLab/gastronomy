import time
from backend import inputs as load_inputs
import numpy as np
import hyperparams as hyp
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from backend import saverloader

# from backend.double_pool import DoublePool
from torchvision import datasets, transforms
from DoublePool import DoublePool_O_f
from DoublePool import MOC_DICT,MOC_QUEUE_NORMAL
import torch.nn.functional as F
import pickle
import utils_improc
import utils_eval
import cross_corr
import utils_improc
import utils_basic
import ipdb
import hardPositiveMiner
st = ipdb.set_trace
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
        self.all_inputs = load_inputs.get_inputs()
        print("------ Done getting inputs ------")
        self.recalls = [10, 20, 30]
        if hyp.exp.do_debug or hyp.do_debug or hyp.low_dict_size:
            self.pool_size = 20
        else:
            self.pool_size = 1000
        self.eval_dicts = {}
        F = 3
        self.pool3D_e = DoublePool_O_f(self.pool_size)
        self.pool3D_g = DoublePool_O_f(self.pool_size)                        
        self.precision3D = np.nan*np.array([0.0, 0.0, 0.0], np.float32)
        self.neighbors3D = np.zeros((F*10, F*11, 3), np.float32)
        self.device = torch.device("cuda")
        if hyp.max.hardmining or hyp.hard_eval or hyp.hard_vis:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE-2*hyp.max.margin, hyp.BOX_SIZE-2*hyp.max.margin, hyp.BOX_SIZE-2*hyp.max.margin)
            self.mbr16 = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE, hyp.BOX_SIZE, hyp.BOX_SIZE)
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(32,32,32)
            self.hpm = hardPositiveMiner.HardPositiveMiner(self.mbr,self.mbr16,self.mbr_unpr)
        elif hyp.do_orientation:
            self.mbr16 = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE, hyp.BOX_SIZE, hyp.BOX_SIZE)
            self.hpm = None
        else:
            self.hpm = None

        if hyp.max.hard_moc:
            self.poolvox_moc = MOC_QUEUE_NORMAL(hyp.max.hard_moc_qsize)

        if hyp.emb_moc.do:
            if hyp.emb_moc.normal_queue:
                self.poolvox_moc = MOC_QUEUE_NORMAL(hyp.emb_moc.max_pool_indices)
            else:
                self.poolvox_moc = MOC_DICT()
        
    def infer(self):
        print('nothing to infer!')

    def momentum_update(self,model_q, model_k, beta = 0.999):
        param_k = model_k.state_dict()
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        model_k.load_state_dict(param_k)

    def init_model_k(self,model_q, model_k):
        param_q = model_q.state_dict()
        model_k.load_state_dict(param_q)

    def go(self):
        self.start_time = time.time()
        self.infer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
        print("------ Done creating models ------")
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        # st()
        if hyp.emb_moc.do or hyp.max.hard_moc:
            self.init_model_k(self.model,self.model_key)

        print("------ Done loading weights ------")
        # filling up the pool
        self.total_steps = 0
        ex = hyp.exp
        maxm = hyp.max
        emb_moc = hyp.emb_moc

        if ex.do:
            ex_set_name = "expectation"
            ex_writer = SummaryWriter(self.log_dir + f'/{ex_set_name}', max_queue=MAX_QUEUE, flush_secs=60)

        if maxm.do:
            if maxm.hardmining:
                max_set_name_p = "maximization_pos"
                max_writer_p = SummaryWriter(self.log_dir + f'/{max_set_name_p}', max_queue=MAX_QUEUE, flush_secs=60)
                max_set_name_neg = "maximization_neg"
                max_writer_neg = SummaryWriter(self.log_dir + f'/{max_set_name_neg}', max_queue=MAX_QUEUE, flush_secs=60)                
            else:
                max_set_name_p = "maximization_p"
                max_writer_p = SummaryWriter(self.log_dir + f'/{max_set_name_p}', max_queue=MAX_QUEUE, flush_secs=60)
                max_set_name_g = "maximization_g"
                max_writer_g = SummaryWriter(self.log_dir + f'/{max_set_name_g}', max_queue=MAX_QUEUE, flush_secs=60)
        if emb_moc.do or hyp.max.hard_moc:
            self.total_embmoc_iters = 0
            embmoc_set_name = "emb_moc"
            embmoc_writer = SummaryWriter(self.log_dir + f'/{ex_set_name}', max_queue=MAX_QUEUE, flush_secs=60)

        if ex.do:
            self.eval_steps = 0
            self.total_exp_iters = 0 
            if ex.tdata:
                inputs = self.all_inputs['train']
            else:
                inputs = self.all_inputs['val']
            ex_log_freq = ex.log_freq
            ex_loader = iter(inputs)
        if maxm.do:
            self.max_steps = 0
            self.total_max_iters = 0
            self.total_max_iters_g = 0
            self.total_max_iters_p = 0
            # if self.start_iter  != 0:
            #     self.max_steps = self.start_iter
            #     self.total_max_iters = self.start_iter
            #     self.total_max_iters_g = self.start_iter
            #     self.total_max_iters_p = self.start_iter  

        #allowing to load the same model dynamically
        hyp.feat_init = hyp.name
        if emb_moc.do or  hyp.max.hard_moc:
            maxm.do = False
            start_time = time.time()
            step =  0
            while True:
                step += 1
                if step % len(inputs) == 0:
                    embmoc_loader = iter(inputs)
                self.total_embmoc_iters += 1

                iter_start_time = time.time()
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                read_start_time = time.time()
                try:
                    feed = next(ex_loader)
                except StopIteration:
                    ex_loader = iter(inputs)
                    feed = next(ex_loader)
                read_time = time.time() - read_start_time
                feed_cuda = {}
                tree_seq_filename = feed.pop('tree_seq_filename')
                filename_e = feed.pop('filename_e')
                filename_g = feed.pop('filename_g')
                if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                    classes = feed.pop('classes')
                for k in feed:
                    feed_cuda[k] = feed[k].cuda(non_blocking=True).float()
                read_time = time.time() - read_start_time
                feed_cuda['tree_seq_filename'] = tree_seq_filename
                feed_cuda['filename_e'] = filename_e
                feed_cuda['filename_g'] = filename_g
                feed_cuda['writer'] = embmoc_writer
                if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                    classes = np.transpose(np.array(classes))
                    feed_cuda['classes'] = classes
                feed_cuda['global_step'] = self.total_embmoc_iters
                feed_cuda['log_freq'] = 100
                feed_cuda['set_name'] = "queue_init"                
                self.model_key.eval()
                with torch.no_grad():                    
                    loss, results_key = self.model_key(feed_cuda)                
                classes_key = results_key['classes']
                emb3D_e_key = results_key['emb3D_e_R']
                emb3D_g_key = results_key['emb3D_g_R']
                vox_emb_key, classes_key = utils_eval.subsample_embs_voxs(emb3D_e_key, emb3D_g_key, classes= classes_key)
                total_time = time.time() - start_time
                iter_time = time.time()- iter_start_time
                self.poolvox_moc.update(vox_emb_key,classes_key)
                if emb_moc.normal_queue or  hyp.max.hard_moc:
                    queue_size = self.poolvox_moc.num
                else:
                    queue_size = len(self.poolvox_moc.fetch().keys())
                print("Queue Initialization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); Keys: %4d"%(
                                                                                        hyp.name,
                                                                                        step,
                                                                                        emb_moc.max_iters_init,
                                                                                        total_time,
                                                                                        iter_time,
                                                                                        read_time,queue_size))
                if self.poolvox_moc.is_full():
                    break
        while True:
            if not emb_moc.own_data_loader:
                if ex.do:
                    exp_done = False
                    start_time = time.time()
                    print("EVAL MODE: ")
                    self.eval_steps += 1
                    maxm.do = False
                    # st()
                    if self.max_steps > 0:
                        print("LOADING WEIGHTS EXPECTATION:")
                        saverloader.load_weights(self.model, self.optimizer)
                    for step in range(ex.max_iters):
                        if step % len(inputs) == 0:
                            ex_loader = iter(inputs)
                        self.total_exp_iters += 1
                        iter_start_time = time.time()
                        log_this = np.mod(self.eval_steps,ex_log_freq) == 0
                        total_time, read_time, iter_time = 0.0, 0.0, 0.0
                        read_start_time = time.time()
                        try:
                            feed = next(ex_loader)
                        except StopIteration:
                            ex_loader = iter(inputs)
                            feed = next(ex_loader)
                        feed_cuda = {}
                        tree_seq_filename = feed.pop('tree_seq_filename')
                        
                        filename_e = feed.pop('filename_e')
                        filename_g = feed.pop('filename_g')
                        
                        if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                            classes = feed.pop('classes')

                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        read_time = time.time() - read_start_time

                        if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                            classes = np.transpose(np.array(classes))
                            feed_cuda['classes'] = classes

                        feed_cuda['tree_seq_filename'] = tree_seq_filename
                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g

                        feed_cuda['writer'] = ex_writer
                        feed_cuda['global_step'] = self.total_exp_iters
                        feed_cuda['log_freq'] = ex_log_freq
                        feed_cuda['set_name'] = ex_set_name
                        # self.model.eval()
                        self.model.eval()
                        with torch.no_grad():                    
                            loss, results = self.model(feed_cuda)
                        loss_vis = loss.cpu().item()
                        summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                               global_step=feed_cuda['global_step'],
                                               set_name=feed_cuda['set_name'],
                                               log_freq=feed_cuda['log_freq'],
                                               fps=8)
                        
                        rgb = results['rgb'].cpu().detach().numpy()
                        rgb = np.transpose(rgb, (0, 2, 3, 1))
                        visual2D = rgb
                        
                        classes = results['classes']

                        filenames_g = results['filenames_g']
                        filenames_e = results['filenames_e']
                        # e used for querying
                        # g for retrieval
                        emb3D_e = results['emb3D_e_R']
                        emb3D_g = results['emb3D_g_R']
                        valid3D = results['valid3D']

                        unp_visRs = results['unp_visRs']

                        filenames = np.stack([filenames_g,filenames_e],axis=1)                    
                        # samps = 100 if pool3D_e'].is_full() else 100
                        # st()
                        # emb3D_e, emb3D_g, visual2D, classes, filenames = utils_eval.subsample_embs_3D_o_cuda(emb3D_e, emb3D_g, valid3D, visual2D, classes= classes,filenames=filenames)
                        if ex.no_update:
                            if self.pool3D_e.num  < self.pool_size:
                                self.pool3D_e.update(emb3D_e, visual2D, classes, filenames_e, unp_visRs)
                                self.pool3D_g.update(emb3D_g, visual2D, classes, filenames_g, unp_visRs) # No need to save again

                        else:
                            self.pool3D_e.update(emb3D_e, visual2D, classes, filenames_e,unp_visRs)
                            self.pool3D_g.update(emb3D_g, visual2D, classes, filenames_g, unp_visRs) # No need to save again

                        already_done = exp_done
                        precision3D, neighbors3D,ranks, exp_done,filenames = utils_eval.compute_precision_o_cuda(
                            self.pool3D_e,
                            self.pool3D_g,
                            hyp.eval_compute_freq,
                            self.hpm,
                            self.mbr16,
                            recalls=self.recalls,
                            pool_size=self.pool_size,
                            steps_done=step)
                        if already_done:
                            exp_done = True
                        files_e,files_g = filenames

                        neighbors3D = torch.from_numpy(neighbors3D).float().permute(2,0,1)

                        ns = "retrieval/"

                        for key,precisions in precision3D.items():
                            if "average" in precisions:
                                average = precisions.pop("average")
                                summ_writer.summ_scalar(ns + 'precision3D_{:02d}_avg'.format(int(key)),average)
                            if hyp.eval_recall_summ_o:
                                summ_writer.summ_scalars(ns + 'precision3D_{:02d}'.format(int(key)),dict(precisions))
                        summ_writer.summ_rgb(ns + 'neighbors3D', utils_improc.preprocess_color(neighbors3D).unsqueeze(0))
                        iter_time = time.time()- iter_start_time
                        total_time = time.time()-start_time


                        print("Expectation: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); pool_fill: [%4d/%4d] Global_Steps: %4d"%(
                                                                                            hyp.name,
                                                                                            step,
                                                                                            ex.max_iters,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            read_time,
                                                                                            self.pool3D_e.num,
                                                                                            self.pool3D_e.pool_size,
                                                                                            self.eval_steps))

                    # st()
                    if exp_done:
                        maxm.do = True
                        export_ranks = True
                    else:
                        maxm.do = False
                        export_ranks = False
                # st()
                if export_ranks and not maxm.hardmining:
                    if hyp.gt:
                        (emb_e, vis_e, class_e, files_e) = self.pool3D_g.fetch()                    
                        (emb_g, vis_g, class_g, files_g) = self.pool3D_g.fetch()
                        files_g_np = np.array(files_g)
                        files_e_np = np.array(files_e)
                        all_files = []
                        for index,class_val_e in enumerate(class_e):
                            if class_val_e not in ["cylinder","cube","sphere"]:
                                indexes_same_class = np.where(np.array(class_g) == class_val_e)[0]
                                files_g_np_f = files_g_np[indexes_same_class]
                                file_e_np = np.array([files_e_np[index]])
                                num_ret = files_g_np_f.shape[0]
                                files_e_np_r = np.repeat(file_e_np,num_ret)
                                files_group = np.stack([files_g_np_f,files_e_np_r],axis=-1)
                                all_files.append(files_group)
                            else:
                                st()
                        final_filenames = np.concatenate(all_files,axis=0)
                    else:
                        num_exms = self.pool3D_e.num
                        perm = np.random.permutation(num_exms)
                        perm = perm[:]                    
                        top_retrieval = ranks[perm][:,0:5]
                        _,retrieved_num = top_retrieval.shape
                        filenames_e = np.array(self.pool3D_e.fetch()[3])
                        filenames_g = np.array(self.pool3D_g.fetch()[3])
                        filenames_e_p = filenames_e[perm]
                        filenames_e_p_e = np.repeat(np.expand_dims(filenames_e_p,axis=1),retrieved_num,axis=1)
                        filenames_g_p = filenames_g[top_retrieval]
                        filenames = np.stack([filenames_e_p_e,filenames_g_p],axis=2)
                        final_filenames = np.reshape(filenames,[-1,2])
                        final_filenames = final_filenames[np.random.permutation(final_filenames.shape[0])]
            else:
                maxm.do = True
                final_filenames = pickle.load(open("all_file_pairs.p","rb"))
            if maxm.do:
                if maxm.hardmining_gt:
                    (emb_e, vis_e, class_e, files_e) = self.pool3D_g.fetch()                    
                    (emb_g, vis_g, class_g, files_g) = self.pool3D_g.fetch()
                    num_total = self.pool3D_g.num
                    files_g_np = np.array(files_g)
                    files_e_np = np.array(files_e)
                    all_ranks = []
                    for index,class_val_e in enumerate(class_e):
                        if class_val_e not in ["cylinder","cube","sphere"]:
                            temp = np.zeros([num_total])
                            indexes_same_class = np.where(np.array(class_g) == class_val_e)[0]
                            temp[indexes_same_class] = 1
                            temp_rank = np.flip(np.argsort(temp))
                            all_ranks.append(temp_rank)
                        else:
                            st()
                    ranks = np.stack(all_ranks,axis=0)
                if maxm.hardmining:
                    start_time = time.time()
                    mining_start_time = time.time()
                    self.max_steps += 1
                    featquery, perm = self.hpm.extractPatches(self.pool3D_e)
                    # positive

                    topkImg, topkScale, topkValue, topkW, topkH , topkD , topPoolgFnames, fname_e, ranks_pos, topkR   = self.hpm.RetrievalRes(self.pool3D_e, ranks, self.pool3D_g, featquery, perm)
                    # topkImg is the indexes of ranks[perm] --> so the 3d tensors used are  emb_e[ranks[perm][topkImg]]

                    nbImgEpoch = maxm.nbImgEpoch

                    batchSize = maxm.B
                    # st()
                    max_log_freq = maxm.log_freq

                    iterEpoch = nbImgEpoch // batchSize

                    # topkImg is the indexes of ranks[perm] --> so the 3d tensors used are  emb_e[ranks[perm][topkImg]] , 
                    posPair, fnamesForDataLoader = self.hpm.TrainPair(nbImgEpoch, topkImg, topkScale, topkW, topkH,  topkD, self.pool3D_g, self.pool3D_e, ranks_pos, topPoolgFnames, topkR)

                    max_loader_og = load_inputs.get_custom_inputs(fnamesForDataLoader)
                    # negative
                    
                    if not maxm.hard_moc:
                        topkImg_neg, topkScale_neg, topkValue_neg, topkW_neg, topkH_neg , topkD_neg , topPoolgFnames_neg, fname_e_neg, ranks_neg, topkR_neg = self.hpm.RetrievalRes(self.pool3D_e, ranks, self.pool3D_g, featquery, perm, negativeSamples = True)
                        posPair_neg, fnamesForDataLoader_neg = self.hpm.TrainPair(nbImgEpoch, topkImg_neg, topkScale_neg, topkW_neg, topkH_neg,  topkD_neg, self.pool3D_g, self.pool3D_e, ranks_neg, topPoolgFnames_neg, topkR_neg)
                        max_loader_og_neg = load_inputs.get_custom_inputs(fnamesForDataLoader_neg)
                    mining_time = time.time()- mining_start_time
                    # posPairEpochIndex = hpm.DataShuffle(posPair, batchSize)

                    unpRs_e = self.pool3D_e.fetchUnpRs()
                    unpRs_g = self.pool3D_g.fetchUnpRs()                    


                    obj_emb_e, ob_visual_2d_e, _, _,_ = self.pool3D_e.fetch()
                    obj_emb_g, ob_visual_2d_g, _, _, _ = self.pool3D_g.fetch()

                    self.model.train()
                    maxm.max_iters = maxm.max_epochs * iterEpoch
                    for num_epoch in range(maxm.max_epochs):
                        if maxm.hard_moc:
                            #  doing moc
                            max_loader = iter(max_loader_og)
                            # max_loader_neg = iter(max_loader_og_neg)
                            for j_ in range(iterEpoch) :
                                iter_start_time = time.time()
                                step = num_epoch * iterEpoch + j_
                                anchorsBatch = []
                                posBatch = []

                                indexes_done = j_ * hyp.max.B
                                self.total_max_iters += 1                    
                                self.total_max_iters_p += 1

                                maxm.predicted_matching = True
                                feed = next(max_loader)
                                feed_cuda = {}

                                filename_e = feed.pop('filename_e')
                                filename_g = feed.pop('filename_g')
                                tree_seq_filename = feed.pop('tree_seq_filename')
                                
                                if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                                    classes_e = feed.pop('object_category_names_e')
                                    classes_g = feed.pop('object_category_names_g')

                                for k in feed:
                                    feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                                if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix":
                                    classes_e = np.transpose(np.array(classes_e))
                                    feed_cuda['classes_e'] = classes_e
                                    classes_g = np.transpose(np.array(classes_g))
                                    feed_cuda['classes_g'] = classes_g

                                feed_cuda['filename_e'] = filename_e
                                feed_cuda['filename_g'] = filename_g
                                feed_cuda['tree_seq_filename'] = tree_seq_filename
                                feed_cuda['writer'] = max_writer_p
                                feed_cuda['global_step'] = self.total_max_iters_p
                                feed_cuda['log_freq'] = max_log_freq
                                feed_cuda['set_name'] = max_set_name_p

                                summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                                       global_step=feed_cuda['global_step'],
                                                       set_name=feed_cuda['set_name'],
                                                       log_freq=feed_cuda['log_freq'],
                                                       fps=8)
                                loss, results = self.model(feed_cuda)

                                self.model_key.eval()
                                with torch.no_grad():                               
                                    loss_key, results_key = self.model_key(feed_cuda)

                                emb3D_e = results['emb3D_e_R']
                                emb3D_g = results_key['emb3D_g_R']
                                # do only 0th example in batch
                                summ_writer.summ_hardmines("positive",[posPair,[topkImg,topkD,topkH,topkW,topkR],ranks_pos,[unpRs_e,unpRs_g],[ob_visual_2d_e,ob_visual_2d_g],indexes_done],self.mbr_unpr)
                                summ_writer.summ_best_orientation("target_bestR_query",[posPair,[topkImg,topkD,topkH,topkW,topkR],ranks_pos,[unpRs_e,unpRs_g,obj_emb_e,obj_emb_g],indexes_done],self.mbr16,self.mbr_unpr)

                                for k_ in range(hyp.max.B):
                                    current_index = indexes_done + k_                     
                                    
                                    emb3D_e_current = emb3D_e[k_]
                                    emb3D_g_current = emb3D_g[k_]
                                    filename_e_current = filename_e[k_].split("/")[-1][:-5]
                                    filename_g_current = filename_g[k_].split("/")[-1][:-5]
                                    fnamesForDataLoader_current_pair = fnamesForDataLoader[current_index]

                                    fname_e_current_pair = fnamesForDataLoader_current_pair[0].split("/")[-2][:-2]
                                    fname_g_current_pair = fnamesForDataLoader_current_pair[1].split("/")[-2][:-2]
                                    anchors,pos_samples = self.hpm.Pos_Examples([emb3D_e_current.unsqueeze(0),emb3D_g_current.unsqueeze(0)], posPair, current_index, topkImg, topkScale, topkD, topkH, topkW, topkR)
                                    anchorsBatch.append(anchors)
                                    posBatch.append(pos_samples)
                                
                                posBatch = torch.cat(posBatch, dim=0)
                                anchorsBatch = torch.cat(anchorsBatch, dim=0)

                                total_loss = 0.0 
                                emb_q = anchorsBatch.reshape(-1,hyp.feat_dim)
                                emb_k = posBatch.reshape(-1,hyp.feat_dim)

                                N = emb_q.shape[0]

                                emb_k = F.normalize(emb_k,dim=1)
                                emb_q = F.normalize(emb_q,dim=1)

                                l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))
                                queue_neg = torch.stack(self.poolvox_moc.fetch())

                                K = queue_neg.shape[0]

                                queue_neg = F.normalize(queue_neg,dim=1)

                                l_neg = torch.mm(emb_q, queue_neg.T)
                                l_pos = l_pos.view(N, 1)
                                logits = torch.cat([l_pos, l_neg], dim=1)
                                
                                labels = torch.zeros(N, dtype=torch.long)
                                labels = labels.to(self.device)

                                cross_entropy_loss = torch.nn.CrossEntropyLoss()
                                temp = 0.07
                                loss = cross_entropy_loss(logits/temp, labels)
                                total_loss += loss                                                                

                                ## make sure that gradient is not zero
                                backprop_start_time = time.time()
                                self.optimizer.zero_grad()
                                total_loss.backward()
                                self.optimizer.step()
                                self.momentum_update(self.model,self.model_key)
                                self.poolvox_moc.update(emb_k.unsqueeze(1))
                                summ_writer.summ_scalar('loss', total_loss.cpu().item())
                                iter_time = time.time()- iter_start_time
                                total_time = time.time()-start_time
                                print("Predicted Maximization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Global_Steps: %4d"% (hyp.name,
                                                                                                        step,
                                                                                                        maxm.max_iters,
                                                                                                        total_time,
                                                                                                        mining_time,
                                                                                                        iter_time,
                                                                                                        loss,self.max_steps))

                        else:
                            max_loader = iter(max_loader_og)
                            max_loader_neg = iter(max_loader_og_neg)
                            
                            for j_ in range(iterEpoch) :
                                iter_start_time = time.time()
                                step = num_epoch * iterEpoch + j_
                                posSimilarityBatch = []
                                indexes_done = j_ * hyp.max.B
                                
                                self.total_max_iters += 1                    
                                self.total_max_iters_p += 1

                                maxm.predicted_matching = True

                                feed = next(max_loader)

                                feed_cuda = {}
                                filename_e = feed.pop('filename_e')
                                filename_g = feed.pop('filename_g')
                                tree_seq_filename = feed.pop('tree_seq_filename')

                                for k in feed:
                                    feed_cuda[k] = feed[k].cuda(non_blocking=True).float()
                                
                                feed_cuda['filename_e'] = filename_e
                                feed_cuda['filename_g'] = filename_g
                                feed_cuda['writer'] = max_writer_p
                                feed_cuda['global_step'] = self.total_max_iters_p
                                feed_cuda['log_freq'] = max_log_freq
                                feed_cuda['set_name'] = max_set_name_p
                                feed_cuda['tree_seq_filename'] = tree_seq_filename
                                # st()

                                summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                                       global_step=feed_cuda['global_step'],
                                                       set_name=feed_cuda['set_name'],
                                                       log_freq=feed_cuda['log_freq'],
                                                       fps=8)
                                loss, results = self.model(feed_cuda)

                                emb3D_e = results['emb3D_e_R']
                                emb3D_g = results['emb3D_g_R']
                                # do only 0th example in batch
                                summ_writer.summ_hardmines("positive",[posPair,[topkImg,topkD,topkH,topkW,topkR],ranks_pos,[unpRs_e,unpRs_g],[ob_visual_2d_e,ob_visual_2d_g],indexes_done],self.mbr_unpr)

                                for k_ in range(hyp.max.B):
                                    current_index = indexes_done + k_                     
                                    
                                    # only for visualization
                                    # end of visualization


                                    emb3D_e_current = emb3D_e[k_]
                                    emb3D_g_current = emb3D_g[k_]

                                    filename_e_current = filename_e[k_].split("/")[-1][:-5]
                                    filename_g_current = filename_g[k_].split("/")[-1][:-5]
                                    fnamesForDataLoader_current_pair = fnamesForDataLoader[current_index]

                                    fname_e_current_pair = fnamesForDataLoader_current_pair[0].split("/")[-2][:-2]
                                    fname_g_current_pair = fnamesForDataLoader_current_pair[1].split("/")[-2][:-2]
                                    
                                    assert fname_e_current_pair == filename_e_current
                                    assert filename_g_current == fname_g_current_pair


                                    emb3D_current = torch.stack([emb3D_e_current,emb3D_g_current]).unsqueeze(0)
                                    posSimilarity = self.hpm.PosSimilarity(emb3D_current, posPair, current_index, topkImg, topkScale, topkD, topkH, topkW, topkR)
                                    posSimilarityBatch = posSimilarityBatch + posSimilarity

                                posSimilarityBatch = torch.cat(posSimilarityBatch, dim=0)

                                #negative
                                posSimilarityBatch_neg = []

                                feed_neg = next(max_loader_neg)

                                feed_cuda_neg = {}
                                filename_e_neg = feed_neg.pop('filename_e')
                                filename_g_neg = feed_neg.pop('filename_g')
                                tree_seq_filename = feed_neg.pop('tree_seq_filename')

                                for k in feed_neg:
                                    feed_cuda_neg[k] = feed_neg[k].cuda(non_blocking=True).float()

                                feed_cuda_neg['filename_e'] = filename_e_neg
                                feed_cuda_neg['filename_g'] = filename_g_neg
                                feed_cuda_neg['writer'] = max_writer_neg
                                feed_cuda_neg['global_step'] = self.total_max_iters_p
                                feed_cuda_neg['log_freq'] = max_log_freq
                                feed_cuda_neg['set_name'] = max_set_name_neg
                                feed_cuda_neg['tree_seq_filename'] = tree_seq_filename

                                summ_writer_neg = utils_improc.Summ_writer(writer=feed_cuda_neg['writer'],
                                                       global_step=feed_cuda_neg['global_step'],
                                                       set_name=feed_cuda_neg['set_name'],
                                                       log_freq=feed_cuda_neg['log_freq'],
                                                       fps=8)
                                loss_neg, results_neg = self.model(feed_cuda_neg)

                                emb3D_e_neg = results_neg['emb3D_e_R']
                                emb3D_g_neg = results_neg['emb3D_g_R']

                                # only taking the 0 th index in the batch
                                # only for visualization
                                summ_writer_neg.summ_hardmines("negative",[posPair_neg,[topkImg_neg,topkD_neg,topkH_neg,topkW_neg,topkR_neg],ranks_neg,[unpRs_e,unpRs_g],[ob_visual_2d_e,ob_visual_2d_g],indexes_done],self.mbr_unpr)
                                # end of visualization

                                for k_ in range(hyp.max.B):
                                    current_index_neg = indexes_done + k_

                                    emb3D_e_current_neg = emb3D_e_neg[k_]
                                    emb3D_g_current_neg = emb3D_g_neg[k_]

                                    filename_e_current_neg = filename_e_neg[k_].split("/")[-1][:-5]
                                    filename_g_current_neg = filename_g_neg[k_].split("/")[-1][:-5]
                                    fnamesForDataLoader_current_pair_neg = fnamesForDataLoader_neg[current_index_neg]

                                    fname_e_current_pair_neg = fnamesForDataLoader_current_pair_neg[0].split("/")[-2][:-2]
                                    fname_g_current_pair_neg = fnamesForDataLoader_current_pair_neg[1].split("/")[-2][:-2]

                                    assert fname_e_current_pair_neg == filename_e_current_neg
                                    assert filename_g_current_neg == fname_g_current_pair_neg

                                    emb3D_current_neg = torch.stack([emb3D_e_current_neg,emb3D_g_current_neg]).unsqueeze(0)
                                    posSimilarity_neg = self.hpm.PosSimilarity(emb3D_current_neg, posPair_neg, current_index_neg, topkImg_neg, topkScale_neg, topkD_neg, topkH_neg, topkW_neg,topkR_neg)
                                    posSimilarityBatch_neg = posSimilarityBatch_neg + posSimilarity_neg
                                
                                posSimilarityBatch_neg = torch.cat(posSimilarityBatch_neg, dim=0)
                                loss = torch.clamp(posSimilarityBatch_neg  + maxm.tripleLossThreshold - 1, min=0) + torch.clamp(maxm.tripleLossThreshold - posSimilarityBatch, min=0)
                                ## make sure that gradient is not zero
                                if (loss > 0).any() :
                                    loss = loss.mean()
                                    backprop_start_time = time.time()
                                    self.optimizer.zero_grad()
                                    loss.backward()
                                    self.optimizer.step()
                                    summ_writer.summ_scalar('loss', loss.cpu().item())
                                    iter_time = time.time()- iter_start_time
                                    total_time = time.time()-start_time
                                    print("Predicted Maximization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Global_Steps: %4d"% (hyp.name,
                                                                                                        step,
                                                                                                        maxm.max_iters,
                                                                                                        total_time,
                                                                                                        mining_time,
                                                                                                        iter_time,
                                                                                                        loss,self.max_steps))


                else:
                    max_loader = load_inputs.get_custom_inputs(final_filenames)
                    max_loader_iter = iter(max_loader)
                    max_log_freq = maxm.log_freq
                    self.model.train()
                    self.max_steps += 1
                    print("MAX MODE: ")
                    for step in range(maxm.max_iters):
                        for _ in range(maxm.p_max_iters):
                            self.total_max_iters += 1                    
                            self.total_max_iters_p += 1
                            maxm.predicted_matching = True
                            iter_start_time = time.time()

                            try:
                                feed = next(max_loader_iter)
                            except StopIteration:
                                max_loader_iter = iter(max_loader)
                                feed = next(max_loader_iter)

                            feed_cuda = {}                        
                            filename_e = feed.pop('filename_e')
                            filename_g = feed.pop('filename_g')

                            for k in feed:
                                feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                            feed_cuda['filename_e'] = filename_e
                            feed_cuda['filename_g'] = filename_g
                            feed_cuda['writer'] = max_writer_p
                            feed_cuda['global_step'] = self.total_max_iters_p
                            feed_cuda['log_freq'] = max_log_freq
                            feed_cuda['set_name'] = max_set_name_p
                            iter_start_time = time.time()
                            if hyp.emb_moc.do:
                                summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                                       global_step=feed_cuda['global_step'],
                                                       set_name=feed_cuda['set_name'],
                                                       log_freq=feed_cuda['log_freq'],
                                                       fps=8)
                                loss, results = self.model(feed_cuda)
                                self.model_key.eval()
                                with torch.no_grad():                               
                                    loss_key, results_key = self.model_key(feed_cuda)
                                
                                classes = results['classes']
                                emb3D_e = results['emb3D_e_R']
                                emb3D_g = results['emb3D_g_R']

                                classes_key = results_key['classes']
                                emb3D_e_key = results_key['emb3D_e_R']
                                emb3D_g_key = results_key['emb3D_g_R']
                                
                                vox_emb, vox_emb_key, classes_key = utils_eval.subsample_embs_voxs_positive(emb3D_e,emb3D_g_key, classes= classes)

                                vox_emb_key_og = vox_emb_key

                                vox_emb_key = vox_emb_key.permute(0,2,1)
                                vox_emb = vox_emb.permute(0,2,1)

                                B,_,_ = vox_emb.shape
                                total_loss = 0.0 
                                if emb_moc.normal_queue:
                                    emb_q = vox_emb.reshape(-1,hyp.feat_dim)
                                    emb_k = vox_emb_key.reshape(-1,hyp.feat_dim)

                                    N = emb_q.shape[0]

                                    emb_k = F.normalize(emb_k,dim=1)
                                    emb_q = F.normalize(emb_q,dim=1)

                                    l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))
                                    if emb_moc.normal_queue:
                                        queue_neg = torch.stack(self.poolvox_moc.fetch())
                                    else:
                                        queue_neg = utils_eval.get_negative_samples(self.poolvox_moc.fetch(),class_val)

                                    K = queue_neg.shape[0]

                                    queue_neg = F.normalize(queue_neg,dim=1)

                                    l_neg = torch.mm(emb_q, queue_neg.T)
                                    l_pos = l_pos.view(N, 1)
                                    logits = torch.cat([l_pos, l_neg], dim=1)
                                    
                                    labels = torch.zeros(N, dtype=torch.long)
                                    labels = labels.to(self.device)

                                    cross_entropy_loss = torch.nn.CrossEntropyLoss()
                                    temp = 0.07
                                    loss = cross_entropy_loss(logits/temp, labels)
                                    total_loss += loss                                
                                else:
                                    for index_batch,emb_q in enumerate(vox_emb):
                                        class_val = classes[index_batch]
                                        N = emb_q.shape[0]
                                        emb_k = vox_emb_key[index_batch]

                                        emb_k = F.normalize(emb_k,dim=1)
                                        emb_q = F.normalize(emb_q,dim=1)

                                        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))
                                        if emb_moc.normal_queue:
                                            queue_neg = torch.stack(self.poolvox_moc.fetch())
                                        else:
                                            queue_neg = utils_eval.get_negative_samples(self.poolvox_moc.fetch(),class_val)
                                        # st()
                                        K = queue_neg.shape[0]

                                        queue_neg = F.normalize(queue_neg,dim=1)

                                        l_neg = torch.mm(emb_q, queue_neg.T)
                                        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)
                                        labels = torch.zeros(N, dtype=torch.long)
                                        labels = labels.to(self.device)

                                        cross_entropy_loss = torch.nn.CrossEntropyLoss()
                                        temp = 0.07
                                        loss = cross_entropy_loss(logits/temp, labels)
                                        total_loss += loss
                                if not emb_moc.normal_queue:
                                    total_loss = total_loss/B
                                backprop_start_time = time.time()
                                self.optimizer.zero_grad()
                                total_loss.backward()
                                self.optimizer.step()
                                self.momentum_update(self.model,self.model_key)
                                self.poolvox_moc.update(vox_emb_key_og,classes_key)
                                summ_writer.summ_scalar('loss', total_loss.cpu().item())
                            else:
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
                                                                                                maxm.max_iters,
                                                                                                total_time,
                                                                                                iter_time,
                                                                                                backprop_time,
                                                                                                total_loss,self.max_steps))
                        for _ in range(maxm.g_max_iters):
                            st()
                            self.total_max_iters += 1
                            self.total_max_iters_g += 1

                            maxm.predicted_matching = False
                            iter_start_time = time.time()
                            try:
                                feed = next(ex_loader)
                            except StopIteration:
                                ex_loader = iter(inputs)
                                feed = next(ex_loader)
                            feed_cuda = {}
                            filename_e = feed.pop('filename_e')
                            filename_g = feed.pop('filename_g')
                            tree_seq_filename = feed.pop('tree_seq_filename')
                            
                            for k in feed:
                                feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                            feed_cuda['filename_e'] = filename_e
                            feed_cuda['filename_g'] = filename_g

                            feed_cuda['tree_seq_filename'] = tree_seq_filename
                            feed_cuda['writer'] = max_writer_g
                            feed_cuda['global_step'] = self.total_max_iters_g
                            feed_cuda['log_freq'] = max_log_freq
                            feed_cuda['set_name'] = max_set_name_g

                            iter_start_time = time.time()
                            loss, results = self.model(feed_cuda)
                            backprop_start_time = time.time()

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            backprop_time = time.time()- backprop_start_time
                            iter_time = time.time()- iter_start_time
                            total_time = time.time()-start_time
                            print("GT Maximization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Global_Steps: %4d"% (hyp.name,
                                                                                                step,
                                                                                                maxm.max_iters,
                                                                                                total_time,
                                                                                                iter_time,
                                                                                                backprop_time,
                                                                                                loss,self.max_steps))



                print("SAVING WEIGHTS MAXIMIZATION:")
                saverloader.save(self.model, self.checkpoint_dir, self.total_max_iters, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
