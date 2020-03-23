import numpy as np
import torch.nn.functional as F
import hyperparams as hyp
import cross_corr
import torch
import os
import ipdb
import hyperparams as hyp
import sys
st = ipdb.set_trace
# import archs.encoder3D3D as abc
from archs.neural_modules import ResNet3D_NOBN

# st()



class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_dim)
        self.embeddings.cuda()

        if init_embeddings is not None:
            print("LOADING embeddings FROM NUMPY: ",init_embeddings)
            cluster_centers = np.load(init_embeddings)
            self.embeddings.weight.data = torch.from_numpy(cluster_centers).to(torch.float32).cuda()
        else:
            limit = 1/self.num_embeddings
            self.embeddings.weight.data.uniform_(-limit,+limit)
        self.commitment_cost = commitment_cost
        if hyp.vq_rotate:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)

    def forward(self, inputs):
        if hyp.vq_rotate:
            input_shape = inputs.shape # torch.Size([2, 32, 16, 16, 16])
            inputs = self.mbr.rotateTensor(inputs)
            B,angles,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            flat_input = inputs.reshape(B,angles,-1)
            assert(C==self.embedding_dim)
            E = self.num_embeddings

            distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
            dB, dA, dF = distances.shape
            distances = distances.view(B, -1)
            if hyp.filter_boxes or hyp.self_improve_iterate:
                return distances
            rotIdxMin = torch.argmin(distances, dim=1).unsqueeze(1)
            best_rotations = rotIdxMin//dF # Find the rotation for min distance
            best_rotations = best_rotations.squeeze(1)
            encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)

            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

            best_rotated_inputs = inputs[torch.arange(B), best_rotations.long()]
            e_latent_loss = F.mse_loss(quantized.detach(), best_rotated_inputs)
            q_latent_loss = F.mse_loss(quantized, best_rotated_inputs.detach())

            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            quantized = best_rotated_inputs + (quantized - best_rotated_inputs).detach()
            quantized_c = quantized.clone()
            if hyp.gt_rotate_combinations:
                quantized_unrotated = quantized
            quantized = self.mbr.rotateTensorToPose(quantized,best_rotations)
            # st()
        else:
            input_shape = inputs.shape
            B,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            assert(C==self.embedding_dim)
            E = self.num_embeddings
            flat_input = inputs.view(B,-1)

            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
                
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
            
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))        
        # convert quantized from BHWC -> BCHW
        if hyp.gt_rotate_combinations:
            return loss, [quantized,best_rotated_inputs,quantized_unrotated,best_rotations], perplexity, encodings,
        else:
            return loss, quantized, perplexity, encodings


class VectorQuantizer_Instance_Vr(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_embeddings, commitment_cost):
        super(VectorQuantizer_Instance_Vr, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_dim)
        self.embeddings.cuda()

        self.genVar = ResNet3D_NOBN(hyp.feat_dim,hyp.feat_dim)
        self.genVar.cuda()

        if init_embeddings is not None:
            cluster_centers = np.load(init_embeddings)
            self.embeddings.weight.data = torch.from_numpy(cluster_centers).to(torch.float32).cuda()
        else:
            limit = 1/self.num_embeddings
            self.embeddings.weight.data.uniform_(-limit,+limit)
        self.commitment_cost = commitment_cost
        if hyp.vq_rotate:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)

    def sample_z(self,args):
        mu, log_sigma = args
        eps = torch.normal(0,hyp.var_coeff,log_sigma.shape).cuda()
        return mu + torch.exp(log_sigma / 2) * eps

    def forward(self, inputs):
        if hyp.vq_rotate:
            input_shape = inputs.shape # torch.Size([2, 32, 16, 16, 16])
            inputs = self.mbr.rotateTensor(inputs)
            B,angles,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            flat_input = inputs.reshape(B,angles,-1)
            assert(C==self.embedding_dim)
            E = self.num_embeddings

            distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
            dB, dA, dF = distances.shape
            distances = distances.view(B, -1)
            rotIdxMin = torch.argmin(distances, dim=1).unsqueeze(1)
            best_rotations = rotIdxMin//dF # Find the rotation for min distance
            best_rotations = best_rotations.squeeze(1)
            encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)

            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
            best_rotated_inputs = inputs[torch.arange(B), best_rotations.long()]

            # st()
            quantized_mean  = quantized
            quantized_log_sigma = self.genVar(quantized)
            quantized_samples = torch.stack([self.sample_z([quantized_mean,quantized_log_sigma]) for i in range(10)],dim=1).reshape([hyp.B,10,-1])
            # st()

            best_rotated_inputs_flat = best_rotated_inputs.reshape([hyp.B,-1])
            all_sample_indices = []
            for index in range(hyp.B):
                best_rotated_inputs_flat_index = best_rotated_inputs_flat[index:index+1]
                quantized_samples_flat_index = quantized_samples[index:index+1].squeeze()
                distances_samples_index = (torch.sum(best_rotated_inputs_flat_index**2, dim=1, keepdim=True) 
                            + torch.sum(quantized_samples_flat_index**2, dim=1)
                            - 2 * torch.matmul(best_rotated_inputs_flat_index,quantized_samples_flat_index.t()))
                sampleIdxMin = torch.argmin(distances_samples_index, dim=1).unsqueeze(1)            
                sample_indices = sampleIdxMin.squeeze()
                all_sample_indices.append(sample_indices)
            # st()l
            all_sample_indices_stacked = torch.stack(all_sample_indices)

            quantized = quantized_samples[torch.arange(B),all_sample_indices_stacked]
            quantized = quantized.reshape(best_rotated_inputs.shape)

            e_latent_loss = F.mse_loss(quantized.detach(), best_rotated_inputs)
            q_latent_loss = F.mse_loss(quantized, best_rotated_inputs.detach())

            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            quantized = self.mbr.rotateTensorToPose(quantized,best_rotations)

            quantized = best_rotated_inputs + (quantized - best_rotated_inputs).detach()
        else:
            input_shape = inputs.shape
            B,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            assert(C==self.embedding_dim)
            E = self.num_embeddings
            flat_input = inputs.view(B,-1)

            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
                
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
            
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized, perplexity, encodings

class VectorQuantizer_Instance_Vr_All(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_embeddings, commitment_cost):
        super(VectorQuantizer_Instance_Vr_All, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_dim)
        self.embeddings.cuda()


        self.genVar = ResNet3D_NOBN(hyp.feat_dim,hyp.feat_dim)
        self.genVar.cuda()        
        
        if init_embeddings is not None:
            cluster_centers = np.load(init_embeddings)
            self.embeddings.weight.data = torch.from_numpy(cluster_centers).to(torch.float32).cuda()
        else:
            limit = 1/self.num_embeddings
            self.embeddings.weight.data.uniform_(-limit,+limit)
        self.commitment_cost = commitment_cost
        if hyp.vq_rotate:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)


    def sample_z(self,args):
        mu, log_sigma = args
        eps = torch.normal(0,hyp.var_coeff,log_sigma.shape).cuda()
        return mu + torch.exp(log_sigma / 2) * eps

    def forward(self, inputs):
        if hyp.vq_rotate:
            input_shape = inputs.shape # torch.Size([2, 32, 16, 16, 16])
            inputs = self.mbr.rotate2D(inputs)
            B,angles,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            flat_input = inputs.reshape(B,angles,-1)
            assert(C==self.embedding_dim)
            # E = self.num_embeddings
            dictionary = self.embeddings.weight
            dictionary = dictionary.reshape([dictionary.shape[0],inputs.shape[2],inputs.shape[3],inputs.shape[4],inputs.shape[5]])

            dictionary_mean  = dictionary
            dictionary_log_sigma = self.genVar(dictionary)
            dictionary_samples = torch.stack([self.sample_z([dictionary_mean,dictionary_log_sigma]) for i in range(hyp.num_rand_samps)],dim=1)
            dictionary_samples = dictionary_samples.reshape([dictionary_samples.shape[0]*dictionary_samples.shape[1],-1])

            # st()
            distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                        + torch.sum(dictionary_samples**2, dim=1)
                        - 2 * torch.matmul(flat_input, dictionary_samples.t()))
            
            dB, dA, dF = distances.shape
            distances = distances.view(B, -1)
            rotIdxMin = torch.argmin(distances, dim=1).unsqueeze(1)
            best_rotations = rotIdxMin//dF # Find the rotation for min distance
            best_rotations = best_rotations.squeeze(1)
            encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)
            # st()
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings*hyp.num_rand_samps, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, dictionary_samples).view(input_shape)
            best_rotated_inputs = inputs[torch.arange(B), best_rotations.long()]
            e_latent_loss = F.mse_loss(quantized.detach(), best_rotated_inputs)
            q_latent_loss = F.mse_loss(quantized, best_rotated_inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            quantized = best_rotated_inputs + (quantized - best_rotated_inputs).detach()
            quantized = self.mbr.rotateTensorToPose(quantized,best_rotations)
            # st()
        else:
            input_shape = inputs.shape
            B,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            assert(C==self.embedding_dim)
            E = self.num_embeddings
            flat_input = inputs.view(B,-1)

            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
                
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
            
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized, perplexity, encodings




class VectorQuantizer_vox(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_embeddings, commitment_cost):
        super(VectorQuantizer_vox, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_dim)
        self.embeddings.cuda()
        if init_embeddings is not None:
            cluster_centers = np.load(init_embeddings)
            self.embeddings.weight.data = torch.from_numpy(cluster_centers).to(torch.float32).cuda()
        else:
            limit = 1/self.num_embeddings
            self.embeddings.weight.data.uniform_(-limit,+limit)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        B,C,D,H,W = list(inputs.shape)
        inputs = inputs.permute(0,2,3,4,1)
        input_shape = inputs.shape
        # C = C*H*W*D
        assert(C==self.embedding_dim)
        E = self.num_embeddings
        flat_input = inputs.reshape([-1,C])
        ## Work on chunks and not the whole set to avoid OOM.
        # Flatten input
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))        
        quantized = quantized.permute(0,4,1,2,3)
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

class VectorQuantizer_Eval(torch.nn.Module):
    def __init__(self):
        super(VectorQuantizer_Eval, self).__init__()
        # init_dir = os.path.join("checkpoints", feat_init)
        # ckpt_names = os.listdir(init_dir)
        # steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        # step = max(steps)
        # model_name = 'model-%d.pth'%(step)
        # path = os.path.join(init_dir, model_name)
        if hyp.use_instances_variation_all:
            self.genVar = ResNet3D_NOBN(hyp.feat_dim,hyp.feat_dim)
            self.genVar.cuda()
        self.embedding_dim = hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.feat_dim
        self.num_embeddings = hyp.object_quantize_dictsize

        self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                             self.embedding_dim)
        self.embeddings.cuda()
        # st()
        # loaded_embeddings = torch.load(path)['model_state_dict']['quantizer.embeddings.weight']
        # self.embeddings.weight.data = loaded_embeddings
        if hyp.vq_rotate:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)

    def sample_z(self,args):
        mu, log_sigma = args
        eps = torch.normal(0,hyp.var_coeff,log_sigma.shape).cuda()
        return mu + torch.exp(log_sigma / 2) * eps

    def predict(self, inputs):
        if hyp.use_instances_variation_all:
            input_shape = inputs.shape # torch.Size([2, 32, 16, 16, 16])
            inputs = self.mbr.rotateTensor(inputs)
            B,angles,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            flat_input = inputs.reshape(B,angles,-1)
            assert(C==self.embedding_dim)
            E = self.num_embeddings

            dictionary = self.embeddings.weight
            dictionary = dictionary.reshape([dictionary.shape[0],inputs.shape[2],inputs.shape[3],inputs.shape[4],inputs.shape[5]])


            dictionary_mean  = dictionary
            dictionary_log_sigma = self.genVar(dictionary)
            dictionary_samples = torch.stack([self.sample_z([dictionary_mean,dictionary_log_sigma]) for i in range(hyp.num_rand_samps)],dim=1)
            dictionary_samples = dictionary_samples.reshape([dictionary_samples.shape[0]*dictionary_samples.shape[1],-1])
            # st()
            distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                        + torch.sum(dictionary_samples**2, dim=1)
                        - 2 * torch.matmul(flat_input, dictionary_samples.t()))
            dB, dA, dF = distances.shape
            distances = distances.view(B, -1)
            rotIdxMin = torch.argmin(distances, dim=1).unsqueeze(1)
            best_rotations = rotIdxMin//dF # Find the rotation for min distance
            best_rotations = best_rotations.squeeze(1)
            encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)        
            encoding_indices = encoding_indices.squeeze(1).cpu().numpy()
            encoding_indices = encoding_indices//hyp.num_rand_samps
        elif hyp.vq_rotate:
            input_shape = inputs.shape # torch.Size([2, 32, 16, 16, 16])
            inputs = self.mbr.rotateTensor(inputs)
            B,angles,C,D,H,W = list(inputs.shape)
            C = C*H*W*D
            flat_input = inputs.reshape(B,angles,-1)
            assert(C==self.embedding_dim)
            E = self.num_embeddings
            distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                      + torch.sum(self.embeddings.weight**2, dim=1)
                      - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            dB, dA, dF = distances.shape
            distances = distances.view(B, -1)
            rotIdxMin = torch.argmin(distances, dim=1).unsqueeze(1)
            best_rotations = rotIdxMin//dF # Find the rotation for min distance
            best_rotations = best_rotations.squeeze(1)
            encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)        
            encoding_indices = encoding_indices.squeeze(1).cpu().numpy()            
        else:
            inputs = inputs.reshape(hyp.B,-1)
            input_shape = inputs.shape
            B,C = list(inputs.shape)
            # C = C*H*W*D
            assert(C==self.embedding_dim)
            # st()
            E = self.num_embeddings
            flat_input = inputs.view(B,-1)

            ## Work on chunks and not the whole set to avoid OOM.
            # Flatten input
            # Calculate distances
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.embeddings.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
                
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encoding_indices = encoding_indices.squeeze(1).cpu().numpy()
            # convert quantized from BHWC -> BCHW
        return encoding_indices