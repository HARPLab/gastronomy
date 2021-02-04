import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

import rnns.attention as attention

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, h_n):
        output, hidden = self.gru(input, h_n)
        output = self.fc(self.relu(output[:, -1]))
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h_0 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return h_0

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        """

        """
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True, 
            dropout = drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU() #TODO might want to try tanh activation instead of relu:https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044
        
    def forward(self, input, h_0=None):
        """

        NOTES:
        - output of LSTM is: output, (h_n, c_n)
            - output is the hidden states of each time step, h_n is hidden of last and c_n is cell state of last
        - h_n.shape (n_layers, batch, hidden_size), c_n.shape (n_layers, batch, hidden_size) 
        - RNN_out has shape=(batch, time_step, output_size)
        - can set h_0 to None for now, if needed see: https://pytorch.org/docs/stable/nn.html
        """
        out, (h_n, c_n) = self.lstm(input, h_0)
        hidden = self.fc(self.relu(h_n)) #NOTE this seems unnecessary
        
        output = []
        for i in range(out.shape[0]): #NOTE not efficient #TODO the batch dim is 0 since batch_first is True
            temp_output = []
            for j in range(out.shape[1]):
                temp_out = self.fc(self.relu(out[i,j,:]))
                temp_output.append(temp_out)
            output.append(torch.stack(temp_output))
        output = torch.stack(output)
        return torch.squeeze(output), (hidden, c_n)
    
    def init_hidden(self, batch_size):
        """
        use to initialize the first hidden state, not mandatory
        """
        weight = next(self.parameters()).data
        h_0 = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return h_0

class ResNetEmbeddings(nn.Module):
    def __init__(self, embed_size, fc1_size=512, fc2_size=512, drop_prob=0.2, momentum=0.1):
        """
        Still making adjustments
        """
        super(ResNetEmbeddings, self).__init__()

        self.drop_prob = drop_prob

        resnet = models.resnet152(pretrained=True)
        # deleting the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc1_size) # 2048 to fc1_size
        # TODO remove batch norm if you aren't using anymore
        self.bn1 = nn.BatchNorm1d(fc1_size, momentum=momentum)
        # https://discuss.pytorch.org/t/i-get-a-much-better-result-with-batch-size-1-than-when-i-use-a-higher-batch-size/20477/8
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(fc2_size, momentum=momentum)
        self.fc3 = nn.Linear(fc2_size, embed_size)

    def forward(self, x):
        """
        Inputs:
            x - (batch_size, sequence_length, D, H, W): Input sequence of images
        """
        embed_seq = []
        # cycle through time steps
        for t in range(x.size(1)):
            with torch.no_grad():
                temp_x = self.resnet(x[:, t, :, :, :])
                temp_x = temp_x.view(temp_x.size(0), -1)

            temp_x = self.fc1(temp_x) # self.bn1(self.fc1(temp_x))
            temp_x = F.relu(temp_x)
            temp_x = self.fc2(temp_x)
            temp_x = F.relu(temp_x)
            temp_x = F.dropout(temp_x, p=self.drop_prob)
            temp_x = self.fc3(temp_x)

            embed_seq.append(temp_x)
        
        # del temp_x
        embed_seq = torch.stack(embed_seq).permute(1,0,2)

        return embed_seq

class Seq2SeqNet(nn.Module):
    """
    Define the model architecture here
    """
    def __init__(self, encoder, decoder, device, embed_in=1, embed_out=1, 
                 fc_size=None, attention_dims=None): #adding dev0 and dev1 might be a good idea and send each model to a device
        super(Seq2SeqNet, self).__init__()
        self.device = device
        self.encoder = encoder.to(device) # I think you need to explicitly send each to GPU since they're different?
        self.decoder = decoder.to(device)
        self.fc0 = nn.Linear(embed_in, embed_out)
        self.fc1 = nn.Linear(embed_in, fc_size)
        self.fc2 = nn.Linear(fc_size, embed_out)
        self.relu = nn.ReLU()
        self.fc_size = fc_size
        self.attention_dims = attention_dims
        if attention_dims is not None:
            raise NotImplemented
            self.attention = attention.Attention(attention_dims)
            self.attention = self.attention.to(device)

    def forward(self, input, object_shape=None):
        embedded =  self.encoder(input)

        if object_shape is not None:
            #NOTE might want to make another network to embed the object size and incldue that
            pass
        # reshape for rnn input to (seq_length, batch, hidden_size)
        embedded = embedded.permute(1,0,2) #TODO you already do this in ResNet forward?

        if self.attention_dims is not None:
            context = None
            #NOTE might want to implement attention here
            embedded = self.attention.forward(embedded, context)

        output, (h_n, c_n) = self.decoder(embedded)

        # output of the encoder for loss calculation
        embed_out = []
        for i in range(embedded.shape[0]): #NOTE not efficient
            temp_output = []
            for j in range(embedded.shape[1]):
                if self.fc_size is None:
                    temp_out = self.fc0(self.relu(embedded[i,j,:].view(1, -1)))
                else:
                    temp_out = self.fc1(self.relu(embedded[i,j,:]))
                    temp_out = self.fc2(temp_out)

                temp_output.append(temp_out)
            embed_out.append(torch.stack(temp_output))
        embed_out = torch.stack(embed_out)
        # del temp_out
        return output, (h_n, c_n), embed_out

    def set_mode(self, mode='evaluate'):
        # Don't think this is necessary or that the eval works properly, should remove
        if mode == 'evaluate':
            self.encoder.eval()
            self.decoder.eval()
        elif mode == 'train':
            self.encoder.train()
            self.decoder.train()

class PlacementShiftDistNet(nn.Module):
    """
    Model for predicting the shift of the object that occurs after it is released from an end effector
    This is based on Alan's paper: 'Towards Robotic Assembly by Predicting Robust, Precise and Task-oriented Grasps'
    (its on his FSR paper now). Outputs a distribution
    """
    def __init__(self, in_channels=2, num_out_dims=2, fc1_size=1024, fc1_input2_size=64,
                 fc2_size=512, input2_size=1, drop_prob=0.5):
        super(PlacementShiftDistNet, self).__init__() #TODO remember to allow for multiple GPUs while your doing this
        # Input size is 64 x 64 x 2
        self.num_out_dims = num_out_dims
        self.drop_prob = drop_prob

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=1), # 62x62x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 60x60x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), # 30x30x32
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 28x28x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 26x26x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=True), # 13x13x32
            nn.ReLU()
        )
   
        self.fc1 = nn.Linear(13*13*32, fc1_size)
        # concatenate ee pose here
        self.fc1_input2 = nn.Linear(input2_size, fc1_input2_size)

        self.fc2_mean = nn.Linear((fc1_size+fc1_input2_size), fc2_size)
        self.fc2_std = nn.Linear((fc1_size+fc1_input2_size), fc2_size)

        self.fc3_mean = nn.Linear(fc2_size, num_out_dims)
        self.fc3_std = nn.Linear(fc2_size, num_out_dims) # TODO: need to change this to covar if you do dependent multivariate dist

        self.relu = nn.ReLU()

    def forward(self, input1, input2, epsilon=1e-8, mean_offset=0.0):
        """
        input1 is the depth images, input 2 is the ee pose
        """
        x = self.layer1(input1 + epsilon)
        x = self.layer2(x + epsilon)

        x = x.view(x.size(0), -1) # flatten each sample
        x = self.relu(self.fc1(x + epsilon))

        x2 = self.relu(self.fc1_input2(input2 + epsilon))
        x = torch.cat((x, x2), dim=-1) # concatenate each sample with second input
        x = F.dropout(x, p=self.drop_prob)

        x_mean = self.relu(self.fc2_mean(x + epsilon))
        x_mean = F.dropout(x_mean, p=self.drop_prob)
        x_mean = self.fc3_mean(x_mean + epsilon) - mean_offset

        x_std = self.relu(self.fc2_std(x + epsilon))
        x_std = F.dropout(x_std, p=self.drop_prob)
        x_std = self.fc3_std(x_std + epsilon)
        x_std = torch.exp(x_std)

        #TODO: predict a distribution for the output
        # can have 2 or 3 independent distributions for each dim prediction or have a multivariate distribution
        output1 = torch.distributions.normal.Normal(x_mean[:,0], x_std[:,0])
        output2 = torch.distributions.normal.Normal(x_mean[:,1], x_std[:,1])
        # torch.distributions.multivariate_normal.MultivariateNormal https://discuss.pytorch.org/t/backward-for-negative-log-likelihood-loss-of-multivariatenormal-in-distributions/15339

        return output1, output2

class PlacementShiftNet(nn.Module):
    def __init__(self, in_channels=2, num_out_dims=2, fc1_size=1024, fc1_input2_size=64,
                 fc2_size=512, input2_size=1, drop_prob=0.5):
        """
        Same as PlacementShiftDistNet, but the vanilla version (i.e. outputs scalars).
        Acitvation of final layer is linear. So other activations can be applied outside
        of this class's forward
        """
        super(PlacementShiftNet, self).__init__() #TODO remember to allow for multiple GPUs while your doing this
        # Input size is 64 x 64 x 2
        self.num_out_dims = num_out_dims
        self.drop_prob = drop_prob

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=1), # 62x62x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 60x60x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), # 30x30x32
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 28x28x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1), # 26x26x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=True), # 13x13x32
            nn.ReLU()
        )
   
        self.fc1 = nn.Linear(13*13*32, fc1_size)
        # concatenate ee pose here
        self.fc1_input2 = nn.Linear(input2_size, fc1_input2_size)
        self.fc2 = nn.Linear((fc1_size+fc1_input2_size), fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_out_dims)

        self.relu = nn.ReLU()

    def forward(self, input1, input2, epsilon=1e-8, mean_offset=0.0):
        """
        input1 is the depth images, input 2 is the ee pose
        """
        x = self.layer1(input1 + epsilon)
        x = self.layer2(x + epsilon)

        x = x.view(x.size(0), -1) # flatten each sample
        x = self.relu(self.fc1(x + epsilon))

        x2 = self.relu(self.fc1_input2(input2 + epsilon))
        x = torch.cat((x, x2), dim=-1) # concatenate each sample with second input
        x = F.dropout(x, p=self.drop_prob)

        x = self.relu(self.fc2(x + epsilon))
        x = F.dropout(x, p=self.drop_prob)
        x = self.fc3(x + epsilon) - mean_offset

        return x


#TODO might want to try making a binary classifier network that predicts whether the 
#object will land in certain pixels. use softmax if you'll only pick one pixel as the
#center, or use sigmoid to predict all pixels that contain the object. Get object size
#from obj image
