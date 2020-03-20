import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
        self.relu = nn.ReLU()
        
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
        for i in range(out.shape[0]): #NOTE not efficient
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

class EmbeddingsCNN(nn.Module):
    """
    Still making adjustments
    Use a CNN as the embeddings for the images
    """
    def __init__(self, input_dims):
        super(EmbeddingsCNN, self).__init__

        self.cnn_layers = nn.Sequential(
            #NOTE the UMontreal paper uses 14 x 14 x 512
            #using vgg11 (configuration A) as reference
            nn.Conv2d(3, 32, 3, stride=1, padding=1), #416x416x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), #208x208x32

            nn.Conv2d(32, 64, 3, stride=1, padding=1), #208x208x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), #104x104x64
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1), #104x104x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #104x104x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), #52x52x128

            nn.Conv2d(128, 256, 3, stride=1, padding=1), #52x52x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), #52x52x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), #26x26x256

            nn.Conv2d(256, 256, 3, stride=1, padding=1), #26x26x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), #26x26x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1), #13x13x256
 
            nn.Linear(13*13*256, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            # nn.Linear(100,10)
            
            # NOTE might need to change this to a smaller number, the output
            # will end up being the size of the input/context vector
            # may cause over fitting, see https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-an-lstm
            
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x

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
        self.fc1 = nn.Linear(resnet.fc.in_features, fc1_size)
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
                 fc_size=None, attention_dims=None):
        super(Seq2SeqNet, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.fc0 = nn.Linear(embed_in, embed_out)
        self.fc1 = nn.Linear(embed_in, fc_size)
        self.fc2 = nn.Linear(fc_size, embed_out)
        self.relu = nn.ReLU()
        self.fc_size = fc_size
        self.attention_dims = attention_dims
        if attention_dims is not None:
            self.attention = attention.Attention(attention_dims)
            self.attention = self.attention.to(device)

    def forward(self, input, object_shape=None):
        embedded =  self.encoder(input)

        if object_shape is not None:
            #NOTE might want to make another network to embed the object size and incldue that
            pass
        # reshape for rnn input to (seq_length, batch, hidden_size)
        embedded = embedded.permute(1,0,2)
        embedded = embedded.to(self.device) #this doesn't seem necessary?

        if self.attention_dims is not None:
            #NOTE might want to implement attention here
            embedded = self.attention(embedded)

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
