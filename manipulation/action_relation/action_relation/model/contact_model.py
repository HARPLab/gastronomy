import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae.models.layers import build_mlp


class ContactDistributionPredictionModel(nn.Module):
    def __init__(self, inp_emb_size, output_size, args):
        super(ContactDistributionPredictionModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.output_size = output_size
        self.args = args

        self.model = nn.Sequential(
            nn.Linear(inp_emb_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, inp):
        x = inp
        out = self.model(x)
        return out

class ContactForceTorquePredictionModel(nn.Module):
    def __init__(self, inp_emb_size, output_size, args):
        super(ContactForceTorquePredictionModel, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.output_size = output_size
        self.args = args

        self.model = nn.Sequential(
            nn.Linear(inp_emb_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, inp):
        x = inp
        out = self.model(x)
        return out
