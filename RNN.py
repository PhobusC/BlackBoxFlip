# This file is for a basic recursive neural network
import torch.nn as nn
import numpy as np

class RNN(nn.module):
    def __init__(self, d_in, d_out, n_layers=1, d_model=64):
        """
        d_model supports list for layers of different dimension or int for constant dim
        """
        self.input_layer = nn.Linear(d_in, d_model)
        if type(d_model) == list or type(d_model) == np.array:
            last_dim = d_in
            layers = []
            for dim in d_model:
                layers.append(nn.Linear(last_dim, dim))
                last_dim = dim
            
            self.layers = nn.ModuleList(layers)
            self.output_layer = nn.Linear(last_dim, d_out)
        else:
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.output_layer = nn.Linear(d_model, d_out)


    def forward(self, data):
        pred = []
        


        return np.array(pred)
