# This file is for a basic recursive neural network
import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, d_model=64):
        """
        d_model supports list for layers of different dimension or int for constant dim
        """
        super().__init__()
        
        if type(d_model) == list or type(d_model) == np.ndarray:
            if n_layers != len(d_model):
                raise ValueError("The number of layers should be equal to the size of d_model")
            
            last_dim = d_in
            W_ih = []
            W_hh = []
            self.hidden_size = []
            for dim in d_model:
                W_ih.append(nn.Linear(last_dim, dim))
                W_hh.append(nn.Linear(dim, dim))
                self.hidden_size.append(dim)
                last_dim = dim
            
            self.W_ih = nn.ModuleList(W_ih)
            self.W_hh = nn.ModuleList(W_hh)
            

            self.output_layer = nn.Linear(last_dim, d_out)

            
        else:
            # W_ih: list of connections from previous layers' hidden states
            # W_hh: list of connections from this layer's hidden state from past timestep

            self.W_ih = nn.ModuleList([nn.linear(d_in, d_model)]
                                        + [nn.Linear(d_model, d_model) for _ in range(1, n_layers)])

            self.W_hh = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.hidden_size = d_model
            self.output_layer = nn.Linear(d_model, d_out)
            

        self.n_layers = n_layers

        # self.activation = nn.ReLU()
        self.activation = nn.Tanh() # apparently better for RNN, prob bc vanishing gradient



    def forward(self, data, h_0=None):
        batch_size, n_obs, features = data.shape


        if h_0 is None:
            if type(self.hidden_size) == int:
                h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
            else:
                h_0 = []
                for size in self.hidden_size:
                    h_0.append(torch.zeros(batch_size, size))


        pred = []



        return np.array(pred) # Change to torch tensor
