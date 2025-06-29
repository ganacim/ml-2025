from torch import nn
from torch.optim import SGD, Adam, AdamW
import torch
from math import ceil
from copy import deepcopy
import numpy as np
import time
from collections import deque

class CNN(nn.Module):
    def __init__(self, num_actions, frame_count = 1, layer_channels = 16, input_shape=(96, 96, 3), lr = 1e-4, softmax = False, use_state_value = False):
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.layer_channels = layer_channels
        self.frame_count = frame_count

        conv_layers = [
            nn.Conv2d(self.input_shape[-1] * frame_count, layer_channels, bias = False, kernel_size = 9, stride=4, padding = 9//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels, layer_channels*2, bias = False, kernel_size = 5, padding = 5//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*2, layer_channels*2, bias = False, kernel_size = 5, stride = 2, padding = 5//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*2, layer_channels*4, bias = False, kernel_size = 3, padding = 3//2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(layer_channels*4, layer_channels*4, bias = False, kernel_size = 3, stride = 2, padding = 3//2),
            nn.LeakyReLU(0.01)
            ]
        
        mlp_layers = [
            nn.Flatten(),
            nn.Linear(layer_channels * 4 * ceil(self.input_shape[0] / 2**4) * ceil(self.input_shape[1] / 2**4), layer_channels),
            nn.LeakyReLU(0.01),
            nn.Linear(layer_channels, num_actions)
        ]

        self.softmax = softmax
        if self.softmax:
            self.softmax = (nn.Softmax(dim = -1))
        else:
            self.softmax = (nn.Identity())
        self.q1 = nn.Sequential(*conv_layers)
        self.q2 = nn.Sequential(*mlp_layers)

        self.use_state_value = use_state_value
        if use_state_value:
            mlp_layers = [
                nn.Flatten(),
                nn.Linear(layer_channels * 4 * ceil(self.input_shape[0] / 2**4) * ceil(self.input_shape[1] / 2**4), layer_channels),
                nn.LeakyReLU(0.01),
                nn.Linear(layer_channels, 1)
            ]
            self.v = nn.Sequential(*mlp_layers)

        #self.optim = Adam(self.parameters(), lr=lr)
        #self.optim = AdamW(self.parameters(), lr=lr, amsgrad=True)
        self.optim = SGD(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()
        x = self.q1(x)
        q = self.q2(x)
        if self.use_state_value:
            v = self.v(x)
            q = v + q - q.mean(dim=1, keepdim=True)
        q = self.softmax(q)
        return q

class TD3_critic(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dims = [256, 128], lr = 1e-4, softmax = True, tanh = False, optim = "AdamW"):
        super(TD3_critic, self).__init__()

        self.c1 = MLP(input_shape, output_shape, hidden_dims, lr, softmax, tanh, optim=0)
        self.c2 = MLP(input_shape, output_shape, hidden_dims, lr, softmax, tanh, optim=0)

        if optim == "Adam":
            self.optim = Adam(self.parameters(), lr=lr)
        elif optim == "AdamW":
            self.optim = AdamW(self.parameters(), lr=lr, amsgrad=True)
        elif optim == "SGD":
            self.optim = SGD(self.parameters(), lr=lr)

    def forward(self, x):
        return self.c1(x), self.c2(x) 

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dims = [256, 128], lr = 1e-4, softmax = True, tanh = False, optim = "AdamW"):
        super(MLP, self).__init__()

        self.hidden_dims = hidden_dims
        self.input_size = np.prod(input_shape)
        self.output_size = np.prod(output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape

        mlp_layers = []

        mlp_layers += [
            nn.Flatten(),
            nn.Linear(self.input_size, hidden_dims[0]),
            #nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0),
        ]
        for i in range(len(hidden_dims) - 1):
            mlp_layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                #nn.LayerNorm(hidden_dims[i+1]),
                nn.LeakyReLU(0),
            ]
        mlp_layers += [
            nn.Linear(hidden_dims[-1], self.output_size),
        ]

        self.softmax = softmax
        if softmax:
            mlp_layers.append(nn.Softmax(dim = -1))
        self.tanh = tanh
        if self.tanh:
            mlp_layers.append(nn.Tanh())
        self.q1 = nn.Sequential(*mlp_layers)

        if optim == "Adam":
            self.optim = Adam(self.parameters(), lr=lr)
        elif optim == "AdamW":
            self.optim = AdamW(self.parameters(), lr=lr, amsgrad=True)
        elif optim == "SGD":
            self.optim = SGD(self.parameters(), lr=lr)

    def forward(self, x):   
        return self.q1(x.float())  
        