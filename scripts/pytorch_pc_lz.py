# %%

"""
pytorch implementation of deep hebbian predictive coding(DHPC) net that enables relatively flexible maniputation of network architecture
code inspired by Matthias's code of PCtorch
"""
import torch
import torch.nn as nn
import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import scipy
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
else:
    dev = "cpu"
    print("Cuda not available")
device = torch.device(dev)

dtype = torch.float  # Set standard datatype


# %%
# helper functions

def sigmoid(x):
    return nn.Sigmoid(x + 3)


# %%
#  layer class
class PredLayer(nn.Module):
    #  object class for standard layer in DHPC with error and representational units
    def __init__(self, layer_size: int, out_size: int, inf_rate: float, act_func=sigmoid, device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PredLayer, self).__init__()  # super().__init__()
        self.layer_size = layer_size  # num of units in this layer
        self.out_size = out_size  # num of units in next layer for constructution of weight matrix

        # self.e_activation = torch.zeros((layer_size))  # activation level of e units
        # self.r_activation = torch.zeros((layer_size))  # activation level of r units
        # self.r_output = torch.zeros((layer_size))  # output firing rates of r units
        # self.r_prediction = torch.zeros((layer_size))  # prediction imposed by higher layer

        self.infRate = inf_rate  # inference rate governing how fast r adjust to errors
        self.actFunc = act_func

        self.weight = Parameter(torch.empty((layer_size, out_size), **factory_kwargs), requires_grad=False)
        self.reset_parameters()
        # self.reset_state()

    def reset_parameters(self) -> None:  # initialise or reset layer weight
        nn.init.normal_(self.weight, 0, 0.5)  # normal distribution
        self.weight = torch.clamp(self.weight, min=0)  # weights clamped to above 0
        self.weight = self.weights / self.out_size  # normalise weights given next layer size

    # def reset_state(self):
    # reinitialise activation and output values

    def forward(self, inputs, bu_errors, r_act, r_out, nextlayer_r_out):
        # values that is needed per layer: e, r_act, r_out
        # prediction: w_l, r_out_l+1
        # inference: e_l (y_l-pred), w_l-1, e_l-1, return updated e, r_act, r_out

        # e, r_act, r_out each an array, each index correspond to layer

        e_act = r_out - torch.matmul(self.weights,
                                     nextlayer_r_out)  # The activity of error neurons is representation - prediction.
        r_act = r_act + self.inference_rate * (
                bu_errors - e_act)  # Inference step: Modify activity depending on error
        r_out = self.actFunc(r_act)  # Apply the activation function to get neuronal output
        return e_act, r_act, r_out


# %%

#  network class
# state dict that registers all the internal activations
# whenever forward pass is called, reads state dict first, then do computation
class DHPC(nn.Module):
    def __init__(self, network_architecture):
        state_dict = []  # a list that always keep tracks of internal state values
        for layer in range(len(network_architecture)):
            r_act = torch.zeros(network_architecture[layer])  # tensor containing activation of representational units
            r_out = torch.zeros(network_architecture[layer])  # tensor containing output of representational units
            r_pred = torch.zeros(network_architecture[layer])  # tensor containing value of prediction
            if layer != len(network_architecture) - 1:
                e_act = torch.zeros(network_architecture[layer])  # tensor containing activation of error units
            layer_dict = [r_pred, e_act, r_act, r_out]  # list logging state values of that layer
            state_dict.append(layer_dict)


#  data preprocessing


#  network instantiation
network_architecture = [100, 50, 10]

#  training loop


#  evaluation
#  register_forward_hook can be used to inspect internal activation
