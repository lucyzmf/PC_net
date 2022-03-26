# %%
###########################
#  layer class
###########################
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


#  sigmoid activation function
def sigmoid(inputs):
    inputs = inputs - 3
    m = nn.Sigmoid()
    return m(inputs)


class Rf_PredLayer(nn.Module):
    #  object class for standard layer in DHPC with error and representational units
    def __init__(self, layer_size: int, out_size: int, filter_size: int, inf_rate: float, device, dtype,
                 lr_rate, act_func
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Rf_PredLayer, self).__init__()  # super().__init__()
        self.layer_size = layer_size  # num of units in this layer
        self.layer_width = math.sqrt(layer_size)
        self.filter_size = filter_size
        self.out_size = out_size  # num of units in next layer for constructution of weight matrix

        self.infRate = inf_rate  # inference rate governing how fast r adjust to errors
        self.actFunc = globals()[act_func]  # activation function
        self.learn_rate = lr_rate  # learning rate

        self.weights = torch.empty((layer_size, out_size), **factory_kwargs)
        self.connectivity_map = torch.empty((layer_size, out_size), **factory_kwargs)
        for m in range(layer_size):  # create connectivity map, first calculate distance, then if > filtersize, 0
            for n in range(out_size):
                self.connectivity_map[m][n] = math.sqrt(
                    (math.floor(m / self.layer_width) - math.floor(n / self.layer_width)) ** 2
                    + (m % self.layer_width - n % self.layer_width) ** 2)
        self.connectivity_map[self.connectivity_map <= filter_size] = 1
        self.connectivity_map[self.connectivity_map > filter_size] = 0  # need to select in this sequence otherwise the matrix gets fked
        self.reset_parameters()
        self.weights = nn.Parameter(self.weights, requires_grad=False)
        # self.reset_state()

        self.device = device

    def reset_parameters(self) -> None:  # initialise or reset layer weight
        nn.init.normal_(self.weights, 0, 0.5)  # normal distribution
        # self.weights = torch.clamp(self.weights, min=0)  # weights clamped to above 0
        self.weights = self.weights / self.filter_size ** 2  # normalise weights given next layer size
        self.weights = self.weights * self.connectivity_map

    # def reset_state(self):
    # reinitialise activation and output values

    def forward(self, bu_errors, r_act, r_out, nextlayer_r_out):
        # values that is needed per layer: e, r_act, r_out
        # prediction: w_l, r_out_l+1
        # inference: e_l (y_l-pred), w_l-1, e_l-1, return updated e, r_act, r_out

        e_act = r_out - torch.matmul(self.weights,
                                     nextlayer_r_out)  # The activity of error neurons is representation - prediction.
        r_act = r_act + self.infRate * (
                bu_errors - e_act)  # Inference step: Modify activity depending on error

        # add competition: calculate mean, if smaller than mean, silence
        mean = torch.mean(r_act)
        std = torch.std(r_act)
        r_act[r_act < (mean + 0.25 * std)] = -1

        r_out = self.actFunc(r_act)  # Apply the activation function to get neuronal output
        return e_act, r_act, r_out

    def w_update(self, e_act, nextlayer_output):
        # Learning step
        l1_reg = torch.clone(self.weights)
        l1_reg[l1_reg > 0] = 1
        l1_reg[l1_reg < 0] = -1
        # Learning step
        delta = self.learn_rate * (torch.matmul(e_act.reshape(-1, 1), nextlayer_output.reshape(1, -1)) + l1_reg)
        # delta = self.learn_rate * torch.matmul(e_act.reshape(-1, 1), nextlayer_output.reshape(1, -1))
        self.weights = nn.Parameter((self.weights + delta) * self.connectivity_map) # get rid of extra rf connections


class input_layer(Rf_PredLayer):
    # Additional class for the input layer. This layer does not use a full inference step (driven only by input).
    def forward(self, inputs, nextlayer_r_out):
        e_act = inputs.to(self.device) - torch.matmul(self.weights, nextlayer_r_out)
        return e_act


class output_layer(Rf_PredLayer):
    # Additional class for last layer. This layer requires a different inference step as no top-down predictions exist.
    def forward(self, bu_errors, r_act):
        r_act = r_act + self.infRate * bu_errors
        r_out = self.actFunc(r_act)

        # add competition: calculate mean, if smaller than mean, silence
        mean = torch.mean(r_act)
        std = torch.std(r_act)
        r_act[r_act < (mean + 0.5 * std)] = -2

        return r_act, r_out


# %%
###########################
#  network class
###########################
# state dict that registers all the internal activations
# whenever forward pass is called, reads state dict first, then do computation

class RfDHPC_cm(nn.Module):
    def __init__(self, network_arch, filter_sizes, inf_rates, lr, act_func, device, dtype):
        super().__init__()
        e_act, r_act, r_out = [], [], []  # a list that always keep tracks of internal state values
        self.layers = nn.ModuleList().to(device)  # create module list containing all layers
        self.architecture = network_arch
        self.inf_rates = inf_rates
        self.device = device
        self.dtype = dtype

        # e, r_act, r_out each an array, each index correspond to layer
        for layer in range(len(network_arch)):
            r_act.append(
                torch.zeros(network_arch[layer]).to(device))  # tensor containing activation of representational units
            r_out.append(
                torch.zeros(network_arch[layer]).to(device))  # tensor containing output of representational units
            if layer == 0:
                # add input layer to module list, add input layer state list
                e_act.append(torch.zeros(network_arch[layer]).to(device))
                self.layers.append(
                    input_layer(network_arch[0], network_arch[1], filter_sizes[0], inf_rates[0], device=device, dtype=dtype, lr_rate=lr,
                                act_func=act_func))  # append input layer to modulelist
            elif layer != len(network_arch) - 1:
                # add middle layer to module list and state list
                e_act.append(torch.zeros(network_arch[layer]).to(device))  # tensor containing activation of error units
                self.layers.append(
                    Rf_PredLayer(network_arch[layer], network_arch[layer + 1], filter_sizes[layer], inf_rates[layer], device=device,
                                 dtype=dtype,
                                 lr_rate=lr, act_func=act_func))
            else:
                # add output layer to module list
                e_act.append(None)
                self.layers.append(output_layer(network_arch[layer], network_arch[layer], filter_sizes[layer], inf_rates[layer],
                                                device=device, dtype=dtype, lr_rate=lr, act_func=act_func))

        self.states = {
            'error': e_act,
            'r_activation': r_act,
            'r_output': r_out,
        }

    def init_states(self):
        # initialise values in state dict
        for i in range(len(self.states['r_activation'])):
            if i != len(self.architecture) - 1:
                self.states['error'][i] = torch.zeros(self.architecture[i]).to(self.device)
            self.states['r_activation'][i] = -2 * torch.ones(self.architecture[i]).to(self.device)
            self.states['r_output'][i] = self.layers[i].actFunc(self.states['r_activation'][i])

    def forward(self, frame, inference_steps):
        # frame is input to the lowest layer, inference steps
        e_act, r_act, r_out = self.states['error'], self.states['r_activation'], self.states['r_output']
        layers = self.layers
        r_act[0] = torch.flatten(frame)  # r units of first layer reflect input
        r_out[0] = layers[0].actFunc(r_act[0])

        # inference process
        for i in range(inference_steps):
            e_act[0] = layers[0](r_act[0], r_out[1])  # update first layer, given inputs, calculate error
            for j in range(1, len(layers) - 1):  # iterate through middle layers, forward inference
                e_act[j], r_act[j], r_out[j] = layers[j](
                    torch.matmul(torch.transpose(layers[j - 1].weights, 0, 1), e_act[j - 1]),
                    r_act[j], r_out[j], r_out[j + 1])
            # update states of last layer
            r_act[-1], r_out[-1] = layers[-1](torch.matmul(torch.transpose(layers[-2].weights, 0, 1), e_act[-2]),
                                              r_act[-1])

    def learn(self):
        # iterate through all non last layers to update weights
        for i in range(len(self.architecture) - 1):
            self.layers[i].w_update(self.states['error'][i], self.states['r_output'][i + 1])

    def total_error(self):
        total = []
        for i in range(len(self.architecture) - 1):
            error = torch.mean(torch.pow(self.states['error'][i], 2))
            total.append(error)
        return torch.mean(torch.tensor(total))

    def reconstruct(self, image, label, infsteps):  # reconstruct from second layer output
        self.init_states()
        self.forward(torch.flatten(image), infsteps)
        label = label.item()
        error = self.total_error()

        reconstructed_frame = self.layers[0].weights @ self.states['r_output'][1]
        reconstructed_frame = reconstructed_frame.detach().cpu().numpy()
        img_width = int(np.sqrt(len(reconstructed_frame)))

        fig, ax = plt.subplots()
        im = ax.imshow(np.reshape(reconstructed_frame, (img_width, img_width)))
        ax.set_title('reconstruction digit %i error %f' % (label, error.item()))
        # plt.show()

        return error, fig
