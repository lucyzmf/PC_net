# %%
###########################
#  layer class
###########################
import os

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn

CONFIG_PATH = "../scripts/"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

reg_strength = config['reg_strength']
reg_type = config['reg_type']
infstep_before_update = config['infstep_before_update']
act_normalise = config['act_norm']
norm_constant = config['norm_constant']
stride = config['stride']


#  sigmoid activation function
def sigmoid(inputs):
    inputs = inputs - 3
    m = nn.Sigmoid()
    return m(inputs)


def relu(inputs):
    m = nn.ReLU()
    return m(inputs)


class RfLayer(nn.Module):
    #  object class for standard layer in DHPC with error and representational units
    def __init__(self, layer_width: int, channel_l: int, channel_next: int, filter_sz: int, inf_rate: float, device,
                 dtype, lr_rate, act_func
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RfLayer, self).__init__()  # super().__init__()
        self.layer_width = layer_width  # size of layer width
        self.filter_sz = filter_sz
        self.infRate = inf_rate  # inference rate governing how fast r adjust to errors
        self.actFunc = globals()[act_func]  # activation function
        self.learn_rate = lr_rate  # learning rate
        self.channel_sz = channel_l

        n_patch = ((layer_width - (
                    filter_sz - stride)) / stride) ** 2  # batch dimension of later computations, num of patches in total
        self.weights = torch.empty((n_patch, channel_l * filter_sz * filter_sz, channel_next), **factory_kwargs)
        self.reset_parameters()
        self.weights = nn.Parameter(self.weights, requires_grad=False)
        # self.reset_state()

        self.device = device

    def reset_parameters(self) -> None:  # initialise or reset layer weight
        nn.init.normal_(self.weights, 0, 1)  # normal distribution
        # nn.init.constant_(self.weights, 0.5)  # constant distribution
        # self.weights = torch.clamp(self.weights, min=0)  # weights clamped to above 0
        self.weights = self.weights / self.filter_sz ** 2  # normalise weights given this layer size

    # def reset_state(self):
    # reinitialise activation and output values

    def forward(self, bu_errors, r_act, r_out, nextlayer_r_out, norm=act_normalise, constant=norm_constant):
        # values that is needed per layer: e, r_act, r_out
        # prediction: w_l, r_out_l+1
        # inference: e_l (y_l-pred), w_l-1, e_l-1, return updated e, r_act, r_out

        # TODO top down need to resolve overlap,
        e_act = r_out - torch.bmm(self.weights,
                                  nextlayer_r_out)  # The activity of error neurons is representation - prediction.
        # pass e_act through activation function
        e_act = torch.tanh(e_act)

        r_act = r_act + self.infRate * (
                bu_errors - e_act)  # Inference step: Modify activity depending on error
        # add activity normalisation of neurons
        if norm:
            r_act = r_act ** 2 / (constant + torch.sum(r_act) ** 2)
        r_out = self.actFunc(r_act)  # Apply the activation function to get neuronal output
        return e_act, r_act, r_out

    def w_update(self, e_act, nextlayer_output, reg_constant=reg_strength, r_type=reg_type):
        if r_type == 'l1':
            reg_matrix = -torch.sign(torch.clone(self.weights))
            reg_matrix = reg_constant * reg_matrix
        elif r_type == 'l2':
            reg_matrix = -torch.sign(torch.clone(self.weights))
            reg_matrix = reg_constant * reg_matrix * self.weights
        else:
            reg_matrix = 0
        # Learning step
        delta = self.learn_rate * (torch.bmm(nextlayer_output, torch.transpose(e_act, 1, 2)) + reg_matrix)
        # self.weights = nn.Parameter(torch.clamp(self.weights + delta, min=0))  # Keep only positive weights
        # self.weights = nn.Parameter(self.weights + delta)  # keep both positive and negative weights


class input_layer(RfLayer):
    # Additional class for the input layer. This layer does not use a full inference step (driven only by input).
    def forward(self, inputs, nextlayer_r_out):
        top_down_projection = torch.bmm(self.weights, nextlayer_r_out)
        # TODO when compute top down projection,
        # compute the actual prediction given overlapping rfs
        inputs = inputs.repeat(self.channel_sz, 1, 1)  # repeat input image(2d) in first dim with channel size
        e_act = inputs.to(self.device) - top_down_projection
        # pass e_act through tanh
        e_act = torch.tanh(e_act)
        return e_act, r_act, r_out


class output_layer(RfLayer):
    # Additional class for last layer. This layer requires a different inference step as no top-down predictions exist.
    def forward(self, bu_errors, r_act):
        r_act = r_act + self.infRate * bu_errors
        r_out = self.actFunc(r_act)
        return r_act, r_out


# %%
###########################
#  network class
###########################
# state dict that registers all the internal activations
# whenever forward pass is called, reads state dict first, then do computation

class RfDHPC(nn.Module):
    def __init__(self, filter_sizes, n_channels, inf_rates, lr, act_func, device, dtype):
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
                    input_layer(network_arch[0], network_arch[1], inf_rates[0], device=device, dtype=dtype, lr_rate=lr,
                                act_func=act_func))  # append input layer to modulelist
            elif layer != len(network_arch) - 1:
                # add middle layer to module list and state list
                e_act.append(torch.zeros(network_arch[layer]).to(device))  # tensor containing activation of error units
                self.layers.append(
                    FCLayer(network_arch[layer], network_arch[layer + 1], inf_rates[layer], device=device,
                            dtype=dtype,
                            lr_rate=lr, act_func=act_func))
            else:
                # add output layer to module list
                e_act.append(None)
                self.layers.append(output_layer(network_arch[layer], network_arch[layer], inf_rates[layer],
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

    def forward(self, frame, inference_steps, istrain=True, inf_before_learn=infstep_before_update):
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

            if istrain:
                if i > 0 and i % inf_before_learn == 0:
                    self.learn()

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
