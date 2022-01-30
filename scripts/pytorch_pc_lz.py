# %%

"""
pytorch implementation of deep hebbian predictive coding(DHPC) net that enables relatively flexible maniputation of network architecture
code inspired by Matthias's code of PCtorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances  # computes the pairwise distance between observations
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
###########################
# helper functions
###########################

def sigmoid(inputs):
    inputs = inputs - 3
    m = nn.Sigmoid()
    return m(inputs)


# %%
###########################
#  layer class
###########################

class PredLayer(nn.Module):
    #  object class for standard layer in DHPC with error and representational units
    def __init__(self, layer_size: int, out_size: int, inf_rate: float, lr_rate=.01, act_func=sigmoid, device=device,
                 dtype=dtype) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PredLayer, self).__init__()  # super().__init__()
        self.layer_size = layer_size  # num of units in this layer
        self.out_size = out_size  # num of units in next layer for constructution of weight matrix

        # self.e_activation = torch.zeros((layer_size))  # activation level of e units
        # self.r_activation = torch.zeros((layer_size))  # activation level of r units
        # self.r_output = torch.zeros((layer_size))  # output firing rates of r units
        # self.r_prediction = torch.zeros((layer_size))  # prediction imposed by higher layer

        self.infRate = inf_rate  # inference rate governing how fast r adjust to errors
        self.actFunc = act_func  # activation function
        self.learn_rate = lr_rate  # learning rate

        self.weights = torch.empty((layer_size, out_size), **factory_kwargs)
        self.reset_parameters()
        self.weights = nn.Parameter(self.weights, requires_grad=False)
        # self.reset_state()

    def reset_parameters(self) -> None:  # initialise or reset layer weight
        nn.init.normal_(self.weights, 0, 0.5)  # normal distribution
        self.weights = torch.clamp(self.weights, min=0)  # weights clamped to above 0
        self.weights = self.weights / self.out_size  # normalise weights given next layer size

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
        r_out = self.actFunc(r_act)  # Apply the activation function to get neuronal output
        return e_act, r_act, r_out

    def w_update(self, e_act, nextlayer_output):
        # Learning step
        delta = self.learn_rate * torch.matmul(e_act.reshape(-1, 1), nextlayer_output.reshape(1, -1))
        self.weights = nn.Parameter(torch.clamp(self.weights + delta, min=0))  # Keep only positive weights


class input_layer(PredLayer):
    # Additional class for the input layer. This layer does not use a full inference step (driven only by input).
    def forward(self, inputs, nextlayer_r_out):
        e_act = inputs - torch.matmul(self.weights, nextlayer_r_out)
        return e_act


class output_layer(PredLayer):
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

class DHPC(nn.Module):
    def __init__(self, network_arch, inf_rates):
        super().__init__()
        e_act, r_act, r_out = [], [], []  # a list that always keep tracks of internal state values
        self.layers = nn.ModuleList()  # create module list containing all layers
        self.architecture = network_arch

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
                    input_layer(network_arch[0], network_arch[1], inf_rates[0]))  # append input layer to modulelist
            elif layer != len(network_arch) - 1:
                # add middle layer to module list and state list
                e_act.append(torch.zeros(network_arch[layer]).to(device))  # tensor containing activation of error units
                self.layers.append(PredLayer(network_arch[layer], network_arch[layer + 1], inf_rates[layer]))
            else:
                # add output layer to module list
                e_act.append(None)
                self.layers.append(output_layer(network_arch[layer], network_arch[layer], inf_rates[layer]))

        self.states = {
            'error': e_act,
            'r_activation': r_act,
            'r_output': r_out,
        }

    def init_states(self):
        # initialise values in state dict
        for i in range(len(self.states['r_activation'])):
            if i != len(self.architecture) - 1:
                self.states['error'][i] = torch.zeros(self.architecture[i]).to(device)
            self.states['r_activation'][i] = -2 * torch.ones(self.architecture[i]).to(device)
            self.states['r_output'][i] = self.layers[i].actFunc(self.states['r_activation'][i])

    def forward(self, frame, inference_steps):
        # frame is input to the lowest layer, inference steps
        e_act, r_act, r_out = self.states['error'], self.states['r_activation'], self.states['r_output']
        layers = self.layers
        r_act[0] = frame  # r units of first layer reflect input

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


# %%
###########################
#  evaluation functions
###########################

def test_frame(model, test_data, inference_steps):
    # test whether error converge for a single frame after inference
    total_error = []  # total error history as MSE of all error units in network
    recon_error = []  # reconstruction error as MSE of error units in the first layer

    model.init_states()

    e_act, r_act, r_out = model.states['error'], model.states['r_activation'], model.states['r_output']
    layers = model.layers
    r_act[0] = test_data  # r units of first layer reflect input

    # inference process
    for i in range(inference_steps):
        e_act[0] = layers[0](r_act[0], r_out[1])  # update first layer, given inputs, calculate error
        recon_error.append(torch.mean(e_act[0].pow(2)))
        for j in range(1, len(layers) - 1):  # iterate through middle layers, forward inference
            e_act[j], r_act[j], r_out[j] = layers[j](
                torch.matmul(torch.transpose(layers[j - 1].weights, 0, 1), e_act[j - 1]),
                r_act[j], r_out[j], r_out[j + 1])
        # update states of last layer
        r_act[-1], r_out[-1] = layers[-1](torch.matmul(torch.transpose(layers[-2].weights, 0, 1), e_act[-2]),
                                          r_act[-1])

        # calculate total error
        total = []
        for i in range(len(model.architecture) - 1):
            error = torch.mean(torch.pow(model.states['error'][i], 2))
            total.append(error)

        total_error.append(total)

    # plot total error history and reconstruction errors
    fig1, ax1 = plt.subplots()
    x = np.arange(inference_steps)
    ax1.plot(total_error)
    ax1.set_title('Total error history')
    ax1.set_ylim([0, None])
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(recon_error)
    ax2.set_title('Reconstruction error (first layer error)')
    ax2.set_ylim([0, None])
    plt.show()


def generate_rdm(model, test_sequences, inference_steps, plot=False):
    # test sequence include all frames of tested sequences
    sequences = len(test_sequences)
    frames = len(test_sequences[0])

    representation = []  # array containing representation from highest layer

    for seq in range(sequences):
        for frame in range(frames):
            model.init_states()  # initialise states between each frame
            model.forward(test_sequences[seq, frame, :], inference_steps)
            representation.append(model.states['r_activation'][-1].detach().numpy())

    pair_dist_cosine = pairwise_distances(representation, metric='cosine')

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(pair_dist_cosine)
        fig.colorbar(im, ax=ax)
        ax.set_title('RDM cosine')
        plt.show()

    return representation


# %%
# test function: takes the model, generates highest level representations, use KNN to classify
def test_accuracy(model, test_data):
    rep_list = generate_rdm(model, test_data, 1000, plot=False)
    label_list = []  # List with correct class labels
    for i in range(test_data.shape[0]):  # Iterate through sequences
        for j in range(test_data.shape[1]):  # Iterate through frames in each sequence
            label_list.append(i)  # Append the label (from 0 to 9)
    labels = np.stack(label_list, axis=0)
    reps = np.stack(rep_list, axis=0)

    # Select two samples of each class as test set, classify with knn (k = 3)
    skf = StratifiedKFold(n_splits=3)  # with six instances per class, each of the three folds contains two
    skf.get_n_splits(reps, labels)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(reps, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = reps[train_index], reps[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        labels_train_vec = np.zeros([40, 10])
        labels_test_vec = np.zeros([20, 10])
        for i in range(40):
            labels_train_vec[i, math.floor(i / 4)] = 1
        for i in range(20):
            labels_test_vec[i, math.floor(i / 2)] = 1
        # neigh = KNeighborsClassifier(n_neighbors=3, metric = distance) # build  KNN classifier for this fold
        # neigh.fit(reps_train, labels_train) # Use training data for KNN classifier
        # labels_predicted = neigh.predict(reps_test) # Predictions across test set

        reg = linear_model.LinearRegression()
        reg.fit(reps_train, labels_train_vec)
        labels_predicted = reg.predict(reps_test)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)
        # https://www.svds.com/the-basics-of-classifier-evaluation-part-1/
        cumulative_accuracy += accuracy / 3
    return cumulative_accuracy


# %%
#  data preprocessing
data = torch.tensor(np.load('data/translationData.npy'), device=device)  # [samples, sequencelength, dim_x, dim_y]
flat_data = torch.flatten(data, 2, 3)
flat_data = flat_data.to(device)
dataWidth = data.shape[-1]

# %%
# data visualisation
# fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
# for i in range(len(data[1, :, :, :])):
#     axs[i].imshow(data[1, i, :, :])
# plt.show()

# %%
###########################
### Training loop
###########################

with torch.no_grad():  # turn off auto grad function

    # Hyperparameters for training
    inference_steps = 10
    epochs = 5
    cycles_per_frame = 5
    cycles_per_sequence = 10

    #  network instantiation
    network_architecture = [dataWidth ** 2, 2000, 500, 30]
    inf_rates = [.05, .05, .05, .05]

    net = DHPC(network_architecture, inf_rates)
    net.to(device)

    sequences = len(flat_data)
    frames = len(flat_data[0])

    total_errors = []

    for epoch in range(epochs):
        error_per_seq = []
        for seq in range(sequences):  # for each sequence
            net.init_states()
            for s in range(cycles_per_sequence):  # several cycles per sequence
                for frame in range(frames):  # for each frame
                    for f in range(cycles_per_frame):  # several cycles per frame
                        data = flat_data[seq, frame, :]
                        net(data, inference_steps)
                        net.learn()
            error_per_seq.append(net.total_error())

        total_errors.append(np.mean(error_per_seq))
        test_accuracy(net, flat_data)

    fig, ax = plt.subplots()
    plt.plot(total_errors)
    plt.show()





# %%
# test_accuracy(net, flat_data)

# generate_rdm(net, flat_data, 10)
#  register_forward_hook can be used to inspect internal activation
