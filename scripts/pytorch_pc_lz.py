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
import torchvision
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import pairwise_distances  # computes the pairwise distance between observations
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
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

#  sigmoid activation function
def sigmoid(inputs):
    inputs = inputs - 3
    m = nn.Sigmoid()
    return m(inputs)


#  pytorch logistic regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs



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


def generate_rdm(model, data_loader, inf_steps, plot=True):  # generate rdm to inspect learned high level representation with either train or test data
    # test sequence include all frames of tested sequences
    representation = []  # array containing representation from highest layer
    labels = []

    for i, (_image, _label) in enumerate(data_loader):
        representation.append(high_level_rep(model, torch.flatten(_image), inf_steps))
        labels.append(_label)

    sorted_label, indices = torch.sort(torch.tensor(labels))
    representation = torch.tensor(representation)
    representation = representation[indices]

    pair_dist_cosine = pairwise_distances(representation, metric='cosine')

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(pair_dist_cosine)
        fig.colorbar(im, ax=ax)
        ax.set_title('RDM cosine')
        plt.show()


def high_level_rep(model, image, inference_steps):
    model.init_states()
    model.forward(torch.flatten(image), inference_steps)
    return model.states['r_activation'][-1].detach()


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


def train_classifier(model, reg_classifier, data_loader):  # take network, classifier, and training data as input, return trained classifier
    train_x = []  # contains last layer representations learned from training data
    train_y = []  # contains labels in training data
    for i, (_image, _label) in enumerate(data_loader):
        train_x.append(high_level_rep(model, torch.flatten(_image), 1000))
        train_y.append(_label)

    train_x, train_y = torch.tensor(train_x), F.one_hot(torch.tensor(train_y), num_classes=10)
    loss_log = []

    for _epoch in range (50):
        reg_classifier.train()
        optimizer.zero_grad()
        outputs = reg_classifier(train_x)
        loss = criterion(outputs, train_y)
        loss_log.append(loss)
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots()
    plt.plot(loss_log)
    plt.title('classifier loss training')
    plt.show()

    with torch.no_grad():
        reg_classifier.eval()
        _output = reg_classifier(train_x)
        _, predicted = torch.max(_output, 1)
        correct = torch.sum(predicted.cpu() == train_y)
        _acc = correct/len(train_y)  # prediction acc of trained classifier

    return _acc


def test_classifier(model, reg_classifier, data_loader):
    test_x = []  # contains last layer representations learned from training data
    test_y = []  # contains labels in training data
    for i, (_image, _label) in enumerate(data_loader):
        test_x.append(high_level_rep(model, torch.flatten(_image), 1000))
        test_y.append(_label)

    test_x, test_y = torch.tensor(test_x), F.one_hot(torch.tensor(test_y), num_classes=10)

    with torch.no_grad():
        reg_classifier.eval()
        _output = reg_classifier(test_x)
        _, predicted = torch.max(_output, 1)
        correct = torch.sum(predicted.cpu() == test_y)
        _acc = correct/len(test_y)  # prediction acc of trained classifier

    return _acc


# %%
#  data preprocessing

full_mnist = torchvision.datasets.MNIST(
    root="/Users/lucyzhang/Documents/research/PC_net/data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# %%

### genearte train test files for digit classification

indices = np.arange(len(full_mnist))
train_indices, test_indices = train_test_split(indices, train_size=100 * 10, test_size=20 * 10,
                                               stratify=full_mnist.targets)

# Warp into Subsets and DataLoaders
train_dataset = Subset(full_mnist, train_indices)
test_dataset = Subset(full_mnist, test_indices)

dataWidth = train_dataset[0][0].shape[1]
numClass = 10  # number of classes in mnist
# %%

train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=True)

# %%
###########################
### Training loop
###########################

# with torch.no_grad():  # turn off auto grad function

# Hyperparameters for training
inference_steps = 10
epochs = 2

#  network instantiation
network_architecture = [dataWidth ** 2, 2000, 500, 30]
inf_rates = [.05, .05, .05, .05]

net = DHPC(network_architecture, inf_rates)
net.to(device)

#  initialising classifier
classifier = LogisticRegression(network_architecture[-1], 10)
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(classifier.parameters(), lr=.01)

# values logged during training
total_errors = []
train_acc_history = []
test_acc_history = []  # acc on test set at the end of each epoch

for epoch in range(epochs):
    errors = []
    for i, (image, label) in enumerate(train_loader):
        net.init_states()
        net(torch.flatten(image), inference_steps)
        net.learn()
        errors.append(net.total_error())

    # train classifier using training data
    train_acc = train_classifier(net, classifier, train_loader)

    total_errors.append(np.mean(errors))  # mean error per epoch

    # test classification acc at the end of each epoch
    test_acc = test_classifier(net, classifier, test_loader)  # test classifier on test set (unseen data)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print('epoch: ', epoch, '. classifier training acc: ', train_acc, '. classifier test acc: ', test_acc)

fig, axs = plt.subplots(1, 3)
axs[0].plot(total_errors)
axs[0].set_title('Total Errors')
axs[1].plot(train_acc_history)
axs[1].set_title('train classification accuracy')
axs[2].plot(test_acc_history)
axs[2].set_title('test classification accuracy')
plt.show()

# %%
# test_accuracy(net, flat_data)

generate_rdm(net, train_loader, 1000, plot=True)
generate_rdm(net, test_loader, 1000, plot=True)
#  register_forward_hook can be used to inspect internal activation
