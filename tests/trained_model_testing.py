'''
This script tests trained models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import math
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
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
        outputs = self.linear(x)
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
        self.weights = self.weights / self.layer_size  # normalise weights given next layer size

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
        e_act = inputs.to(device) - torch.matmul(self.weights, nextlayer_r_out)
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
    def __init__(self, network_arch, inf_rates, lr):
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
                    input_layer(network_arch[0], network_arch[1], inf_rates[0],
                                lr_rate=lr))  # append input layer to modulelist
            elif layer != len(network_arch) - 1:
                # add middle layer to module list and state list
                e_act.append(torch.zeros(network_arch[layer]).to(device))  # tensor containing activation of error units
                self.layers.append(
                    PredLayer(network_arch[layer], network_arch[layer + 1], inf_rates[layer], lr_rate=lr))
            else:
                # add output layer to module list
                e_act.append(None)
                self.layers.append(output_layer(network_arch[layer], network_arch[layer], inf_rates[layer], lr_rate=lr))

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


def generate_rdm(model, data_loader,
                 inf_steps):  # generate rdm to inspect learned high level representation with either train or test data
    # test sequence include all frames of tested sequences
    representation = []  # array containing representation from highest layer
    labels = []

    for i, (_image, _label) in enumerate(data_loader):
        representation.append(high_level_rep(model, torch.flatten(_image), inf_steps))
        labels.append(_label)

    sorted_label, indices = torch.sort(torch.tensor(labels))
    representation = torch.stack(representation)
    representation = representation[indices]
    labels = torch.cat(labels)

    pair_dist_cosine = pairwise_distances(representation.cpu(), metric='cosine')

    fig, ax = plt.subplots()
    im = ax.imshow(pair_dist_cosine)
    fig.colorbar(im, ax=ax)
    ax.set_title('RDM cosine')
    #     plt.show()

    return representation, labels, fig  # these have been sorted by class label

# %%

def high_level_rep(model, image, inference_steps):
    model.init_states()
    model.forward(torch.flatten(image), inference_steps)
    return model.states['r_activation'][-1].detach()


# %%
# test function: takes the model, generates highest level representations, use KNN to classify
def test_accuracy(model, data_loader):
    rep_list, labels, _ = generate_rdm(model, data_loader, 10)
    rep_list = rep_list.cpu()
    labels = np.array(labels)
    #     print(labels)

    # Select two samples of each class as test set, classify with knn (k = 5)
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # split into 5 folds
    skf.get_n_splits(rep_list, labels)
    # sample_size = len(data_loader)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(rep_list, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = rep_list[train_index], rep_list[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        labels_train_vec = F.one_hot(torch.tensor(labels_train)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(labels_test)).numpy()

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

        cumulative_accuracy += accuracy / 5

    return cumulative_accuracy


def train_classifier(model, reg_classifier,
                     data_loader,
                     epochNum):  # take network, classifier, and training data as input, return trained classifier
    train_x = []  # contains last layer representations learned from training data
    train_y = []  # contains labels in training data
    for i, (_image, _label) in enumerate(data_loader):
        train_x.append(high_level_rep(model, torch.flatten(_image), 3000))
        train_y.append(_label)

    print('finished learning, start training classifier')
    train_x, train_y = torch.stack(train_x), torch.cat(train_y)
    dataset = data.TensorDataset(train_x, train_y)
    dataloader_classify = DataLoader(dataset, shuffle=True)
    loss_log = []

    reg_classifier.train()
    for _epoch in range(epochNum):
        loss_epoch = []
        for i, (_image, _label) in enumerate(dataloader_classify):
            optimizer.zero_grad()
            outputs = reg_classifier(_image.to(device))
            loss = criterion(outputs, _label.to(device))
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_log.append(np.mean(loss_epoch))

    fig, ax = plt.subplots()
    plt.plot(loss_log)
    plt.title('classifier loss training')
    plt.show()

    with torch.no_grad():
        reg_classifier.eval()
        correct = []
        total = 0
        for i, (_image, _label) in enumerate(dataloader_classify):
            _output = reg_classifier(_image.to(device))
            _, predicted = torch.max(F.softmax(_output, dim=1), 1)
            correct += (predicted.cpu() == train_y[i])
            total += 1

        _acc = np.sum(correct) / total  # prediction acc of trained classifier

    return _acc


def test_classifier(model, reg_classifier, data_loader):
    test_x = []  # contains last layer representations generated from test data
    test_y = []  # contains labels in test data

    for i, (_image, _label) in enumerate(data_loader):
        test_x.append(high_level_rep(model, torch.flatten(_image), 3000))
        test_y.append(_label)

    print('testing classifier')
    test_x, test_y = torch.stack(test_x), torch.cat(test_y)
    dataset = data.TensorDataset(test_x, test_y)
    dataloader_classify = DataLoader(dataset, shuffle=True)

    with torch.no_grad():
        reg_classifier.eval()
        correct = []
        total = 0
        for i, (_image, _label) in enumerate(dataloader_classify):
            _output = reg_classifier(_image.to(device))
            _, predicted = torch.max(F.softmax(_output, dim=1), 1)
            correct += (predicted.cpu() == test_y[i])
            total += 1

        _acc = np.sum(correct) / total  # prediction acc of trained classifier with test dataset

    return _acc


# %%
#  load the training set used during training
train_loader = torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/trained_model/threelayer_test2/train_loader.pth')
test_loader = torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/trained_model/threelayer_test2/test_loader.pth')

# %%

dataWidth = 28  # width of mnist
numClass = 10  # number of classes in mnist

# %%
# load trained model
# Hyperparameters for training
inference_steps = 100
epochs = 200

#  network instantiation
network_architecture = [dataWidth ** 2, 1000, 100]
inf_rates = [.4, .28, .2]  # double the value used during training to speed up convergence
lr = .05
per_im_repeat = 1

trained_net = DHPC(network_architecture, inf_rates, lr)
trained_net.load_state_dict(torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/trained_model/threelayer_test2/acc_train0.102acc_test0.17readout.pth',
    map_location=device))
trained_net.eval()


# %%
# distribution of trained weights

fig, axs = plt.subplots(1, len(network_architecture) - 1, figsize=(12, 4))
for i in range(len(network_architecture) - 1):
    axs[i].hist(trained_net.layers[i].weights)

plt.show()

# %%
# code from online
classifier = LogisticRegression(100, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# generate representations using train and test images

train_x = []  # contains last layer representations learned from training data
train_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(train_loader):
    train_x.append(high_level_rep(trained_net, torch.flatten(_image), 700))
    train_y.append(_label)
train_x, train_y = torch.stack(train_x), torch.cat(train_y)
dataset = data.TensorDataset(train_x, train_y)
train_loader_rep = DataLoader(dataset, shuffle=True)


train_x = []  # contains last layer representations learned from training data
train_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(test_loader):
    train_x.append(high_level_rep(trained_net, torch.flatten(_image), 700))
    train_y.append(_label)
train_x, train_y = torch.stack(train_x), torch.cat(train_y)
dataset = data.TensorDataset(train_x, train_y)
test_loader_rep = DataLoader(dataset, shuffle=True)

# %%
# train and test classifier

epochs = 300
iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader_rep):
        images = Variable(images.view(-1, 100))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader_rep:
                images = Variable(images.view(-1, 100))
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))