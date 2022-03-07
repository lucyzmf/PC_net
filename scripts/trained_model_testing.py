'''
This script tests trained models
'''
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas
import torchvision
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
import torchvision
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from network import DHPC, sigmoid
from evaluation import *
import seaborn as sns
import os

file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/pilot new learning paradigm/nomem_10sample_lyricwood89')


if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
else:
    dev = "cpu"
    print("Cuda not available")
device = torch.device(dev)

dtype = torch.float  # Set standard datatype


#  pytorch logistic regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


# %%
#  load the training set used during training
train_loader = torch.load(os.path.join(file_path, 'train_loader.pth'))
test_loader = torch.load(os.path.join(file_path, 'test_loader.pth'))

# %%

dataWidth = 28  # width of mnist
numClass = 10  # number of classes in mnist

# %%
# load trained model
# Hyperparameters for training
inference_steps = 100
epochs = 200

#  network instantiation
network_architecture = [dataWidth ** 2, 1000, 50]
inf_rates = [.2, .14, .1]  # double the value used during training to speed up convergence
lr = .05
per_im_repeat = 1

trained_net = DHPC(network_architecture, inf_rates, lr=lr, act_func=sigmoid, device=device, dtype=dtype)
trained_net.load_state_dict(torch.load(os.path.join(file_path,
                                                    'no_mem[784, 1000, 50][0.1, 0.07, 0.05]readout.pth'),
                                       map_location=device))
trained_net.eval()


# %%
# distribution of trained weights

fig, axs = plt.subplots(1, len(network_architecture) - 1, figsize=(12, 4))
for i in range(len(network_architecture) - 1):
    axs[i].hist(trained_net.layers[i].weights)
    axs[i].set_title('layer'+str(i))

plt.show()
fig.savefig(os.path.join(file_path, 'weight_dist'))

# %%
# code from online
classifier = LogisticRegression(network_architecture[-1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# generate representations using train and test images

train_x = []  # contains last layer representations learned from training data
train_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(train_loader):
    train_x.append(high_level_rep(trained_net, torch.flatten(_image), 1000))
    train_y.append(_label)
train_x, train_y = torch.stack(train_x), torch.cat(train_y)
dataset = data.TensorDataset(train_x, train_y)
train_loader_rep = DataLoader(dataset, shuffle=True)


test_x = []  # contains last layer representations learned from testing data
test_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(test_loader):
    test_x.append(high_level_rep(trained_net, torch.flatten(_image), 1000))
    test_y.append(_label)
test_x, test_y = torch.stack(test_x), torch.cat(test_y)
dataset = data.TensorDataset(test_x, test_y)
test_loader_rep = DataLoader(dataset, shuffle=True)

# %%
# train and test classifier

epochs = 300
iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader_rep):
        images = Variable(images.view(-1, network_architecture[-1]))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader_rep:
                images = Variable(images.view(-1, network_architecture[-1]))
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

# %%
# sort representations by class first before passing to rdm function
train_indices = np.argsort(train_y)
train_x = train_x[train_indices]

test_indices = np.argsort(test_y)
test_x = test_x[test_indices]

rdm_train = rdm_w_rep(train_x, 'cosine', istrain=True)
rdm_train.savefig(os.path.join(file_path, 'rdm_train.png'))
rdm_test = rdm_w_rep(test_x, 'cosine', istrain=False)
rdm_test.savefig(os.path.join(file_path, 'rdm_test.png'))

# %%
# tSNE clustering
print('tSNE clustering')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(train_x)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# %%
# visualisation
df = pandas.DataFrame()
df['tsne-one'] = tsne_results[:, 0]
df['tsne-two'] = tsne_results[:, 1]
df['y'] = train_y
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="y",
    palette=sns.color_palette("bright", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)

plt.show()
fig.savefig(os.path.join(file_path, 'tSNE_clustering_rep'))
