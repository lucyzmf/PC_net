'''
This script tests trained models
'''

import torchvision
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from network import DHPC, sigmoid
import torch.cuda as cuda
from evaluation import *
import torch.profiler
import glob

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
train_loader = torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/catmem_10sample_finewater88/train_loader.pth')
test_loader = torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/catmem_10sample_finewater88/test_loader.pth')

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
trained_net.load_state_dict(torch.load(
    '/Users/lucyzhang/Documents/research/PC_net/results/catmem_10sample_finewater88/[784, 1000, 50][0.1, 0.07, 0.05]readout.pth',
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
classifier = LogisticRegression(network_architecture[-1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# generate representations using train and test images

cat_mem = torch.zeros(network_architecture[-1]).to(device)

train_x = []  # contains last layer representations learned from training data
train_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(train_loader):
    train_x.append(high_level_rep(trained_net, torch.flatten(_image), 700, cat_mem))
    train_y.append(_label)
train_x, train_y = torch.stack(train_x), torch.cat(train_y)
dataset = data.TensorDataset(train_x, train_y)
train_loader_rep = DataLoader(dataset, shuffle=True)


test_x = []  # contains last layer representations learned from testing data
test_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(test_loader):
    test_x.append(high_level_rep(trained_net, torch.flatten(_image), 700, cat_mem))
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

        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader_rep:
                images = Variable(images.view(-1, network_architecture[-1]))
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

# %%

rdm_w_rep(train_x, 'cosine')
rdm_w_rep(test_x, 'cosine')