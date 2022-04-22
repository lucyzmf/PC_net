'''
this script takes the images from fashion mnist to generate datasets that can be fed into the network
input parameters from config: samples per class from fashion mnist, num of frames per sequences, morph type,
'''
import os
import pickle

import torchvision
import yaml
from sklearn.model_selection import train_test_split
# folder to load config file
from torch.utils import data
from torch.utils.data import Subset

from util import *

# TODO: rewrite datageneration script
CONFIG_PATH = "../scripts/"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

data_dir = config['dataset_dir']  # where to save train and test data
dataset = config['dataset_type']
train_size = config['train_size']  # training dataset size
test_size = config['test_size']  # test dataset size
frames_per_sequence = config['frame_per_sequence']
padding = config['padding_size']
data_width = 28 + padding * 2
num_classes = config['num_classes']

# download data
if dataset == 'MNIST':
    full_dataset = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(padding)])
    )
else:
    full_dataset = torchvision.datasets.FashionMNIST(
        root='./data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(padding)])
    )

# %%
# get only certain number of classes
targets = torch.randperm(10)
if num_classes < 10:
    targets = targets[:num_classes]
    idx = 0
    for t in range(len(targets)):
        idx += full_dataset.targets == targets[t]
    # full_dataset = Subset(full_dataset, np.where(idx == 1)[0])
    indices = np.where(idx == 1)[0]
else:
    indices = np.arange(len(full_dataset))

# %%
train_indices, test_indices = train_test_split(indices, train_size=train_size * len(targets), test_size=test_size * len(targets),
                                               stratify=full_dataset.targets[indices])

# Warp into Subsets and DataLoaders
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

torch.save(train_dataset, os.path.join(data_dir, str(dataset) + 'train_image.pt'))
torch.save(test_dataset, os.path.join(data_dir, str(dataset) + 'test_image.pt'))

# %%
##########################
# spinning sequences
##########################
# iterate through train and test dataset samples to generate spinning sequences
anticlockwise_deg = [-20, -15, -10, -5, 0, 5, 10, 15, 20]  # the degrees of rotation for each frame in seq
clockwise_deg = np.flip(anticlockwise_deg)
degrees = np.vstack((clockwise_deg, anticlockwise_deg))

rotation_axis = 2  # vertical and horizontal
direction = 2  # clockwise and anticlockwise

f = (2 * data_width ** 2) ** .5  # focal length for projection

# TODO: keep track of which rotation type and what direction per seq in a separate file


# %%
# generate train sequence
train_x, train_y, train_log = generate_spin_sequence(train_dataset, rotation_axis, direction, degrees,
                                                     frames=frames_per_sequence, w=data_width, focal=f)

# %%
# shuffle sequence, targets, and log
train_x, train_y, train_log['rotation_axis'], train_log['direction'] = shuffle(
    [train_x, train_y, train_log['rotation_axis'], train_log['direction']])

# %%
# collapse first two dimensions of train_x, add dimension to train_y
train_x = train_x.view(-1, data_width, data_width)
train_y = torch.unsqueeze(train_y, dim=1)
train_y = torch.flatten(train_y.repeat(1, frames_per_sequence))

# %%
# visualisation confirmation for correct dataset generation
# rand = [40, 45, 50, 55]
#
# fig, axs = plt.subplots(len(rand), 5, sharey=True)
# for k in range(len(rand)):
#     collage = train_x[rand[k]:rand[k] + 5]
#     for i in range(5):
#         axs[k][i].imshow(collage[i])
#
# plt.show()


# %%
# generate test sequence
test_x, test_y, test_log = generate_spin_sequence(test_dataset, rotation_axis, direction, degrees,
                                                  frames=frames_per_sequence, w=data_width, focal=f)

# shuffle sequence, targets, and log
test_x, test_y, test_log['rotation_axis'], test_log['direction'] = shuffle(
    [test_x, test_y, test_log['rotation_axis'], test_log['direction']])

# %%
# collapse first two dimensions of train_x, add dimension to train_y
test_x = test_x.view(-1, data_width, data_width)
test_y = torch.unsqueeze(test_y, dim=1)
test_y = torch.flatten(test_y.repeat(1, frames_per_sequence))

# %%
# save spin dataset to correct directory
train_set_spin = data.TensorDataset(train_x, train_y)
# train_loader_spin = data.DataLoader(train_set_spin, batch_size=batchSize, num_workers=n_workers, pin_memory=pin_mem)
test_set_spin = data.TensorDataset(test_x, test_y)
# test_loader_spin = data.DataLoader(test_set_spin, batch_size=batchSize, num_workers=n_workers, pin_memory=pin_mem)

torch.save(train_set_spin, os.path.join(data_dir, str(dataset) + 'train_set_spin.pt'))
torch.save(test_set_spin, os.path.join(data_dir, str(dataset) + 'test_set_spin.pt'))

# %%
# save log of stimuli rotation type and direction to csv

with open(data_dir + '/train_log.pkl', 'wb') as f:
    pickle.dump(train_log, f)

with open(data_dir + '/test_log.pkl', 'wb') as f:
    pickle.dump(test_log, f)

#
# %%

# # translation from distance change
# dist = [-10, -5, 0, 5, 10]
# fig, axs = plt.subplots(len(dist), 1, sharey=True, figsize=(5, 20))
# for i in range(len(dist)):
#     axs[i].imshow(transform(sample, translation=(0, 0, dist[i])))
# plt.show()
