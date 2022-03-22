'''
this script takes the images from fashion mnist to generate datasets that can be fed into the network
input parameters from config: samples per class from fashion mnist, num of frames per sequences, morph type,
'''
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from sklearn.model_selection import train_test_split
# folder to load config file
from torch.utils.data import Subset

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
pin_mem = config['pin_mem']
batchSize = config['batch_size']
n_workers = config['num_workers']
data_width = 28
num_classes = 10

# download data
if dataset == 'MNIST':
    full_dataset = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
else:
    full_dataset = torchvision.datasets.FashionMNIST(
        root='./data/',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

# %%
indices = np.arange(len(full_dataset))
train_indices, test_indices = train_test_split(indices, train_size=train_size * 10, test_size=test_size * 10,
                                               stratify=full_dataset.targets)

# Warp into Subsets and DataLoaders
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)


# %%
# transformation function
# code from https://towardsdatascience.com/how-to-transform-a-2d-image-into-a-3d-space-5fc2306e3d36
def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):
    # get the values on each axis
    image = image.numpy()  # change tensor into numpy array

    t_x, t_y, t_z = translation
    r_x, r_y, r_z = rotation
    sc_x, sc_y, sc_z = scaling

    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)

    # get the height and the width of the image
    h, w = image.shape[:2]
    # compute its diagonal
    diag = (h ** 2 + w ** 2) ** 0.5
    # compute the focal length
    f = diag
    if np.sin(theta_rz) != 0:
        f /= 2 * np.sin(theta_rz)

    # set the image from cartesian to projective dimension
    H_M = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 1],
                    [0, 0, 1]])
    # set the image projective to carrtesian dimension
    Hp_M = np.array([[f, 0, w / 2, 0],
                     [0, f, h / 2, 0],
                     [0, 0, 1, 0]])
    """
            We will define our matrices here in next parts
                                                            """
    Identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    # adjust the translation on z
    t_z = (f - t_z) / sc_z ** 2
    # translation matrix to translate the image
    T_M = np.array([[1, 0, 0, t_x],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0, 1]])

    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    # get the rotation matrix on x axis
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz, cos_rz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # compute the full rotation matrix
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

    # get the scaling matrix
    Sc_M = np.array([[sc_x, 0, 0, 0],
                     [0, sc_y, 0, 0],
                     [0, 0, sc_z, 0],
                     [0, 0, 0, 1]])

    # compute the full transform matrix
    M = Identity
    M = np.dot(T_M, M)
    M = np.dot(R_M, M)
    M = np.dot(Sc_M, M)
    # M = np.dot(Sh_M, M)
    M = np.dot(Hp_M, np.dot(M, H_M))
    # apply the transformation
    image = cv2.warpPerspective(image, M, (w, h))
    return torch.tensor(image)


# %%
##########################
# spinning sequences
##########################
# iterate through train and test dataset samples to generate spinning sequences
anticlockwise_deg = [-20, -10, 0, 10, 20]  # the degrees of rotation for each frame in seq
clockwise_deg = np.flip(anticlockwise_deg)
degrees = np.vstack((clockwise_deg, anticlockwise_deg))

rotation_axis = 2  # vertical and horizontal
direction = 2  # clockwise and anticlockwise

f = (2 * 28 ** 2) ** .5  # focal length for projection


# TODO: keep track of which rotation type and what direction per seq in a separate file

# %%
# function that generates sequences given a dataset (either train or test)
# returns: images and class labels that can be loaded into dataloader, a second dictionary that logs the type of each seq
# these two outputs should have corresponding index for later referencing

def generate_spin_sequence(dataset, rotation_axis, direction, frames=frames_per_sequence, data_width=28):
    data_seq = []
    labels = []

    # create arrays that log seq type
    clock_anticlock = []  # each element logs whether seq rotates clockwise (0) or anticlockwise (1)
    verti_hori = []  # each element logs whether seq spins along vertical (0) or horizontal axis (1)

    for i, (_image, _label) in enumerate(dataset):
        _sample = []
        for axis in range(rotation_axis):  # for each rotation axes
            for dir in range(direction):  # for each rotation direction
                verti_hori.append(axis)
                clock_anticlock.append(dir)
                seq = torch.empty(frames, data_width, data_width)
                deg = degrees[dir]
                if axis == 0:
                    for fr in range(len(seq)):
                        seq[fr] = transform(_image,
                                            translation=(
                                                np.sin(np.deg2rad(deg[fr])) * f, 0,
                                                (1 - np.cos(np.deg2rad(deg[fr]))) * f),
                                            rotation=(0, rotated_angle[fr], 0))
                    labels.append(_label)
                    _sample.append(seq)
                else:
                    for fr in range(len(seq)):
                        seq[fr] = transform(_image,
                                            translation=(
                                                0, np.sin(np.deg2rad(deg[fr])) * f,
                                                (1 - np.cos(np.deg2rad(deg[fr]))) * f),
                                            rotation=(rotated_angle[fr], 0, 0))
                    labels.append(_label)
                    _sample.append(seq)
        _sample = torch.stack(_sample)
        data_seq.append(_sample)

    log = {
        'rotation_axis': torch.tensor(verti_hori),
        'direction': torch.tensor(clock_anticlock)
    }

    data_seq = torch.cat(data_seq)  # contatenate all tensors to create one big sensor containing frames of all seq
    labels = torch.tensor(labels)

    return data_seq, labels, log  # one label and one log per sequence


# %%
def shuffle(data):  # take tensor datasets and targets and returns shuffled datasets and targets
    idx = torch.randperm(len(data))
    data = data[idx]

    return data


# %%
# generate sequence
train_x, train_y, train_log = generate_spin_sequence(train_dataset, rotation_axis, direction)

# shuffle sequence, targets, and log
train_x = shuffle(train_x)
train_y = shuffle(train_y)
train_log['rotation_axis'] = shuffle(train_log['rotation_axis'])
train_log['direction'] = shuffle(train_log['direction'])



# TODO: save to dataset dir


# %%
# visualise 5 samples of the same class
idx = torch.tensor(full_dataset.train_labels) == 0
collage = full_dataset.data[idx]
collage = collage[:5]

fig, axs = plt.subplots(1, 5, sharey=True)
for i in range(5):
    axs[i].imshow(collage[i])

plt.show()

# %%
# trial sequence generation with one image
sample, label = train_dataset[0]
sample = torch.squeeze(sample).numpy()

# %%
plt.imshow(sample)
plt.show()

# %%


# %%
# rotation around y
rotated_angle = [-20, -10, 0, 10, 20]
fig, axs = plt.subplots(len(rotated_angle), 1, sharey=True, figsize=(5, 20))
for i in range(len(rotated_angle)):
    rot_sample = transform(sample, translation=(
        np.sin(np.deg2rad(rotated_angle[i])) * f, 0, (1 - np.cos(np.deg2rad(rotated_angle[i]))) * f),
                           rotation=(0, rotated_angle[i], 0))
    axs[i].imshow(rot_sample)
plt.show()

# %%
# rotation around x
rotated_angle = [-20, -10, 0, 10, 20]
fig, axs = plt.subplots(len(rotated_angle), 1, sharey=True, figsize=(5, 20))
for i in range(len(rotated_angle)):
    rot_sample = transform(sample, translation=(
        0, np.sin(np.deg2rad(rotated_angle[i])) * f, (1 - np.cos(np.deg2rad(rotated_angle[i]))) * f),
                           rotation=(rotated_angle[i], 0, 0))
    axs[i].imshow(rot_sample)
plt.show()

# %%
# translation from distance change
dist = [-10, -5, 0, 5, 10]
fig, axs = plt.subplots(len(dist), 1, sharey=True, figsize=(5, 20))
for i in range(len(dist)):
    axs[i].imshow(transform(sample, translation=(0, 0, dist[i])))
plt.show()
