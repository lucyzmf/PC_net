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
pin_mem = config['pin_mem']
batchSize = config['batch_size']
n_workers = config['num_workers']

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
# transformation
# code from https://towardsdatascience.com/how-to-transform-a-2d-image-into-a-3d-space-5fc2306e3d36
def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):
    # get the values on each axis
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
    return image


# %%
# rotation around y
rotated_angle = [-20, -10, 0, 10, 20]
fig, axs = plt.subplots(len(rotated_angle), 1, sharey=True, figsize=(5, 20))
for i in range(len(rotated_angle)):
    rot_sample = transform(sample, translation=(rotated_angle[i]*.85, 0, 0), rotation=(0, rotated_angle[i], 0))
    axs[i].imshow(rot_sample)
plt.show()

# %%
# rotation around x
rotated_angle = [-20, -10, 0, 10, 20]
fig, axs = plt.subplots(len(rotated_angle), 1, sharey=True, figsize=(5, 20))
for i in range(len(rotated_angle)):
    rot_sample = transform(sample, translation=(0, rotated_angle[i]*.85, 0), rotation=(rotated_angle[i], 0, 0))
    axs[i].imshow(rot_sample)
plt.show()

# %%
plt.imshow(transform(sample, translation=(40, 0, 0), rotation=(0, 40, 0)))
plt.show()