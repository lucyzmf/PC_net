import numpy as np
import torch
from skimage.transform import warp

# %%
from torch import optim


def create_w_optimizers(net, lr):
    w_optimizer = []

    for l in range(len(net.architecture) - 1):
        w_optimizer += [optim.SGD([net.layers[l].weights], lr=lr)]

    return w_optimizer


def reset_w_grads(w_optimizer):
    n_optimizers = len(w_optimizer)

    for l in range(n_optimizers):
        w_optimizer[l].zero_grad()


def compute_loss(update_for, net, criterion):
    losses = []
    if update_for == 'weights':
        for l in range(len(net.architecture) - 1):
            losses += [criterion(net.states['r_output'][l],
                                 torch.matmul(net.layers[l].weights, net.states['r_output'][l+1]))]

    if len(losses) != len(net.architecture)-1:
        raise Exception('loss computation is wrong')

    return losses


def compute_gradient(net, losses):
    n_layer = len(losses)

    for l in range(n_layer):
        torch.autograd.backward(net.layers[l].weights, grad_tensors=net.layers[l].weights.grad, retain_graph=True)


def w_update(w_optimizer):
    n_optimizers = len(w_optimizer)

    for l in range(n_optimizers):
        w_optimizer[l].step()


# %%
# transformation function
# code from https://towardsdatascience.com/how-to-transform-a-2d-image-into-a-3d-space-5fc2306e3d36


def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1)):
    # get the values on each axis
    image = torch.squeeze(image)
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
    image = warp(image, M)

    return torch.tensor(image)


# %%
# function that generates sequences given a dataset (either train or test)
# returns: images and class labels that can be loaded into dataloader, a second dictionary that logs the type of each seq
# these two outputs should have corresponding index for later referencing

def generate_spin_sequence(dataset, rotation_axis, direction, degrees, frames, w, focal):
    data_seq = []
    labels = []

    # create arrays that log seq type
    clock_anticlock = []  # each element logs whether seq rotates clockwise (0) or anticlockwise (1)
    verti_hori = []  # each element logs whether seq spins along vertical (0) or horizontal axis (1)

    for i, (_image, _label) in enumerate(dataset):
        _sample = []
        for axis in range(rotation_axis):  # for each rotation axes
            for direc in range(direction):  # for each rotation direction
                verti_hori.append(axis)
                clock_anticlock.append(direc)
                seq = torch.empty(frames, w, w)
                deg = degrees[direc]
                for fr in range(len(seq)):
                    if axis == 0:
                        seq[fr] = transform(_image,
                                            translation=(np.sin(np.deg2rad(deg[fr])) * focal, 0,
                                                         (1 - np.cos(np.deg2rad(deg[fr]))) * focal),
                                            rotation=(0, deg[fr], 0))
                    else:
                        seq[fr] = transform(_image,
                                            translation=(0, np.sin(np.deg2rad(deg[fr])) * focal,
                                                         (1 - np.cos(np.deg2rad(deg[fr]))) * focal),
                                            rotation=(deg[fr], 0, 0))

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
def shuffle(data_list):  # take tensor datasets and targets and returns shuffled datasets and targets
    idx = torch.randperm(len(data_list[0]))
    for i in range(len(data_list)):
        data_list[i] = data_list[i][idx]

    return data_list
