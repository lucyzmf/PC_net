# %%
###########################
#  evaluation functions
###########################
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


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


def generate_rdm(model, data_loader, inf_steps):  # generate rdm to inspect learned high level representation with either train or test data
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


def high_level_rep(model, image, inference_steps):
    model.init_states()
    model.forward(torch.flatten(image), inference_steps)
    return model.states['r_output'][-1].detach()


# %%
# linear classifier fitted on all samples and tested with stratified kfold knn
def linear_classifier_kfold(train_images, train_labels, test_images, test_labels):
    rep_list = np.concatenate((train_images, test_images))
    labels_list = np.concatenate((train_labels, test_labels))

    # Select two samples of each class as test set, classify with knn (k = 5)
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # split into 5 folds
    skf.get_n_splits(rep_list, labels_list)
    # sample_size = len(data_loader)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(rep_list, labels_list):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = rep_list[train_index], rep_list[test_index]
        labels_train, labels_test = labels_list[train_index], labels_list[test_index]
        labels_train_vec = F.one_hot(torch.tensor(labels_train)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(labels_test)).numpy()

        neigh = KNeighborsClassifier(n_neighbors=5)  # build  KNN classifier for this fold
        neigh.fit(reps_train, labels_train)  # Use training data for KNN classifier
        labels_predicted = neigh.predict(reps_test)  # Predictions across test set
        accuracy = neigh.score(reps_test, labels_test)  # score predictions on test set

        # reg = linear_model.LinearRegression()
        # reg.fit(reps_train, labels_train_vec)
        # labels_predicted = reg.predict(reps_test)

        # Convert to one-hot
        # labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        # accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy += accuracy / 5

    return rep_list, labels_list, cumulative_accuracy

# %%
# linear classifier fitted to train loader images and tested on test loader images
def linear_classifier(train_images, train_labels, test_images, test_labels):
    # avg classification performance over 10 rounds
    cumulative_accuracy_train = 0
    cumulative_accuracy_test = 0
    for i in range(10):
        labels_train_vec = F.one_hot(torch.tensor(train_labels)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(test_labels)).numpy()

        reg = linear_model.LinearRegression()
        reg.fit(train_images, labels_train_vec)
        # assess train set acc
        labels_predicted_train = reg.predict(train_images)
        labels_predicted_train = (labels_predicted_train == labels_predicted_train.max(axis=1, keepdims=True)).astype(int)
        acc_train = accuracy_score(labels_train_vec, labels_predicted_train)
        cumulative_accuracy_train += acc_train / 10

        # assess performance on test set
        labels_predicted = reg.predict(test_images)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy_test += accuracy / 10

    return cumulative_accuracy_train, cumulative_accuracy_test


# %%
# generate rdm with representations

def rdm_w_rep(representations, metric_type, istrain):
    pair_dist_cosine = pairwise_distances(representations, metric=metric_type)

    fig, ax = plt.subplots()
    im = ax.imshow(pair_dist_cosine)
    fig.colorbar(im, ax=ax)
    if istrain:
        ax.set_title('RDM cosine train data')
    else:
        ax.set_title('RDM cosine test data')
    plt.show()

    return fig
