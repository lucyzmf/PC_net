# %%
###########################
#  evaluation functions
###########################
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")


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


def high_level_rep(model, image, inference_steps):
    model.init_states()
    model.forward(torch.flatten(image), inference_steps)
    return model.states['r_output'][-1].detach()


# %%
# linear classifier using knn to test generalisation
def knn_classifier(train_images, train_labels, test_images, test_labels):
    cum_acc_train = 0
    cum_acc_test = 0

    for i in range(10):
        neigh = KNeighborsClassifier(n_neighbors=5)  # build  KNN classifier for this fold
        neigh.fit(train_images, train_labels)  # Use training data for KNN classifier
        # access acc on training data
        cum_acc_train += neigh.score(train_images, train_labels) / 10

        # labels_predicted = neigh.predict(test_images)  # Predictions across test set
        # access acc on test data
        cum_acc_test += neigh.score(test_images, test_labels) / 10  # score predictions on test set

    return cum_acc_train, cum_acc_test


# %%
# linear regression to test generalisation
def linear_regression(train_images, train_labels, test_images, test_labels):
    # avg classification performance over 10 rounds
    cumulative_accuracy_train = 0
    cumulative_accuracy_test = 0
    for i in range(20):
        labels_train_vec = F.one_hot(torch.tensor(train_labels)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(test_labels)).numpy()

        reg = linear_model.LinearRegression()
        reg.fit(train_images, labels_train_vec)
        # assess train set acc
        labels_predicted_train = reg.predict(train_images)
        labels_predicted_train = (labels_predicted_train == labels_predicted_train.max(axis=1, keepdims=True)).astype(
            int)
        acc_train = accuracy_score(labels_train_vec, labels_predicted_train)
        cumulative_accuracy_train += acc_train / 20

        # assess performance on test set
        labels_predicted = reg.predict(test_images)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy_test += accuracy / 20

    return cumulative_accuracy_train, cumulative_accuracy_test


# %%
def within_sample_classification_stratified(representations, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # split into 5 folds
    skf.get_n_splits(representations, labels)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(representations, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = representations[train_index], representations[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        labels_train_vec = F.one_hot(torch.tensor(labels_train)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(labels_test)).numpy()

        reg = linear_model.LinearRegression()
        reg.fit(reps_train, labels_train_vec)

        # assess performance on test set
        labels_predicted = reg.predict(reps_test)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy += accuracy / 5

    return cumulative_accuracy


# %%
# compute pairwise distance matrix
def cosine_dis(reps, labels, metric_type):
    idx = np.argsort(labels)
    reps = reps[idx]
    pair_dist_cosine = pairwise_distances(reps, metric=metric_type)
    return pair_dist_cosine

# %%
# generate rdm with representations

def rdm_w_rep(representations, labels, metric_type, ticklabel, title):  # inputs here are unordered
    idx = np.argsort(labels)
    representations = representations[idx]
    pair_dist_cosine = pairwise_distances(representations, metric=metric_type)

    fig, ax = plt.subplots()
    sns.heatmap(pair_dist_cosine, xticklabels=ticklabel, yticklabels=ticklabel, ax=ax, cmap='viridis',
                cbar_kws={'label': 'cosine distance'})
    # fig.colorbar(im, ax=ax)
    ax.set_title(title)
    # plt.show()

    return fig


# %%
def rdm_w_rep_title(representations, labels, metric_type, title):
    idx = np.argsort(labels)
    representations = representations[idx]
    pair_dist_cosine = pairwise_distances(representations, metric=metric_type)

    fig, ax = plt.subplots()
    im = ax.imshow(pair_dist_cosine)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    # plt.show()

    return fig


# %%
def plot_tsne(reps, labels, title):
    print('tSNE clustering')
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=400)
    tsne_results = tsne.fit_transform(np.vstack(reps))
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #
    # # %%
    # # visualisation
    df = pd.DataFrame()
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['y'] = labels
    fig, ax1 = plt.subplots(figsize=(7, 5))
    sns.despine()
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="y",
        palette=sns.color_palette("bright", config['num_classes']),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )

    h, _ = ax1.get_legend_handles_labels()
    ax1.legend(h, ['T-shirt/top', 'Trouser', 'Dress', 'Sneaker', 'Bag'], frameon=False)
    plt.title(title)

    return fig

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def pca_vis_2d(reps, labels):
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(reps)
    plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=labels, cmap='Paired')
    plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(labels)))

    return plt


def pca_vis_3d(reps, labels):
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(reps)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Xt[:, 1], Xt[:, 2], Xt[:, 3], c=labels, cmap='Paired')
    # ax.legend(handles=ax.legend_elements()[0], labels=list(np.unique(labels)))

    return fig


# %%

def append_rep_layer(model, target, _tr, _r_act, _r_out, _e_out, _is_train, _layer, _labels):
    for l in range(len(model.architecture)):
        _is_train.append(_tr)
        _layer.append(l)
        _labels.append(int(target.cpu().numpy()))
        # reps from false net
        _r_act.append(model.states['r_activation'][l].detach().cpu().numpy())
        _r_out.append(model.states['r_output'][l].detach().cpu().numpy())

        if l == (len(model.architecture) - 1):
            _e_out.append(np.zeros(model.layers[l].layer_size))
        else:
            _e_out.append(model.states['error'][l].detach().cpu().numpy())


def generate_reps(model, _dataloaders, infSteps, resetPerFrame):
    print('function called')
    is_train, layer, labels = [], [], []  # whether rep is generated from training set
    # contains reps generated without resetting per frame seq dataset
    r_act = []
    r_out = []
    e_out = []

    df_rep = pd.DataFrame()

    with torch.no_grad():
        model.init_states()
        for loader in range(len(_dataloaders)):
            print(len(_dataloaders[loader]))
            tr = 1 if loader == 0 else 0  # log whether rep is generated from train or test set
            for i, (_image, _label) in enumerate(_dataloaders[loader]):
                # print(i)
                model(_image, infSteps, istrain=False)
                if not resetPerFrame:
                    if (i + 1) % config['frame_per_sequence'] == 0:  # at the end of eqch sequence
                        append_rep_layer(model, _label, tr, r_act, r_out, e_out, is_train, layer, labels)

                        print('%i seqs done' % ((i + 1) / 9), len(is_train))
                        model.init_states()
                else:
                    append_rep_layer(model, _label, tr, r_act, r_out, e_out, is_train, layer, labels)
                    model.init_states()
                    if i % 20 == 0:
                        print('%i frames done' % i)

    df_rep['is_train'] = is_train
    df_rep['layer'] = layer
    df_rep['r_out'] = r_out
    df_rep['r_act'] = r_act
    df_rep['e_out'] = e_out
    df_rep['labels'] = labels

    return df_rep

# %%
def get_layer_gen_acc(dataframe, _layer):
    acc_train, acc_test = linear_regression(
        np.vstack(dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['r_out'].to_numpy()),
        dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['labels'].to_numpy(),
        np.vstack(dataframe[dataframe['is_train'] == 0][dataframe['layer'] == _layer][
                      'r_out'].to_numpy()),
        dataframe[dataframe['is_train'] == 0][dataframe['layer'] == _layer]['labels'].to_numpy()
    )
    return acc_train, acc_test


# %%
def get_layer_clustering(dataframe, _layer):
    cum_acc = within_sample_classification_stratified(
        np.vstack(dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['r_out'].to_numpy()),
        dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['labels'].to_numpy())

    return cum_acc

# %%
def generate_acc_df(rep_df, conditions_code, condition_name, isGen):  # conditions include a list that codes the conditions
    acc = []
    accIsTest = []  # train 0, test 1
    conditions_log = []
    by_layer = []

    for m in range(len(rep_df)):
        for i in range(4):
            if isGen:
                train, test = get_layer_gen_acc(rep_df[m], i)
                acc.append(train)
                accIsTest.append(0)
                by_layer.append(i)
                conditions_log.append(conditions_code[m])

                acc.append(test)
                accIsTest.append(1)
                by_layer.append(i)
                conditions_log.append(conditions_code[m])
            else:
                by_layer.append(i)
                conditions_log.append(conditions_code[m])
                acc.append(get_layer_clustering(rep_df[m], i))


    df_acc = pd.DataFrame()
    df_acc['acc'] = acc
    df_acc['layer'] = by_layer
    df_acc[condition_name] = conditions_log
    if isGen:
        df_acc['accIsTest'] = accIsTest

    return df_acc
