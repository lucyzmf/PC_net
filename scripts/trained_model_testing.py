'''
This script evalute trained models
'''
import glob
import os

import pandas as pd
import seaborn as sns
import yaml
from torch import nn
from torch.utils import data

from evaluation import *
from fc_net import FcDHPC
from rf_net_cm import RfDHPC_cm

# TODO change representation generation function since on trained dataset it should be representations generated from sequences

# load config
CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
    file_path = '../results/80_epochs'
else:
    dev = "cpu"
    print("Cuda not available")
    file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs')
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
train_loader_spin = torch.load(os.path.join(file_path, 'fashionMNISTtrain_loader_spin.pth'))
test_loader_spin = torch.load(os.path.join(file_path, 'fashionMNISTtest_loader_spin.pth'))

# %%
train_seq_spin, train_labels_spin = [], []
for i, (_frame, _label) in enumerate(train_loader_spin):
    train_seq_spin.append(torch.flatten(_frame).data.numpy())
    train_labels_spin.append(_label.data)

train_seq_spin = np.vstack(train_seq_spin)
train_labels_spin = torch.concat(train_labels_spin).numpy()

# %%
test_seq_spin, test_labels_spin = [], []
for i, (_frame, _label) in enumerate(test_loader_spin):
    test_seq_spin.append(torch.flatten(_frame).data.numpy())
    test_labels_spin.append(_label.data)

test_seq_spin = np.vstack(test_seq_spin)
test_labels_spin = torch.concat(test_labels_spin).numpy()

# %%

dataWidth = 28 + 2 * config['padding_size']  # width of mnist + padding
numClass = 10  # number of classes in mnist

# %%
# load trained model
# Hyperparameters used during training
inference_steps = config['infsteps']  # num infsteps per image
epochs = config['epochs']  # total training epochs
infrates = config['infrates']  # inf rates each layer
lr = config['learning_rate']  # lr for weight updates
arch = config['network_size']  # size of each layer
per_seq_repeat = config['per_seq_repeat']  # num of repeats per image/sequence
arch_type = config['architecture']

#  network instantiation
per_im_repeat = 1

# trained_net = DHPC(network_architecture, inf_rates, lr=lr, act_func=sigmoid, device=device, dtype=dtype)
if config['architecture'] == 'FcDHPC':
    trained_net_true = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                              act_func=config['act_func'],
                              device=device, dtype=dtype)
    trained_net_false = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                               act_func=config['act_func'],
                               device=device, dtype=dtype)
elif config['architecture'] == 'RfDHPC_cm':
    trained_net = RfDHPC_cm(config['network_size'], config['rf_sizes'], config['infrates'] * 4,
                            lr=config['learning_rate'],
                            act_func=config['act_func'],
                            device=device, dtype=dtype)

# %%
# # distribution of trained weights
#
# fig, axs = plt.subplots(1, len(arch) - 1, figsize=(12, 4))
# for i in range(len(arch) - 1):
#     axs[i].hist(trained_net.layers[i].weights)
#     axs[i].set_title('layer' + str(i))
#
# plt.show()
# fig.savefig(os.path.join(file_path + '/reset_per_frame_false/', 'weight_dist'))

# %%
# create dataframe of representations using train and test images
# TODO classification acc on r from all layers

print('generate representations')
# create data frame that has columns: is_train, layers, r_act, r_out, e_out, reset at the end of each sequence
trained_net_false.load_state_dict(
    torch.load(glob.glob(file_path + '/reset_per_frame_false/trained_model/*readout.pth')[0],
               map_location=torch.device('cpu')))
trained_net_false.eval()

trained_net_true.load_state_dict(
    torch.load(glob.glob(file_path + '/reset_per_frame_true/trained_model/*readout.pth')[0],
               map_location=torch.device('cpu')))
trained_net_true.eval()

# %%
is_train, layer = [], []  # whether rep is generated from training set
# contains reps generated without resetting per frame seq dataset
seq_r_act_f, seq_r_act_t = [], []
seq_r_out_f, seq_r_out_t = [], []
seq_e_out_f, seq_e_out_t = [], []


labels = []

df_seq_f, df_seq_t = pd.DataFrame(), pd.DataFrame()

dataloaders = [train_loader_spin, test_loader_spin]

##############################################################
# generate sequence representations
##############################################################

with torch.no_grad():
    for loader in range(len(dataloaders)):
        print(len(dataloaders[loader]))
        tr = 1 if loader == 0 else 0  # log whether rep is generated from train or test set
        for i, (_image, _label) in enumerate(dataloaders[loader]):
            # print(i)
            trained_net_true.forward(_image, config['infsteps'], istrain=False)
            trained_net_false(_image, config['infsteps'], istrain=False)
            if (i + 1) % 5 == 0:  # at the end of eqch sequence
                for l in range(len(trained_net_false.architecture)):
                    is_train.append(tr)
                    layer.append(l)
                    labels.append(int(_label.cpu().numpy()))
                    # reps from false net
                    seq_r_act_f.append(trained_net_false.states['r_activation'][l].detach().cpu().numpy())
                    seq_r_out_f.append(trained_net_false.states['r_output'][l].detach().cpu().numpy())

                    # reps from true net
                    seq_r_act_t.append(trained_net_true.states['r_activation'][l].detach().cpu().numpy())
                    seq_r_out_t.append(trained_net_true.states['r_output'][l].detach().cpu().numpy())

                    if l == (len(trained_net_false.architecture) - 1):
                        seq_e_out_f.append(np.zeros(trained_net_false.layers[l].layer_size))
                        seq_e_out_t.append(np.zeros(trained_net_true.layers[l].layer_size))
                    else:
                        seq_e_out_f.append(trained_net_false.states['error'][l].detach().cpu().numpy())
                        seq_e_out_t.append(trained_net_true.states['error'][l].detach().cpu().numpy())

                print('%i seqs done' % ((i + 1) / 5), len(is_train))
                trained_net_false.init_states()
                trained_net_true.init_states()

print(len(is_train))

# raise Exception('pause program here')
# %%
# store all generated values into dataframes
df_seq_f['is_train'] = is_train
df_seq_f['layer'] = layer
df_seq_f['r_out'] = seq_r_out_f
df_seq_f['r_act'] = seq_r_act_f
df_seq_f['e_out'] = seq_e_out_f
df_seq_f['labels'] = labels

df_seq_t['is_train'] = is_train
df_seq_t['layer'] = layer
df_seq_t['r_out'] = seq_r_out_t
df_seq_t['r_act'] = seq_r_act_t
df_seq_t['e_out'] = seq_e_out_t
df_seq_t['labels'] = labels

# %%
##############################################################
# generate frame representations
##############################################################
# contains reps generated resetting per frame from seq dataset
is_train_frame, layer_frame, labels_frame = [], [], []

frame_r_act_f, frame_r_act_t = [], []
frame_r_out_f, frame_r_out_t = [], []
frame_e_out_f, frame_e_out_t = [], []

with torch.no_grad():
    for loader in range(len(dataloaders)):
        print(len(dataloaders[loader]))
        tr = 1 if loader == 0 else 0  # log whether rep is generated from train or test set
        for i, (_image, _label) in enumerate(dataloaders[loader]):
            # print(i)
            trained_net_true.forward(_image, config['infsteps'], istrain=False)
            trained_net_false(_image, config['infsteps'], istrain=False)
            # for every frame
            for l in range(len(trained_net_false.architecture)):
                is_train_frame.append(tr)
                layer_frame.append(l)
                labels_frame.append(int(_label.cpu().numpy()))
                # reps from false net
                frame_r_act_f.append(trained_net_false.states['r_activation'][l].detach().cpu().numpy())
                frame_r_out_f.append(trained_net_false.states['r_output'][l].detach().cpu().numpy())

                # reps from true net
                frame_r_act_t.append(trained_net_true.states['r_activation'][l].detach().cpu().numpy())
                frame_r_out_t.append(trained_net_true.states['r_output'][l].detach().cpu().numpy())

                if l == (len(trained_net_false.architecture) - 1):
                    frame_e_out_f.append(np.zeros(trained_net_false.layers[l].layer_size))
                    frame_e_out_t.append(np.zeros(trained_net_true.layers[l].layer_size))
                else:
                    frame_e_out_f.append(trained_net_false.states['error'][l].detach().cpu().numpy())
                    frame_e_out_t.append(trained_net_true.states['error'][l].detach().cpu().numpy())

            if i % 20 == 0:
                print('%i frames done' % i)
            trained_net_false.init_states()
            trained_net_true.init_states()

# %%
df_frame_f, df_frame_t = pd.DataFrame(), pd.DataFrame()

# store all generated values into dataframes
df_frame_f['is_train'] = is_train_frame
df_frame_f['layer'] = layer_frame
df_frame_f['r_out'] = frame_r_out_f
df_frame_f['r_act'] = frame_r_act_f
df_frame_f['e_out'] = frame_e_out_f
df_frame_f['labels'] = labels_frame

df_frame_t['is_train'] = is_train_frame
df_frame_t['layer'] = layer_frame
df_frame_t['r_out'] = frame_r_out_t
df_frame_t['r_act'] = frame_r_act_t
df_frame_t['e_out'] = frame_e_out_t
df_frame_t['labels'] = labels_frame

# %%
##############################################################
# get subset of still reps fro frame reps
##############################################################


# %%
##################
# classification acc
##################

# seq to seq
seq_to_seq_acc = []
reset_per_frame = []
by_layer = []


def get_layer_acc(dataframe, _layer):
    _, acc = linear_regression(
        np.vstack(dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['r_out'].to_numpy()),
        dataframe[dataframe['is_train'] == 1][dataframe['layer'] == _layer]['labels'].to_numpy(),
        np.vstack(dataframe[dataframe['is_train'] == 0][dataframe['layer'] == _layer][
                      'r_out'].to_numpy()),
        dataframe[dataframe['is_train'] == 0][dataframe['layer'] == _layer]['labels'].to_numpy()
        )
    return acc


# %%
for i in range(3):
    # append acc from reset per frame false
    by_layer.append(i)
    reset_per_frame.append(0)
    seq_to_seq_acc.append(get_layer_acc(df_seq_f, i))

    # append acc from reset per frame true
    by_layer.append(i)
    reset_per_frame.append(1)
    seq_to_seq_acc.append(get_layer_acc(df_seq_t, i))

df_seq_to_seq_acc = pd.DataFrame()
df_seq_to_seq_acc['acc'] = seq_to_seq_acc
df_seq_to_seq_acc['layer'] = by_layer
df_seq_to_seq_acc['reset_per_frame'] = reset_per_frame

# %%
fig, ax = plt.subplots()
ax = sns.barplot(data=df_seq_to_seq_acc, x='layer', y='acc', hue='reset_per_frame')
for container in ax.containers:
    ax.bar_label(container)
plt.title('seq to seq generalisation: acc by layer')
# plt.show()
plt.savefig(os.path.join(file_path, 'seq to seq generalisation: acc by layer.png'))

 # %%
# frame to frame
frame_to_frame_acc = []
reset_per_frame = []
by_layer = []

for i in range(3):
    # append acc from reset per frame false
    by_layer.append(i)
    reset_per_frame.append(0)
    frame_to_frame_acc.append(get_layer_acc(df_frame_f, i))

    # append acc from reset per frame true
    by_layer.append(i)
    reset_per_frame.append(1)
    frame_to_frame_acc.append(get_layer_acc(df_frame_t, i))

df_frame_to_frame_acc = pd.DataFrame()
df_frame_to_frame_acc['acc'] = frame_to_frame_acc
df_frame_to_frame_acc['layer'] = by_layer
df_frame_to_frame_acc['reset_per_frame'] = reset_per_frame

# %%
fig, ax = plt.subplots()
ax = sns.barplot(data=df_frame_to_frame_acc, x='layer', y='acc', hue='reset_per_frame')
for container in ax.containers:
    ax.bar_label(container)
plt.title('frame to frame generalisation: acc by layer')
# plt.show()
plt.savefig(os.path.join(file_path, 'frame to frame generalisation: acc by layer.png'))

# %%
# still to still
df_still_f = pd.DataFrame()
df_still_t = pd.DataFrame()


# %%
# testing on still img
# load images
train_set = torch.load(os.path.join(file_path, 'fashionMNISTtrain_image.pt'))
test_set = torch.load(os.path.join(file_path, 'fashionMNISTtest_image.pt'))
# train_set = torch.load('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/fashionMNISTtrain_image.pt')
# test_set = torch.load('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/fashionMNISTtest_image.pt')

# %%
train_indices = train_set.indices
test_indices = test_set.indices

# %%
padding = 5
train_images = train_set.dataset.data[train_indices]
train_images = nn.functional.pad(train_images, (padding, padding, padding, padding))
# train_images = torch.flatten(train_images, start_dim=1).numpy()
train_labels = train_set.dataset.targets[train_indices]

# %%
test_images = test_set.dataset.data[test_indices]
test_images = nn.functional.pad(test_images, (padding, padding, padding, padding))
# test_images = torch.flatten(test_images, start_dim=1).numpy()
test_labels = test_set.dataset.targets[test_indices]

# %%
train_still = data.TensorDataset(train_images, train_labels)
train_still_loader = data.DataLoader(train_still, shuffle=True)
test_still = data.TensorDataset(test_images, test_labels)
test_still_loader = data.DataLoader(test_still, shuffle=True)

# %%
# generate rep for still images
is_train_still, layer_still, labels_still = [], [], []
# contains reps generated from still img dataset, purely structural
still_r_act_f, still_r_act_t = [], []
still_r_out_f, still_r_out_t = [], []
still_e_out_f, still_e_out_t = [], []

dataloaders = [train_still_loader, test_still_loader]
for loader in range(len(dataloaders)):
    print(len(dataloaders[loader]))
    tr = 1 if loader == 0 else 0
    for i, (_image, _label) in enumerate(dataloaders[loader]):
        trained_net_true.forward(_image, config['infsteps'], istrain=False)
        trained_net_false(_image, config['infsteps'], istrain=False)
        # for every frame
        for l in range(len(trained_net_false.architecture)):
            is_train_still.append(tr)
            layer_still.append(l)
            labels_still.append(int(_label.cpu().numpy()))
            # reps from false net
            still_r_act_f.append(trained_net_false.states['r_activation'][l].detach().cpu().numpy())
            still_r_out_f.append(trained_net_false.states['r_output'][l].detach().cpu().numpy())

            # reps from true net
            still_r_act_t.append(trained_net_true.states['r_activation'][l].detach().cpu().numpy())
            still_r_out_t.append(trained_net_true.states['r_output'][l].detach().cpu().numpy())

            if l == (len(trained_net_false.architecture) - 1):
                still_e_out_f.append(np.zeros(trained_net_false.layers[l].layer_size))
                still_e_out_t.append(np.zeros(trained_net_true.layers[l].layer_size))
            else:
                still_e_out_f.append(trained_net_false.states['error'][l].detach().cpu().numpy())
                still_e_out_t.append(trained_net_true.states['error'][l].detach().cpu().numpy())

        if i % 20 == 0:
            print('%i img done' % i)
        trained_net_false.init_states()
        trained_net_true.init_states()


# %%
df_still_f['is_train'] = is_train_still
df_still_f['layer'] = layer_still
df_still_f['r_out'] = still_r_out_f
df_still_f['r_act'] = still_r_act_f
df_still_f['e_out'] = still_e_out_f
df_still_f['labels'] = labels_still

df_still_t['is_train'] = is_train_still
df_still_t['layer'] = layer_still
df_still_t['r_out'] = still_r_out_t
df_still_t['r_act'] = still_r_act_t
df_still_t['e_out'] = still_e_out_t
df_still_t['labels'] = labels_still

# %%
still_to_still_acc = []
reset_per_frame = []
by_layer = []

for i in range(3):
    # append acc from reset per frame false
    by_layer.append(i)
    reset_per_frame.append(0)
    still_to_still_acc.append(get_layer_acc(df_still_f, i))

    # append acc from reset per frame true
    by_layer.append(i)
    reset_per_frame.append(1)
    still_to_still_acc.append(get_layer_acc(df_still_t, i))

df_still_to_still_acc = pd.DataFrame()
df_still_to_still_acc['acc'] = still_to_still_acc
df_still_to_still_acc['layer'] = by_layer
df_still_to_still_acc['reset_per_frame'] = reset_per_frame

# %%
fig, ax = plt.subplots()
ax = sns.barplot(data=df_still_to_still_acc, x='layer', y='acc', hue='reset_per_frame')
for container in ax.containers:
    ax.bar_label(container)
plt.title('still to still generalisation: acc by layer')
# plt.show()
plt.savefig(os.path.join(file_path, 'still to still generalisation: acc by layer.png'))


# %%
# tSNE clustering
# print('tSNE clustering')
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
# tsne_results = tsne.fit_transform(train_x)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
#
# # %%
# # visualisation
# df = pandas.DataFrame()
# df['tsne-one'] = tsne_results[:, 0]
# df['tsne-two'] = tsne_results[:, 1]
# df['y'] = train_y
# fig, ax1 = plt.subplots(figsize=(10, 8))
# sns.scatterplot(
#     x="tsne-one", y="tsne-two",
#     hue="y",
#     palette=sns.color_palette("bright", 10),
#     data=df,
#     legend="full",
#     alpha=0.3,
#     ax=ax1
# )
#
# plt.show()
# fig.savefig(os.path.join(file_path, 'tSNE_clustering_rep'))

# %%
# # sort representations by class first before passing to rdm function
# train_indices = np.argsort(train_y)
# train_x = train_x[train_indices]
#
# test_indices = np.argsort(test_y)
# test_x = test_x[test_indices]
#
# rdm_train = rdm_w_rep(train_x, 'cosine', istrain=True)
# rdm_train.savefig(os.path.join(file_path, 'rdm_train.png'))
# rdm_test = rdm_w_rep(test_x, 'cosine', istrain=False)
# rdm_test.savefig(os.path.join(file_path, 'rdm_test.png'))

# %%
