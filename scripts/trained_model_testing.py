'''
This script evalute trained models
'''
import glob
import os

import pandas as pd
import yaml

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

dataWidth = 28 + 2*config['padding_size']  # width of mnist + padding
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
    trained_net = FcDHPC(config['network_size'], config['infrates'] * 2, lr=config['learning_rate'],
                         act_func=config['act_func'],
                         device=device, dtype=dtype)
elif config['architecture'] == 'RfDHPC_cm':
    trained_net = RfDHPC_cm(config['network_size'], config['rf_sizes'], config['infrates'] * 2, lr=config['learning_rate'],
                            act_func=config['act_func'],
                            device=device, dtype=dtype)
trained_net.load_state_dict(torch.load(glob.glob(file_path + '/reset_per_frame_false/trained_model/*readout.pth')[0], map_location=torch.device('cpu')))
trained_net.eval()

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
'''this is testing the network that is trained with seq '''

print('generate high level representations used for training and testing classifier')
# create data frame that has columns: is_train, layers, r_act, r_out, e_out, reset at the end of each sequence

is_train = []  # whether rep is generated from training set
layer = []
r_act = []
r_out = []
e_out = []
labels = []

with torch.no_grad():
    for i, (_image, _label) in enumerate(train_loader_spin):
        trained_net(_image, config['infsteps'], istrain=False)
        if i+1 % 5 == 0:
            print('1 seq done', len(is_train))
            for l in range(len(trained_net.architecture)):
                is_train.append(1)
                layer.append(l)
                r_act.append(trained_net.states['r_activation'][l].detach().cpu().numpy())
                r_out.append(trained_net.states['r_output'][l].cpu().numpy())
                e_out.append(trained_net.states['error'][l].cpu().numpy())
                labels.append(_label.cpu().numpy())
            trained_net.init_states()

print(len(is_train))

# %%
# generate representations from test spin loader
with torch.no_grad():
    for i, (_image, _label) in enumerate(test_loader_spin):
        trained_net(_image, config['infsteps'], istrain=False)
        if i+1 % 5 == 0:
            print('1 seq done', len(is_train))
            for l in range(len(trained_net.architecture)):
                is_train.append(0)
                layer.append(l)
                r_act.append(trained_net.states['r_activation'][l].detach().cpu().numpy())
                r_out.append(trained_net.states['r_output'][l].detach().cpu().numpy())
                e_out.append(trained_net.states['error'][l].detach().cpu().numpy())
            trained_net.init_states()

df_reps = pd.DataFrame()

df_reps['is_train'] = is_train
df_reps['layer'] = layer
df_reps['r_out'] = r_out
df_reps['r_act'] = r_act
df_reps['e_act'] = e_out

# df_reps.to_csv('../results/reps_df_reset_per_frame_false.csv')
# %%


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
