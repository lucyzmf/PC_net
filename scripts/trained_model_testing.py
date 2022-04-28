'''
This script evalute trained models
'''
import os.path

from torch import nn
from torch.utils import data

from evaluation import *
# load config
from scripts.fc_net import FcDHPC

# print("Number of processors: ", mp.cpu_count())
# TODO change representation generation function since on trained dataset it should be representations generated from sequences

CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
    file_path = '../results/morph_test_9'
    dataDir = '/40_10perclass/'
else:
    dev = "cpu"
    print("Cuda not available")
    file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_9/')
    dataDir = '/40_10perclass/'
device = torch.device(dev)

dtype = torch.float  # Set standard datatype

torch.manual_seed(0)

#  pytorch logistic regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# raise Exception('stop')

# %%
#  load the training set used during training
train_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtrain_set_spin.pt'))
trainLoaderSpin = data.DataLoader(train_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                  pin_memory=config['pin_mem'], shuffle=False)
test_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtest_set_spin.pt'))
testLoaderSpin = data.DataLoader(test_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                 pin_memory=config['pin_mem'], shuffle=False)


# %%
def turn_tensordataset_to_np(seq_list, labels_list, dataloader):
    for i, (_frame, _label) in enumerate(dataloader):
        seq_list.append(torch.flatten(_frame).data.numpy())
        labels_list.append(_label.data)

    seq_list = np.vstack(seq_list)
    labels_list = torch.concat(labels_list).numpy()

    return seq_list, labels_list

# %%
train_seq_spin, train_labels_spin = [], []
test_seq_spin, test_labels_spin = [], []

train_seq_spin, train_labels_spin = turn_tensordataset_to_np(train_seq_spin, train_labels_spin, trainLoaderSpin)
test_seq_spin, test_labels_spin = turn_tensordataset_to_np(test_seq_spin, test_labels_spin, testLoaderSpin)

# %%

dataWidth = 28 + 2 * config['padding_size']  # width of mnist + padding
numClass = 10  # number of classes in mnist

# %%
# load trained model
# Hyperparameters used during training
inference_steps_T = 250  # num infsteps per image during testing
inference_steps_F = 200

infrates_T = [.07, .07, .05, .05]  # inf rates each layer
infrates_F = [.05, .05, .03, .03]

lr = config['learning_rate']  # lr for weight updates
arch = config['network_size']  # size of each layer
per_seq_repeat = config['per_seq_repeat']  # num of repeats per image/sequence
arch_type = config['architecture']

#  network instantiation
per_im_repeat = 1

# trained_net = DHPC(network_architecture, inf_rates, lr=lr, act_func=sigmoid, device=device, dtype=dtype)
if config['architecture'] == 'FcDHPC':
    trained_net_true = FcDHPC(config['network_size'], infrates_T, lr=config['learning_rate'],
                              act_func=config['act_func'],
                              device=device, dtype=dtype)
    trained_net_false = FcDHPC(config['network_size'], infrates_F, lr=config['learning_rate'],
                               act_func=config['act_func'],
                               device=device, dtype=dtype)
# elif config['architecture'] == 'RfDHPC_cm':
#     trained_net = RfDHPC_cm(config['network_size'], config['rf_sizes'], config['infrates'],
#                             lr=config['learning_rate'],
#                             act_func=config['act_func'],
#                             device=device, dtype=dtype)


# %%
# create dataframe of representations using train and test images
# TODO classification acc on r from all layers

# raise Exception('stop')

print('generate representations')
# create data frame that has columns: is_train, layers, r_act, r_out, e_out, reset at the end of each sequence
trained_net_false.load_state_dict(
    torch.load(os.path.join(file_path, 'resetFalse_seqtrainTrue2022-04-25 16:44:35.188886/trained_model/spin'
                                       '[1444, 2500, 800, 100][0.05, 0.05, 0.03, '
                                       '0.03]Truel2_0.018.0end_trainingreadout.pth'),
               map_location=torch.device('cpu')))
trained_net_false.eval()

trained_net_true.load_state_dict(
    torch.load(os.path.join(file_path, 'resetTrue_seqtrainTrue2022-04-26 13:07:27.507141/trained_model/spin'
                                       '[1444, 2500, 800, 100][0.07, 0.07, 0.05, '
                                       '0.05]Truel2_0.0110.0end_trainingreadout.pth'),
        map_location=torch.device('cpu')))
trained_net_true.eval()

# %%
# # distribution of trained weights
#
# fig, axs = plt.subplots(1, len(arch) - 1, figsize=(12, 4))
# for i in range(len(arch) - 1):
#     axs[i].hist(trained_net_false.layers[i].weights)
#     axs[i].set_title('layer' + str(i))
#
# plt.show()

# raise Exception('stop')
# %%
is_train, layer = [], []  # whether rep is generated from training set
# contains reps generated without resetting per frame seq dataset
seq_r_act_f, seq_r_act_t = [], []
seq_r_out_f, seq_r_out_t = [], []
seq_e_out_f, seq_e_out_t = [], []

labels = []

df_seq_f, df_seq_t = pd.DataFrame(), pd.DataFrame()

dataloaders = [trainLoaderSpin, testLoaderSpin]

##############################################################
# generate sequence representations
##############################################################

with torch.no_grad():
    trained_net_true.init_states()
    trained_net_false.init_states()
    for loader in range(len(dataloaders)):
        print(len(dataloaders[loader]))
        tr = 1 if loader == 0 else 0  # log whether rep is generated from train or test set
        for i, (_image, _label) in enumerate(dataloaders[loader]):
            # print(i)
            trained_net_true(_image, inference_steps_T, istrain=False)
            trained_net_false(_image, inference_steps_F, istrain=False)
            if (i + 1) % config['frame_per_sequence'] == 0:  # at the end of eqch sequence
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

                print('%i seqs done' % ((i + 1) / 9), len(is_train))
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
    trained_net_true.init_states()
    trained_net_false.init_states()
    for loader in range(len(dataloaders)):
        print(len(dataloaders[loader]))
        tr = 1 if loader == 0 else 0  # log whether rep is generated from train or test set
        for i, (_image, _label) in enumerate(dataloaders[loader]):
            # print(i)
            trained_net_true.forward(_image, inference_steps_T, istrain=False)
            trained_net_false(_image, inference_steps_F, istrain=False)
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
train_set = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtrain_image.pt'))
test_set = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtest_image.pt'))
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
with torch.no_grad():
    for loader in range(len(dataloaders)):
        trained_net_true.init_states()
        trained_net_false.init_states()
        print(len(dataloaders[loader]))
        tr = 1 if loader == 0 else 0
        for i, (_image, _label) in enumerate(dataloaders[loader]):
            trained_net_true.forward(_image, inference_steps_T, istrain=False)
            trained_net_false(_image, inference_steps_F, istrain=False)
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
cum_acc = within_sample_classification_stratified(np.vstack(df_seq_t[df_seq_t['is_train']==0][df_seq_t['layer']==2]['r_out'].to_numpy()),
                                                  df_seq_t[df_seq_t['is_train']==0][df_seq_t['layer']==2]['labels'].to_numpy())
print(cum_acc)

# %%
acc1, acc2 = linear_regression(np.vstack(df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['r_out'].to_numpy()),
                           df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['labels'].to_numpy(),
                           np.vstack(df_seq_f[df_seq_f['is_train'] == 0][df_seq_f['layer'] == 2]['r_out'].to_numpy()),
                           df_seq_f[df_seq_f['is_train'] == 0][df_seq_f['layer'] == 2]['labels'].to_numpy()
                           )
print(acc1, acc2)
# %%
acc1, acc2 = linear_regression(np.vstack(df_frame_t[df_frame_t['is_train']==1][df_frame_t['layer']==2]['r_out'].to_numpy()),
                           df_frame_t[df_frame_t['is_train']==1][df_frame_t['layer']==2]['labels'].to_numpy(),
                           np.vstack(df_frame_t[df_frame_t['is_train'] == 0][df_frame_t['layer'] == 2]['r_out'].to_numpy()),
                           df_frame_t[df_frame_t['is_train'] == 0][df_frame_t['layer'] == 2]['labels'].to_numpy()
                           )
print(acc1, acc2)

# %%
# tSNE clustering


plot_tsne(df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['r_out'].to_numpy(), df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['labels'].to_numpy())

# %%
highest_level_rep_f = np.vstack(df_seq_f[df_seq_f['layer']==2]['r_out'].to_numpy())
labels_f = df_seq_f[df_seq_f['layer']==2]['labels'].to_numpy()
highest_level_rep_t = np.vstack(df_frame_t[df_frame_t['layer']==2]['r_out'].to_numpy())
labels_t = df_frame_t[df_frame_t['layer']==2]['labels'].to_numpy()

# %%
cum_acc = within_sample_classification_stratified(highest_level_rep_f, labels_f)
print(cum_acc)

cum_acc = within_sample_classification_stratified(highest_level_rep_t, labels_t)
print(cum_acc)

# %%
# try normalising activity and test decoding
def normalise(rep_array):
    for i in range(len(rep_array)):
        mean = np.mean(rep_array[i])
        var = np.var(rep_array[i])
        rep_array[i] = (rep_array[i]-mean) / np.sqrt(var)
    return rep_array

# %%
acc1, acc2 = linear_regression(normalise(np.vstack(df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['r_out'].to_numpy())),
                           df_seq_f[df_seq_f['is_train']==1][df_seq_f['layer']==2]['labels'].to_numpy(),
                           normalise(np.vstack(df_seq_f[df_seq_f['is_train'] == 0][df_seq_f['layer'] == 2]['r_out'].to_numpy())),
                           df_seq_f[df_seq_f['is_train'] == 0][df_seq_f['layer'] == 2]['labels'].to_numpy()
                           )
print(acc1, acc2)
# %%
acc1, acc2 = linear_regression(np.nan_to_num(normalise(np.vstack(df_frame_t[df_frame_t['is_train']==1][df_frame_t['layer']==2]['r_out'].to_numpy())), nan=0),
                           df_frame_t[df_frame_t['is_train']==1][df_frame_t['layer']==2]['labels'].to_numpy(),
                           np.nan_to_num(normalise(np.vstack(df_frame_t[df_frame_t['is_train'] == 0][df_frame_t['layer'] == 2]['r_out'].to_numpy())), nan=0),
                           df_frame_t[df_frame_t['is_train'] == 0][df_frame_t['layer'] == 2]['labels'].to_numpy()
                           )
print(acc1, acc2)

# %%
# # sort representations by class first before passing to rdm function
idx_f = np.argsort(labels_f)
idx_t = np.argsort(labels_t)

rdm_f = rdm_w_rep(highest_level_rep_f[idx_f], 'euclidean', istrain=True)
# rdm_train.savefig(os.path.join(file_path, 'rdm_train.png'))
rdm_t = rdm_w_rep(highest_level_rep_t[idx_t], 'euclidean', istrain=False)
# rdm_test.savefig(os.path.join(file_path, 'rdm_test.png'))

