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
#  load the training and testing data for spinning
train_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtrain_set_spin.pt'))
trainLoaderSpin = data.DataLoader(train_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                  pin_memory=config['pin_mem'], shuffle=False)
test_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtest_set_spin.pt'))
testLoaderSpin = data.DataLoader(test_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                 pin_memory=config['pin_mem'], shuffle=False)

# %%
# load still img used
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
# TODO need to compare 9 different frames vs 9 of the same frames


# %%
def get_data(tensordataset):
    rep, label = tuple(zip(*tensordataset))
    rep = np.array(torch.stack(rep))
    label = np.array(torch.stack(label))

    return rep, label


# %%
# get data
train_seq_spin, train_labels_spin = get_data(train_set_spin)
test_seq_spin, test_labels_spin = get_data(test_set_spin)
train_still, train_still_labels = get_data(train_still)
test_still, test_still_labels = get_data(test_still)

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
    train_net_still = FcDHPC(config['network_size'], infrates_T, lr=config['learning_rate'],
                             act_func=config['act_func'],
                             device=device, dtype=dtype)


# elif config['architecture'] == 'RfDHPC_cm':
#     trained_net = RfDHPC_cm(config['network_size'], config['rf_sizes'], config['infrates'],
#                             lr=config['learning_rate'],
#                             act_func=config['act_func'],
#                             device=device, dtype=dtype)

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

raise Exception('stop')
# %%
##############################################################
# generate representations
##############################################################

dataloaders = [trainLoaderSpin, testLoaderSpin]
# sequence representations
df_seq_f = generate_reps(trained_net_false, dataloaders, inference_steps_F, resetPerFrame=False)

# %%
# frame representations
df_frame_t = generate_reps(trained_net_true, dataloaders, inference_steps_T, resetPerFrame=True)

# %%
# still img representations
dataloaders_still = [train_still_loader, test_still_loader]
df_still_f = generate_reps(trained_net_false, dataloaders_still, inference_steps_F, resetPerFrame=True)
df_still_t = generate_reps(trained_net_true, dataloaders_still, inference_steps_T, resetPerFrame=True)

# %%
# test the importance of spatial statistics of morphing stimuli: exposed reset_T net to longer inference time
# that is equivalent to sequence length
df_f_no_spatial_diff = generate_reps(trained_net_false, dataloaders_still, inference_steps_T*9, resetPerFrame=True)
# %%
# df_frame_f = generate_reps(trained_net_false, dataloaders, inference_steps_F*9, resetPerFrame=True)
# %%
# baselines
_, seq_input_gen_baseline = linear_regression(torch.flatten(torch.tensor(train_seq_spin), start_dim=1).numpy(),
                                              train_labels_spin,
                                              torch.flatten(torch.tensor(test_seq_spin), start_dim=1).numpy(),
                                              test_labels_spin)
# %%
_, still_input_gen_baseline = linear_regression(torch.flatten(torch.tensor(train_still), start_dim=1).numpy(),
                                                train_still_labels,
                                                torch.flatten(torch.tensor(test_still), start_dim=1).numpy(),
                                                test_still_labels)


# %%
def sigmoid(inputs):
    inputs = inputs - 3
    m = nn.Sigmoid()
    return m(inputs)


# %%
linear_regression(sigmoid(torch.flatten(torch.tensor(train_seq_spin), start_dim=1)).numpy(), train_labels_spin,
                  sigmoid(torch.flatten(torch.tensor(test_seq_spin), start_dim=1)).numpy(), test_labels_spin)

# %%
##################
# classification acc
##################

# compare sets of reps
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


# %%
# plot clustering by layer
df_clustering_reset = generate_acc_df([df_seq_f, df_frame_t], [0, 1], 'reset_per_frame', False)

# %%
# plot cluster acc
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.barplot(data=df_clustering_reset, x='layer', y='acc', hue='reset_per_frame')
for container in ax.containers:
    ax.bar_label(container)
plt.legend(title='Reset per frame', loc='lower right')
plt.title('clustering by layer')
plt.show()
# %%
df_seqvsframe_reset = generate_acc_df([df_seq_f, df_frame_t], [0, 1], 'reset_per_frame', True)

# %%
fig, ax = plt.subplots()
ax = sns.catplot(data=df_seqvsframe_reset, x='layer', y='acc', hue='reset_per_frame', col='accIsTest', kind='bar')
# for container in ax.containers:
#     ax.bar_label(container)
plt.title('generalisation: acc by layer')
plt.show()
# plt.savefig(os.path.join(file_path, 'generalisation: acc by layer.png'))


# %%
# still to still
still_to_still_acc = []
reset_per_frame = []
by_layer = []

for i in range(4):
    # append acc from reset per frame false
    by_layer.append(i)
    reset_per_frame.append(0)
    still_to_still_acc.append(get_layer_gen_acc(df_still_f, i))

    # append acc from reset per frame true
    by_layer.append(i)
    reset_per_frame.append(1)
    still_to_still_acc.append(get_layer_gen_acc(df_still_t, i))

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
plt.show()
# plt.savefig(os.path.join(file_path, 'still to still generalisation: acc by layer.png'))

# %%
# clustering by layer and network
cluster_acc = []
reset_per_frame = []
by_layer = []
for i in range(4):
    # append acc from reset per frame false
    by_layer.append(i)
    reset_per_frame.append(0)
    cluster_acc.append(get_layer_clustering(df_seq_f, i))

    # append acc from reset per frame true
    by_layer.append(i)
    reset_per_frame.append(1)
    cluster_acc.append(get_layer_clustering(df_frame_t, i))

df_cluster_acc = pd.DataFrame()
df_cluster_acc['acc'] = cluster_acc
df_cluster_acc['layer'] = by_layer
df_cluster_acc['reset_per_frame'] = reset_per_frame



# %%
# tSNE clustering
plot_tsne(df_seq_f[df_seq_f['is_train'] == 1][df_seq_f['layer'] == 2]['r_out'].to_numpy(),
          df_seq_f[df_seq_f['is_train'] == 1][df_seq_f['layer'] == 2]['labels'].to_numpy())

# %%
high_layer_pca = pca_vis_3d(np.vstack(df_seq_f[df_seq_f['layer'] == 3]['r_out'].to_numpy()),
                            df_seq_f[df_seq_f['layer'] == 3]['labels'].to_numpy())
# high_layer_pca.title('layer 3 cluster visualisation')
high_layer_pca.show()


# %%
# try normalising activity and test decoding
def normalise(rep_array):
    for i in range(len(rep_array)):
        mean = np.mean(rep_array[i])
        var = np.var(rep_array[i])
        rep_array[i] = (rep_array[i] - mean) / np.sqrt(var)
    return rep_array

# %%
rdm_w_rep_title(np.vstack(df_frame_t[df_frame_t['layer']==2]['r_out'].to_numpy()), df_frame_t[df_frame_t['layer']==2]['labels'].to_numpy(), 'cosine', 'layer2 reset true')
plt.show()