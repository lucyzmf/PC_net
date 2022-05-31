from glob import glob

import torch.cuda as cuda
import torch.profiler
from torch import nn
from torch.utils import data

from evaluation import *
from fc_net import FcDHPC

filePath = '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_10/correlation/'
# %%
# load saved still reps from training set
still_rep_20 = pd.read_pickle(glob(os.path.join(filePath, '20_datasetsize*/trained_model/still_rep.pkl'))[0])
still_rep_40 = pd.read_pickle(glob(os.path.join(filePath, '40_datasetsize*/trained_model/still_rep.pkl'))[0])
still_rep_100 = pd.read_pickle(glob(os.path.join(filePath, '100_datasetsize*/trained_model/still_rep.pkl'))[0])
still_rep_200 = pd.read_pickle(glob(os.path.join(filePath, '200_datasetsize*/trained_model/still_rep.pkl'))[0])
still_rep_500 = pd.read_pickle(glob(os.path.join(filePath, '500_datasetsize*/trained_model/still_rep.pkl'))[0])

# %%
# load test data (200)
test_img = torch.load(os.path.join(filePath, 'data200/fashionMNISTtest_image.pt'))

padding = 8


# from still to loader
def stilldataset_to_dataloader(still_tensordataset):
    idx = still_tensordataset.indices
    img = still_tensordataset.dataset.data[idx]
    img = nn.functional.pad(img, (padding, padding, padding, padding))
    label = still_tensordataset.dataset.targets[idx]
    dataset = data.TensorDataset(img, label)
    loader = data.DataLoader(dataset, batch_size=config['batch_size'],
                             num_workers=config['num_workers'], pin_memory=config['pin_mem'], shuffle=True)

    return loader


test_loader = stilldataset_to_dataloader(test_img)
# %%
# load trained network
# load config
CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
    cuda.manual_seed_all(0)

else:
    dev = "cpu"
    print("Cuda not available")
device = torch.device(dev)
dtype = torch.float  # Set standard datatype

config = load_config("config.yaml")

# %%
# load state dict of trained networks
trainednet_20 = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                       act_func=config['act_func'], device=device, dtype=dtype)
trainednet_20.load_state_dict(torch.load(glob(os.path.join(filePath, '20_datasetsize*/trained_model/*trainingreadout'
                                                                     '.pth'))[0],
                                         map_location=torch.device('cpu')))

# %%
trainednet_40 = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                       act_func=config['act_func'], device=device, dtype=dtype)
trainednet_40.load_state_dict(torch.load(glob(os.path.join(filePath, '40_datasetsize*/trained_model/*trainingreadout'
                                                                     '.pth'))[0],
                                         map_location=torch.device('cpu')))

# %%
trainednet_100 = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                        act_func=config['act_func'], device=device, dtype=dtype)
trainednet_100.load_state_dict(torch.load(glob(os.path.join(filePath, '100_datasetsize*/trained_model/*trainingreadout'
                                                                      '.pth'))[0],
                                          map_location=torch.device('cpu')))

# %%
trainednet_200 = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                        act_func=config['act_func'], device=device, dtype=dtype)
trainednet_200.load_state_dict(torch.load(glob(os.path.join(filePath, '200_datasetsize*/trained_model/*trainingreadout'
                                                                      '.pth'))[0],
                                          map_location=torch.device('cpu')))

# %%
trainednet_500 = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                        act_func=config['act_func'], device=device, dtype=dtype)
trainednet_500.load_state_dict(torch.load(glob(os.path.join(filePath, '500_datasetsize*/trained_model/*trainingreadout'
                                                                      '.pth'))[0],
                                          map_location=torch.device('cpu')))


# %%
# generate new test rep
def generate_test_reps(network, loader):
    test_rep = []
    for i, (_image, _label) in enumerate(loader):  # iterate through still image loader to generate reps
        network.init_states()
        network(_image, config['infsteps'], istrain=False)
        test_rep.append(network.states['r_output'][-1].detach().numpy())
        if i % 50 == 0:
            print('%i done' % i)

    test_rep = np.vstack(test_rep)
    return test_rep


# %%
def get_label_np(tensordataset):
    _, label = tuple(zip(*tensordataset))
    label = np.array(torch.concat(label))

    return label


# %%
# get all test labels from tensor to np
test_labels = get_label_np(test_loader)
# %%
# generate test reps for each network
test_reps_20 = generate_test_reps(trainednet_20, test_loader)
test_reps_40 = generate_test_reps(trainednet_40, test_loader)
test_reps_100 = generate_test_reps(trainednet_100, test_loader)
test_reps_200 = generate_test_reps(trainednet_200, test_loader)
test_reps_500 = generate_test_reps(trainednet_500, test_loader)

# %%
dataset_size = [20, 40, 100, 20, 500]
gen_acc = []


def get_gen_acc(still_rep_df, test_reps, labels):
    gen_accs = linear_regression(
        np.vstack(still_rep_df[still_rep_df['is_train'] == 1][still_rep_df['layer'] == 3]['r_out'].to_numpy()),
        still_rep_df[still_rep_df['is_train'] == 1][still_rep_df['layer'] == 3]['labels'].to_numpy(), test_reps,
        labels)

    return gen_accs


gen_acc_20 = get_gen_acc(still_rep_20, test_reps_20, test_labels)
gen_acc_40 = get_gen_acc(still_rep_40, test_reps_40, test_labels)
gen_acc_100 = get_gen_acc(still_rep_100, test_reps_100, test_labels)
gen_acc_200 = get_gen_acc(still_rep_200, test_reps_100, test_labels)
gen_acc_500 = get_gen_acc(still_rep_500, test_reps_500, test_labels)

# %%
df_gen = pd.DataFrame()
df_gen['samples per class'] = dataset_size
df_gen['acc'] = [gen_acc_20, gen_acc_40, gen_acc_100, gen_acc_200, gen_acc_500]

sns.barplot(data=df_gen, x='samples per class', y='acc')
plt.show()