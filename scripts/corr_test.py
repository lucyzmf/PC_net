from glob import glob

import torch.profiler
from torch import nn
from torch.utils import data

from evaluation import *

filePath = '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_10/correlation/'
raise Exception('p')
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
                             num_workers=config['num_workers'], pin_memory=config['pin_mem'], shuffle=False)

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
# load state dict of trained networks
trainednet = FcDHPC(config['network_size'], config['infrates'], lr=config['learning_rate'],
                       act_func=config['act_func'], device=device, dtype=dtype)


def get_gen_acc(still_rep_df, test_reps, labels):
    _, gen_accs = linear_regression(
        np.vstack(still_rep_df[still_rep_df['is_train'] == 1][still_rep_df['layer'] == 3]['r_out'].to_numpy()),
        still_rep_df[still_rep_df['is_train'] == 1][still_rep_df['layer'] == 3]['labels'].to_numpy(), test_reps,
        labels)

    return gen_accs


def load_and_generate(trained_net, _dataset_size, _test_loader):
    gen_acc_list = []
    for i in range(5):
        trained_net.load_state_dict(torch.load(glob(os.path.join(filePath, str(_dataset_size) + '_datasetsize*/trained_model/*trainingreadout.pth'))[i],
                                               map_location=torch.device('cpu')))
        still_rep = pd.read_pickle(glob(os.path.join(filePath, str(_dataset_size) + '_datasetsize*/trained_model/still_rep.pkl'))[i])
        test_reps = generate_test_reps(trained_net, _test_loader)
        acc = get_gen_acc(still_rep, test_reps, test_labels)
        gen_acc_list.append(acc)
        print(gen_acc_list)

    return gen_acc_list

# %%
gen_acc_20_ls = load_and_generate(trainednet, 20, test_loader)
gen_acc_40_ls = load_and_generate(trainednet, 40, test_loader)
gen_acc_100_ls = load_and_generate(trainednet, 100, test_loader)
gen_acc_200_ls = load_and_generate(trainednet, 200, test_loader)
gen_acc_500_ls = load_and_generate(trainednet, 500, test_loader)

# %%
dataset_size = np.concatenate(
    (
        np.full(5, 20),
        np.full(5, 40),
        np.full(5, 100),
        np.full(5, 200),
        np.full(5, 500)
    )
)
gen_acc = np.concatenate((
    gen_acc_20_ls,
    gen_acc_40_ls,
    gen_acc_100_ls,
    gen_acc_200_ls,
    gen_acc_500_ls
))

# %%
df_gen = pd.DataFrame()
df_gen['samples per class'] = dataset_size
df_gen['acc'] = gen_acc

fig, ax = plt.subplots()
sns.despine()
sns.barplot(data=df_gen, x='samples per class', y='acc')
plt.show()
