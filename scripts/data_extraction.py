import os

import numpy as np
import torch
import torchvision
import yaml
from sklearn.model_selection import train_test_split
# folder to load config file
from torch.utils.data import DataLoader
from torch.utils.data import Subset

CONFIG_PATH = "../scripts/"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

data_dir = config['dataset_dir'] # where to save train and test data
dataset = config['dataset_type']
train_size = config['train_size'] # training dataset size
test_size = config['test_size'] # test dataset size
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
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=n_workers, pin_memory=True, batch_size=batchSize)
test_loader = DataLoader(test_dataset, shuffle=True, num_workers=n_workers, pin_memory=True, batch_size=batchSize)

torch.save(train_loader, os.path.join(data_dir, str(dataset) + 'train_loader.pth'))
torch.save(test_loader, os.path.join(data_dir, str(dataset) + 'test_loader.pth'))