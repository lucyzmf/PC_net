# %%
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

full_mnist = torchvision.datasets.MNIST(
    root="/Users/lucyzhang/Documents/research/PC_net/data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# %%

### genearte train test files for digit classification

indices = np.arange(len(full_mnist))
train_indices, test_indices = train_test_split(indices, train_size=100*10, test_size=10*10, stratify=full_mnist.targets)

# Warp into Subsets and DataLoaders
train_dataset = Subset(full_mnist, train_indices)
test_dataset = Subset(full_mnist, test_indices)

# %%

train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=True)

# %%
