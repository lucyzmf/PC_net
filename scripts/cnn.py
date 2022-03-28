import os

import torch
import torch.nn as nn
# %%
import wandb
# Device configuration
import yaml
from torch.utils import data

wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')

wandb.init(project="DHPC_morph_cnn", entity="lucyzmf")  # , mode='disabled')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

wbconfig = wandb.config

# Hyper parameters
wbconfig.num_epochs = 100
wbconfig.num_classes = 10
batch_size = 100
wbconfig.learning_rate = 0.001

CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

# seed the CPUs and GPUs
torch.manual_seed(0)

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
    # cuda.manual_seed_all(0)

else:
    dev = "cpu"
    print("Cuda not available")
device = torch.device(dev)

# dtype = torch.float  # Set standard datatype

# git config values
dataset = config['dataset_type']
inference_steps = config['infsteps']  # num infsteps per image
epochs = config['epochs']  # total training epochs
infrates = config['infrates']  # inf rates each layer
lr = config['learning_rate']  # lr for weight updates
arch = config['network_size']  # size of each layer
per_seq_repeat = config['per_seq_repeat']  # num of repeats per image/sequence
arch_type = config['architecture']
morph_type = config['morph_type']
frame_per_seq = config['frame_per_sequence']
padding = config['padding_size']

# load data
train_loader = torch.load(
    os.path.join(config['dataset_dir'], str(dataset) + 'train_loader_' + str(morph_type) + '.pth'))
test_loader = torch.load(
    os.path.join(config['dataset_dir'], str(dataset) + 'test_loader_' + str(morph_type) + '.pth'))

# load test still images
test_set = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_image.pt'))
test_indices = test_set.indices
test_img_still = test_set.dataset.data[test_indices]
test_img_still = nn.functional.pad(test_img_still, (padding, padding, padding, padding))
test_labels_still = test_set.dataset.targets[test_indices]
still_img_dataset = data.TensorDataset(test_img_still, test_labels_still)
still_img_loader = data.DataLoader(still_img_dataset, batch_size=config['batch_size'],
                                   num_workers=config['num_workers'], pin_memory=config['pin_mem'])

# %%

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(2592, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(wbconfig.num_classes).to(device)
wandb.watch(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wbconfig.learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(wbconfig.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = torch.unsqueeze(images, dim=1).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        wandb.log({
            'epoch': epoch,
            'loss': loss
        })

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, wbconfig.num_epochs, i + 1, total_step, loss.item()))

    if (i + 1) % 10 == 0:
        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in still_img_loader:
                images = torch.unsqueeze(images.float(), dim=1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            wandb.log({
                'test_acc': 100 * correct / total
            })

            print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')