'''
This script evalute trained models
'''
import glob
import os
import time

import pandas
import seaborn as sns
import yaml
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

from evaluation import *
from fc_net import FcDHPC
from rf_net_cm import RfDHPC_cm

# load config
CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/2022-03-17 15:58:40.930652')

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
else:
    dev = "cpu"
    print("Cuda not available")
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
train_loader = torch.load(os.path.join(file_path, 'train_loader.pth'))
test_loader = torch.load(os.path.join(file_path, 'test_loader.pth'))

# %%

dataWidth = 28  # width of mnist
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
trained_net.load_state_dict(torch.load(glob.glob(file_path + '/*/*readout.pth')[0], map_location=torch.device('cpu')))
trained_net.eval()

# %%
# distribution of trained weights

fig, axs = plt.subplots(1, len(arch) - 1, figsize=(12, 4))
for i in range(len(arch) - 1):
    axs[i].hist(trained_net.layers[i].weights)
    axs[i].set_title('layer' + str(i))

plt.show()
fig.savefig(os.path.join(file_path, 'weight_dist'))

# %%
# code from online
classifier = LogisticRegression(arch[-1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# generate representations using train and test images

print('generate high level representations used for training and testing classifier')
cat_mem = torch.zeros(arch[-1]).to(device)

train_x = []  # contains last layer representations learned from training data
train_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(train_loader):
    train_x.append(high_level_rep(trained_net, torch.flatten(_image), 500))
    train_y.append(_label)
train_x, train_y = torch.stack(train_x), torch.cat(train_y)
dataset = data.TensorDataset(train_x, train_y)
train_loader_rep = DataLoader(dataset, shuffle=True)

test_x = []  # contains last layer representations learned from testing data
test_y = []  # contains labels in training data
for i, (_image, _label) in enumerate(test_loader):
    test_x.append(high_level_rep(trained_net, torch.flatten(_image), 500))
    test_y.append(_label)
test_x, test_y = torch.stack(test_x), torch.cat(test_y)
dataset = data.TensorDataset(test_x, test_y)
test_loader_rep = DataLoader(dataset, shuffle=True)

# %%
# train and test classifier

epochs = 200
iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader_rep):
        images = Variable(images.view(-1, arch[-1]))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for i, (_images, _labels) in enumerate(test_loader_rep):
                _images = Variable(_images.view(-1, arch[-1]))
                outputs = classifier(_images)
                _, predicted = torch.max(outputs.data, 1)
                total += _labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == _labels).sum()
            accuracy = 100 * correct / total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

# %%
# tSNE clustering
print('tSNE clustering')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(train_x)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# %%
# visualisation
df = pandas.DataFrame()
df['tsne-one'] = tsne_results[:, 0]
df['tsne-two'] = tsne_results[:, 1]
df['y'] = train_y
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="y",
    palette=sns.color_palette("bright", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)

plt.show()
fig.savefig(os.path.join(file_path, 'tSNE_clustering_rep'))

# %%
# sort representations by class first before passing to rdm function
train_indices = np.argsort(train_y)
train_x = train_x[train_indices]

test_indices = np.argsort(test_y)
test_x = test_x[test_indices]

rdm_train = rdm_w_rep(train_x, 'cosine', istrain=True)
rdm_train.savefig(os.path.join(file_path, 'rdm_train.png'))
rdm_test = rdm_w_rep(test_x, 'cosine', istrain=False)
rdm_test.savefig(os.path.join(file_path, 'rdm_test.png'))

# %%
