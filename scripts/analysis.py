import os

import yaml

from evaluation import *
from fc_net import FcDHPC
from rf_net_cm import RfDHPC_cm

file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs')

# %%
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
# inspect convergence of last layer
image = train_seq_spin[0]
inf_step = np.arange(0, 5000000)
high_layer_output = []
error = []

trained_net_false.init_states()
for i in inf_step:
    trained_net_false(torch.tensor(image), 1, istrain=False)
    high_layer_output.append(trained_net_false.states['r_output'][-1].detach().cpu().numpy())
    error.append(trained_net_false.states['error'][-2].detach().cpu().numpy())

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(inf_step, high_layer_output)
axs[0].set_title('output')
axs[1].plot(inf_step, error)
axs[1].set_title('error')
plt.show()
