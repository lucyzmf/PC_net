import glob
import os

import yaml

from evaluation import *
from fc_net import FcDHPC
from rf_net_cm_scratch import RfDHPC_cm

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

if config['architecture'] == 'FcDHPC':
    trained_net = FcDHPC(config['network_size'], config['infrates']*10, lr=config['learning_rate'],
                              act_func=config['act_func'],
                              device=device, dtype=dtype)
elif config['architecture'] == 'RfDHPC_cm':
    trained_net = RfDHPC_cm(config['network_size'], config['rf_sizes'], config['infrates'] * 4,
                            lr=config['learning_rate'],
                            act_func=config['act_func'],
                            device=device, dtype=dtype)

trained_net.load_state_dict(
    torch.load(glob.glob(file_path + '/reset_per_frame_false/trained_model/*readout.pth')[0],
               map_location=torch.device('cpu')))

# %%
# inspect convergence of last layer
image = train_seq_spin[0]
inf_step = np.arange(0, 50000)
high_layer_output = []
mid_layer_output = []
input_layer_output = []
error_intput = []
error_mid = []

trained_net.init_states()
for i in inf_step:
    trained_net(torch.tensor(image), 1, istrain=False)
    high_layer_output.append(trained_net.states['r_output'][-1].detach().cpu().numpy())
    mid_layer_output.append(trained_net.states['r_output'][-2].detach().cpu().numpy())
    input_layer_output.append(trained_net.states['r_output'][0].detach().cpu().numpy())

    error_intput.append(trained_net.states['error'][0].detach().cpu().numpy())
    error_mid.append(trained_net.states['error'][-2].detach().cpu().numpy())

# %%
fig, axs = plt.subplots(2, 3, figsize=(24, 10))
w_0_1 = trained_net.layers[0].weights
bu_error_to_hidden = np.matmul(torch.transpose(w_0_1, 0, 1).numpy(), np.transpose(np.vstack(error_intput)))

axs[0][0].plot(inf_step, (np.transpose(bu_error_to_hidden) - error_mid))
axs[0][0].set_title('amount of update to hidden layer')
axs[0][1].plot(inf_step, mid_layer_output)
axs[0][1].set_title('hidden layer output')
axs[0][2].plot(inf_step, high_layer_output)
axs[0][2].set_title('highest layer output')

mse_input = np.mean(np.vstack(error_intput), axis=1) ** 2
mse_mid = np.mean(np.vstack(error_mid), axis=1) ** 2

axs[1][0].plot(inf_step, mse_input)
axs[1][0].set_title('input layer MSE')
axs[1][1].plot(inf_step, mse_mid)
axs[1][1].set_title('hidden layer MSE')

# plot bu_error - e_act to layer 2
w_1_2 = trained_net.layers[1].weights
bu_error_hidden_to_last = np.matmul(torch.transpose(w_1_2, 0, 1).numpy(), np.transpose(np.vstack(error_mid)))
axs[1][2].plot(inf_step, (np.transpose(bu_error_hidden_to_last)))
axs[1][2].set_title('amount of update to last layer')

plt.show()
# plt.savefig(trained_model_dir + '/convergence.png')