import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances  # computes the pairwise distance between observations

# %%
####################################
# data generation
####################################

# Create array of correct dimensions to store data.
dataWidth = 7  # for simplicity used for vertical bars as well
barWidth = 2
frameNum = dataWidth - barWidth + 1  # until bar reaches the other side
seqNum = 8  # (vertical, horizontal, diagonal 1, diagonal 2 bar) * (direction 1, 2)
data = np.zeros(shape=(seqNum, frameNum, dataWidth, dataWidth))
flat_data = np.zeros(shape=(seqNum, frameNum, dataWidth ** 2))

# For diagonal bars: Create larger, still image with diagonal in the center and later take shifting quadratic frames
# from it
diagonal_full_1 = np.zeros([dataWidth, 3 * dataWidth])
diagonal_full_2 = np.zeros([dataWidth, 3 * dataWidth])
for i in range(barWidth):
    diagonal_full_1 += np.roll(
        np.concatenate((np.zeros([dataWidth, dataWidth]), np.identity(dataWidth), np.zeros([dataWidth, dataWidth])), 1),
        -i)
    diagonal_full_2 += np.roll(np.concatenate(
        (np.zeros([dataWidth, dataWidth]), np.fliplr(np.identity(dataWidth)), np.zeros([dataWidth, dataWidth])), 1), -i)

# Create frames.
for seqit in range(seqNum):
    for frameit in range(frameNum):
        # Horizontal bars
        if seqit < 2:
            for barit in range(barWidth):
                data[seqit, frameit, barit + frameit, :] = 1  # Position of bar is shifted each frameit
            # Flip second sequence: Upward moving bar
            if seqit == 1:
                data[seqit, frameit] = np.flipud(data[seqit, frameit])
        # Vertical bars
        elif seqit < 4:
            for barit in range(barWidth):
                data[seqit, frameit, :, barit + frameit] = 1
            if seqit == 3:
                data[seqit, frameit] = np.fliplr(data[seqit, frameit])
        # Diagonal bars 1
        elif seqit == 4:
            startId = dataWidth - barWidth + int(dataWidth / 2) - frameit + 1
            data[seqit, frameit] = diagonal_full_1[:, (startId):(startId + dataWidth)]
        elif seqit == 5:
            data[seqit, frameit] = data[4, frameNum - 1 - frameit]  # Reverse order of sequence 4
        # Diagonal bars 2
        elif seqit == 6:
            startId = dataWidth - barWidth + int(dataWidth / 2) - frameit + 1
            data[seqit, frameit] = diagonal_full_2[:, (startId):(startId + dataWidth)]
        elif seqit == 7:
            data[seqit, frameit] = data[6, frameNum - 1 - frameit]

        flat_data[seqit, frameit] = data[seqit, frameit].flatten()  # Make each frame 1D to use as network input


# %%
# visualise dataset

# vertical bar 1 sequence
# fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
# count = 0
# for row in range(2):
#     for col in range(3):
#         axs[row, col].imshow(np.reshape(flat_data[0, count, :], (7,7)))
#         count += 1
#
# plt.show()
# plt.close(fig)
#
# # diagonal bar 2 sequence
# fig2, axs = plt.subplots(2, 3, sharex=True, sharey=True)
# count = 0
# for row in range(2):
#     for col in range(3):
#         axs[row, col].imshow(np.reshape(flat_data[7, count, :], (7,7)))
#         count += 1
#
# plt.show()
# plt.close(fig2)

# %%
# helper functions

def sigmoid(x):  # sigmoid activation function
    return scipy.special.expit(x-3)


# %%
# defining layer and network class

class Layer:
    def __init__(self, input_dim, output_dim, inf_rate, islast=False, act_func=sigmoid):
        self.size = input_dim
        self.output_dim = output_dim  # size of next layer
        self.r_activation = np.zeros(input_dim)  # initialise representational units
        self.r_output = np.zeros(input_dim)  # output firing rates of r units
        self.inf_rate = inf_rate  # inference rate governing how fast r adjust to errors
        self.islast = islast
        self.act_func = act_func
        if not self.islast:
            self.e_activation = np.zeros(input_dim)  # initialise error units
            self.weights = np.random.normal(0, .5, [input_dim, output_dim])  # weight matrix
            self.weights = self.weights.clip(min=0)  # clip weights
            self.weights = self.weights/output_dim
            self.r_prediction = np.zeros(input_dim)  # prediction imposed by higher layer

    def __call__(self):
        self.r_output = self.act_func(self.r_activation)


# %%

class PC_Net:
    def __init__(self, network_dim, inf_rates, inference_steps=10,
                 lr=0.01):  # defining network architecture and parameters
        self.lr = lr  # learning rate of weight updates
        self.inf_steps = inference_steps  # num of inference steps per input before weight update
        self.inf_rates = inf_rates
        self.architecture = network_dim

        self.layers = []

        # build network layers given network dimensions
        for i in range(len(network_dim)):
            if i != len(network_dim) - 1:
                _layer = Layer(network_dim[i], network_dim[i + 1], inf_rates[i])
                self.layers.append(_layer)
            else:
                _layer = Layer(network_dim[i], network_dim[i], inf_rates[i], islast=True)
                self.layers.append(_layer)

    def initialise_states(self):
        # first layer activation reflect inputs, higher layer activation initialised to small constant value = 0.1
        for layer in self.layers:
            if layer.size != self.architecture[0]:  # if layer size = input size
                layer.r_activation = np.full(layer.size, -2)
                layer()

    def __call__(self, inputs, inf_steps=5):
        # initialise network
        self.layers[0].r_activation = self.layers[0].r_output = inputs

        error_iter = []  # each element is the sum of activation of all error units per inference step

        # inference pass
        for iter in range(inf_steps):
            error_layer = []  # each element is avg square error in each layer
            for i in range(len(self.layers) - 1):  # iterate through non last layers
                _layer = self.layers[i]
                # generate prediction
                _layer.r_prediction = _layer.weights @ self.layers[i + 1].r_output
                assert len(_layer.r_prediction) == _layer.size
                # calculate error
                _layer.e_activation = _layer.r_output - _layer.r_prediction
                error_layer.append(np.average(np.power(_layer.e_activation, 2)))  # average squared error per layer

            error_iter.append(error_layer)

            # update of r activation given error
            for i in np.arange(1, len(self.layers)):  # iterate through non first layers
                if not self.layers[i].islast:
                    self.layers[i].r_activation = self.layers[i].r_activation + self.layers[i].inf_rate * \
                                                  (self.layers[i - 1].weights.T @ self.layers[i - 1].e_activation
                                                   - self.layers[i].e_activation)
                else:
                    self.layers[i].r_activation = self.layers[i].r_activation + self.layers[i].inf_rate * \
                                                  (self.layers[i - 1].weights.T @ self.layers[i - 1].e_activation)
                assert len(self.layers[i].r_activation) == self.layers[i].size
                self.layers[i]()  # compute output given updated activation

        total_error = np.average(error_iter)  # total error per inference call

        return total_error  #, error_iter[-1]

    def weight_update(self):  # update weights
        for i in range(len(self.layers) - 1):
            _layer = self.layers[i]
            delta = self.lr * (np.transpose([_layer.e_activation]) @ [
                self.layers[i + 1].r_output])  # making both 1d arrays nd to perform multiplication
            assert np.shape(delta) == np.shape(_layer.weights)
            _layer.weights = _layer.weights + delta
            _layer.weights = _layer.weights.clip(min=0)

    def generate(self, inputs, inf_steps_gen=50):  # given input generate predictions
        self.initialise_states()
        self.layers[0].r_activation = self.layers[0].r_output = inputs

        error = self.__call__(inputs, inf_steps=inf_steps_gen)  # infer given input

        self.layers[0].r_activation = self.layers[0].weights @ self.layers[1].r_output

        return self.layers[0].r_activation, error

    def reset(self):  # reset all activation and weights
        for layer in self.layers:
            layer.r_activation = np.zeros(layer.size)
            layer.r_output = np.zeros(layer.size)
            if not layer.islast:
                layer.e_activation = np.zeros(layer.size)
                layer.r_prediction = np.zeros(layer.size)
                layer.weights = np.random.normal(0, .5, [layer.size, layer.output_dim])  # weight matrix
                layer.weights = layer.weights.clip(min=0)  # clip weights
                layer.weights = layer.weights / layer.output_dim
    # def add_layer


# %%
##############################
# instantiating network
##############################
input_dim = dataWidth ** 2
network_dimensions = [input_dim, 50, 25, 10]  # input, hidden, output dimensions of network
inf_baserate = .2  # base inference rate
inf_rates = [inf_baserate, inf_baserate, inf_baserate/5, inf_baserate/5]  # ajustment rate of representations for each layer
learningRate = 0.01

net = PC_Net(network_dimensions, inf_rates, lr=learningRate)

# %%

# training the network just one sample
# samples = flat_data[0, :, :]
#
# error = []
#
# epochNum = 2000
# for epoch in range(epochNum):
#     _e = []
#     for i in samples:
#         # net.initialise_states()
#         e, _ = net(i, training=True)  # error at the end of each inference call
#         _e.append(e)
#
#     error.append(np.average(_e))
#
# x = np.arange(0, epochNum)
# y = error
# plt.plot(x, y)
# plt.show()
# %%
#############################
# training loop
#############################

epochNum = 20
repNum = 10  # num of times a sequence is repeated before moving to next sequence
frameRep = 5  # num of times of repeat per frame

net.reset()
avg_error = []

for epoch in range(epochNum):
    errors = []  # errors of last inference steps per epoch
    for seq in range(seqNum):
        net.initialise_states()
        for rep in range(repNum):
            for frame in range(frameNum):
                for i in range(frameRep):
                    inputs = flat_data[seq, frame, :]
                    error_last = net(inputs, inf_steps=10)
                    errors.append(error_last)
                    net.weight_update()

    avg_error.append(np.average(errors))

x = np.arange(epochNum)
y = avg_error
plt.plot(x, y)
plt.show()

# save trained network
net_trained = net

# %%
#############################
# testing
#############################

# extract highest level of representation for RSA
representation = []
for seq in range(0, seqNum, 2):
    for frame in range(frameNum):
        net_trained.initialise_states()
        net_trained(flat_data[seq, frame, :], inf_steps=6000)
        representation.append(net_trained.layers[-1].r_activation)

pair_dist_euclidean = pairwise_distances(representation)  # euclidean distance of high level representations
pair_dist_cosine = pairwise_distances(representation, metric='cosine')  # cosine distance of high level representations

fig, axs = plt.subplots(1, 2)
im1 = axs[0].imshow(pair_dist_euclidean)
axs[0].set_title('euclidean')
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(pair_dist_cosine)
axs[1].set_title('cosine')
fig.colorbar(im2, ax=axs[1])
plt.show()

# plt.savefig('/Users/lucyzhang/Documents/research/PC_net/results/RSA_last_layer.png')

# %%

# generate test data for generative capacity
testFrame = flat_data[np.random.randint(0, seqNum), np.random.randint(0, frameNum), :]
# testFrame[:21] = 0  # partially mask test frame

recoveredFrame, _ = net_trained.generate(testFrame)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.reshape(testFrame, (7, 7)), vmin=0, vmax=1)
ax2.imshow(np.reshape(recoveredFrame, (7, 7)), vmin=0, vmax=1)
plt.show()  # regeneration of image show average learning

# %%
# inspect weight distribution
fig, axs = plt.subplots(1, len(network_dimensions)-1, sharey=True)
for i in range(len(network_dimensions)-1):
    axs[i].hist(net_trained.layers[i].weights)
plt.show()
# %%
# inspect output distribution
fig, axs = plt.subplots(1, len(network_dimensions), sharey=True)
for i in range(len(network_dimensions)):
    axs[i].hist(net_trained.layers[i].r_output)
plt.show()

# %%
# plot output of last layer during inference in trained network
steps = 4000
x = np.arange(steps)

outputs = []
net_trained.initialise_states()
for inf in range(steps):
    net_trained(testFrame, inf_steps=1)
    outputs.append(net_trained.layers[-1].r_output)

outputs = np.reshape(outputs, (steps, 10))

for i in range(10):
    plt.plot(x, outputs[:, i])
plt.show()

# %%
# train a classifier to see whether representation of last layer can predict sequence category
#
# X = []
# Y = []
# for seq in seqNum:
#     for frame in frameNum:
#         X.append(flat_data[seq, frame:])
#         Y.append(seq)
#
# # train test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=len(X) * .7)