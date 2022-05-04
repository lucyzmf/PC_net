# this script returns the baseline classification acc on images of dataset
# contains three different classification methods: knn, pure linear classifier, and logistic regression

import pandas
import torch.profiler
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

from evaluation import *

if torch.cuda.is_available():  # Use GPU if possible
    dev = "cuda:0"
    print("Cuda is available")
else:
    dev = "cpu"
    print("Cuda not available")
device = torch.device(dev)

dtype = torch.float  # Set standard datatype

# load config
CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_9')
dataDir = '/40_10perclass_largerangle/'

config = load_config("config.yaml")

padding = config['padding_size']
data_width = 28 + padding * 2
num_classes = 10


# %%
# load images
# train_set = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_image.pt'))
# test_set = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_image.pt'))
train_set = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtrain_image.pt'))
test_set = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtest_image.pt'))

# %%
train_indices = train_set.indices
test_indices = test_set.indices

# %%
train_images = train_set.dataset.data[train_indices]
train_images = nn.functional.pad(train_images, (padding, padding, padding, padding))
train_images = torch.flatten(train_images, start_dim=1).numpy()
train_labels = train_set.dataset.targets[train_indices].numpy()

# %%
test_images = test_set.dataset.data[test_indices]
test_images = nn.functional.pad(test_images, (padding, padding, padding, padding))
test_images = torch.flatten(test_images, start_dim=1).numpy()
test_labels = test_set.dataset.targets[test_indices].numpy()

# %%
print('Assess how clustered train still images are')
acc_train_still = within_sample_classification_stratified(train_images, train_labels)
print('within-sample linear regression (stratified kfold) on train still images: %.4f' % acc_train_still)

# %%
print('Assess how generalisable linear classifiers (regression and knn) are on still images')
# acc of linear classifier on images
_, acc_knn = knn_classifier(train_images, train_labels, test_images, test_labels)
print('knn classifier test acc on still images %.4f' % acc_knn)


# %%
cum_acc_train, cum_acc_test = linear_regression(train_images, train_labels, test_images, test_labels)
print('linear regression test acc on still %.4f' % cum_acc_test)


# %%
# logreg
#  pytorch logistic regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


classifier = LogisticRegression(data_width ** 2, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# train and test classifier
train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(test_set, shuffle=True)

epochs = 200
iter = 0
final_acc = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, data_width ** 2)).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 5000 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for i, (_images, _labels) in enumerate(test_loader):
                _images = Variable(_images.view(-1, data_width ** 2)).to(device)
                _labels = Variable(_labels).to(device)
                outputs = classifier(_images)
                _, predicted = torch.max(outputs.data, 1)
                total += _labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == _labels).sum()
            accuracy = 100 * correct / total
            final_acc = accuracy
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

print('logistic regression test acc on still img: %.2f' % final_acc)

# %%
images_all = np.concatenate((train_images, test_images))
labels_all = np.concatenate((train_labels, test_labels))
# use clustering technique on flattened images to examine baseline clusters of dataset
print('tSNE clustering')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(images_all)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# %%
# visualisation
df = pandas.DataFrame()
df['tsne-one'] = tsne_results[:, 0]
df['tsne-two'] = tsne_results[:, 1]
df['y'] = labels_all
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="y",
    palette=sns.color_palette("bright", 5),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)

plt.title('tSNE clustering baseline on fashionMNIST images ')
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'tSNE_clustering_rep'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'tSNE_clustering_rep'))

# %%
######################
# classification of sequence training images
######################

# train_set_spin = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_loader_spin.pth'))
# test_set_spin = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_loader_spin.pth'))
train_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtrain_set_spin.pt'))
trainLoaderSpin = data.DataLoader(train_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                  pin_memory=config['pin_mem'], shuffle=False)
test_set_spin = torch.load(os.path.join(file_path + dataDir, 'fashionMNISTtest_set_spin.pt'))
testLoaderSpin = data.DataLoader(test_set_spin, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                 pin_memory=config['pin_mem'], shuffle=False)
# %%
train_seq_spin, train_labels_spin = [], []
for i, (_frame, _label) in enumerate(trainLoaderSpin):
    train_seq_spin.append(torch.flatten(_frame).data.numpy())
    train_labels_spin.append(_label.data)

train_seq_spin = np.vstack(train_seq_spin)
train_labels_spin = torch.concat(train_labels_spin).numpy()

# %%
test_seq_spin, test_labels_spin = [], []
for i, (_frame, _label) in enumerate(testLoaderSpin):
    test_seq_spin.append(torch.flatten(_frame).data.numpy())
    test_labels_spin.append(_label.data)

test_seq_spin = np.vstack(test_seq_spin)
test_labels_spin = torch.concat(test_labels_spin).numpy()

# %%
# shuffle frames in sequence dataset
idx = np.arange(len(train_seq_spin))
idx = np.random.permutation(idx)
train_seq_spin = train_seq_spin[idx]
train_labels_spin = train_labels_spin[idx]

idx = np.arange(len(test_seq_spin))
idx = np.random.permutation(idx)
test_seq_spin = test_seq_spin[idx]
test_labels_spin = test_labels_spin[idx]

# %%
print('Assess how clustered train seq frames are')
acc_train_seq = within_sample_classification_stratified(train_seq_spin, train_labels_spin)
print('within-sample linear regression (stratified kfold) on train seq frames: %.4f' % acc_train_seq)

# %%

cum_acc_train_spin, cum_acc_test_spin = linear_regression(train_seq_spin, train_labels_spin, test_seq_spin,
                                                          test_labels_spin)
print('linear regression test acc on sequence dataset %.4f' % cum_acc_test_spin)


# %%
# acc of linear classifier on images
_, acc_knn_test_spin = knn_classifier(train_seq_spin, train_labels_spin, test_seq_spin, test_labels_spin)
print('knn classifier test acc on sequence dataset %.4f' % acc_knn_test_spin)

# %%
# acc of linear classifier trained on frames of sequence and tested on still images
####################
# key baseline
####################

acc_train_seq, acc_test_still = linear_regression(train_seq_spin, train_labels_spin, test_images, test_labels)
print('linear regression trained on sequence frames tested on still test acc %.4f' % acc_test_still)

acc_train_seq, acc_test_still = knn_classifier(train_seq_spin, train_labels_spin, test_images, test_labels)
print('knn classifier trained on sequence frames tested on still test acc %.4f' % acc_test_still)

# %%
#######################
# visualisation of sequence statistics
#######################

# rdm of all images
# sort labels first
sorted_label, indices = torch.sort(torch.tensor(train_labels))
train_labels = train_labels[indices]
train_images = train_images[indices]

sorted_label, indices = torch.sort(torch.tensor(test_labels))
test_labels = test_labels[indices]
test_images = test_images[indices]

images = np.concatenate((train_images, test_images))

pair_dist_cosine = pairwise_distances(images, metric='cosine')

fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of train and test images (sorted by class within each dataset)')
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'RDM coscience of train images'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'RDM coscience of train images'))

# %%
# rdm of train sequences
# train_loader_seq = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_loader_spin.pth'))
seq_frames = []
seq_labels = []
plotting = []
for i, (_image, _label) in enumerate(trainLoaderSpin):
    seq_frames.append(torch.flatten(torch.squeeze(_image)).numpy())
    plotting.append(torch.squeeze(_image).numpy())
    seq_labels.append(_label)
seq_labels = torch.cat(seq_labels)
seq_frames = np.vstack(seq_frames)

# sort frames
seq_sort_label, seq_indices = torch.sort(seq_labels)
seq_labels = seq_labels[seq_indices]
seq_frames = seq_frames[seq_indices]
# %%
# example sequence
fig, axs = plt.subplots(1, config['frame_per_sequence'], sharey=True, figsize=(20, 5))
for i in range(config['frame_per_sequence']):
    axs[i].imshow(plotting[i+279])
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'example sequence in training set'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'example sequence in training set'))



# %%
# rdm of one sequence
pair_dist_cosine = pairwise_distances(seq_frames[:9], metric='cosine')

fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames one sequence in training set')
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames one sequence in training set'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'RDM coscience of frames of one sequence in training set'))


# %%
# rdm of a class of sequence
pair_dist_cosine = pairwise_distances(seq_frames[:config['frame_per_sequence']*4*10], metric='cosine')
fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames one class in training set')
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames one class in training set'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'RDM coscience of frames one class in training set'))


# %%
# rdm of all classes of sequences
pair_dist_cosine = pairwise_distances(seq_frames, metric='cosine')
fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames all classes in training set')
plt.show()
# fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames all classes in training set'))
# fig.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs', 'RDM coscience of frames all classes in training set'))

# %%
pair_dist_cosine = pairwise_distances(seq_frames, metric='cosine')
# %%
fig, ax = plt.subplots()
# im = ax.imshow(pair_dist_cosine)
sns.heatmap(pair_dist_cosine, cmap='mako')
# fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames all classes in training set')
plt.show()