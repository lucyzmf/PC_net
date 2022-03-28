# this script returns the baseline classification acc on images of dataset
# contains three different classification methods: knn, pure linear classifier, and logistic regression
import os
import time

import pandas
import seaborn as sns
import torch.profiler
import yaml
from sklearn.manifold import TSNE
from torch.autograd import Variable
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


config = load_config("config.yaml")

padding = config['padding_size']
data_width = 28 + padding * 2
num_classes = 10

# %%
# load images
train_set = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_image.pt'))
test_set = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_image.pt'))

# %%
train_indices = train_set.indices
test_indices = test_set.indices

# %%
train_images = train_set.dataset.data[train_indices]
train_images = torch.flatten(train_images, start_dim=1).numpy()
train_labels = train_set.dataset.targets[train_indices].numpy()

# %%
test_images = test_set.dataset.data[test_indices]
test_images = torch.flatten(test_images, start_dim=1).numpy()
test_labels = test_set.dataset.targets[test_indices].numpy()


# %%
# linear classifier fitted on all samples and tested with stratified kfold knn
def linear_classifier_kfold(train_images, train_labels, test_images, test_labels):
    rep_list = np.concatenate((train_images, test_images))
    labels_list = np.concatenate((train_labels, test_labels))

    # Select two samples of each class as test set, classify with knn (k = 5)
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # split into 5 folds
    skf.get_n_splits(rep_list, labels_list)
    # sample_size = len(data_loader)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(rep_list, labels_list):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = rep_list[train_index], rep_list[test_index]
        labels_train, labels_test = labels_list[train_index], labels_list[test_index]
        labels_train_vec = F.one_hot(torch.tensor(labels_train)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(labels_test)).numpy()

        # neigh = KNeighborsClassifier(n_neighbors=3, metric = distance) # build  KNN classifier for this fold
        # neigh.fit(reps_train, labels_train) # Use training data for KNN classifier
        # labels_predicted = neigh.predict(reps_test) # Predictions across test set

        reg = linear_model.LinearRegression()
        reg.fit(reps_train, labels_train_vec)
        labels_predicted = reg.predict(reps_test)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy += accuracy / 5

    return rep_list, labels_list, cumulative_accuracy


# %%
# acc of linear classifier on images
images_all, labels_all, acc_knn = linear_classifier_kfold(train_images, train_labels, test_images, test_labels)
print('cumulative accuracy over 5 folds for knn classifier %.2f' % acc_knn)


# %%
# linear classifier fitted to train loader images and tested on test loader images
def linear_classifier(train_images, train_labels, test_images, test_labels):
    # avg classification performance over 10 rounds
    cumulative_accuracy_train = 0
    cumulative_accuracy_test = 0
    for i in range(10):
        labels_train_vec = F.one_hot(torch.tensor(train_labels)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(test_labels)).numpy()

        reg = linear_model.LinearRegression()
        reg.fit(train_images, labels_train_vec)
        # assess train set acc
        labels_predicted_train = reg.predict(train_images)
        labels_predicted_train = (labels_predicted_train == labels_predicted_train.max(axis=1, keepdims=True)).astype(int)
        acc_train = accuracy_score(labels_train_vec, labels_predicted_train)
        cumulative_accuracy_train += acc_train / 10

        # assess performance on test set
        labels_predicted = reg.predict(test_images)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy_test += accuracy / 10

    return cumulative_accuracy_train, cumulative_accuracy_test


# %%
cum_acc_train, cum_acc_test = linear_classifier(train_images, train_labels, test_images, test_labels)
print('avg accuracy over 10 runs for linear classifier on train set %.4f' % cum_acc_train)
print('avg accuracy over 10 runs for linear classifier on test set %.4f' % cum_acc_test)


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

print('final logistic regression classification acc on image itself: %.2f' % final_acc)

# %%
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
    palette=sns.color_palette("bright", 10),
    data=df,
    legend="full",
    alpha=0.3,
    ax=ax1
)

plt.title('tSNE clustering baseline on fashionMNIST images ')
plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'tSNE_clustering_rep'))

# %%
# classification of sequence training images
train_set_spin = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_loader_spin.pth'))
test_set_spin = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_loader_spin.pth'))

# %%
train_seq_spin, train_labels_spin = [], []
for i, (_frame, _label) in enumerate(train_set_spin):
    train_seq_spin.append(torch.flatten(_frame).data.numpy())
    train_labels_spin.append(_label.data)

train_seq_spin = np.vstack(train_seq_spin)
train_labels_spin = torch.concat(train_labels_spin).numpy()

# %%
test_seq_spin, test_labels_spin = [], []
for i, (_frame, _label) in enumerate(test_set_spin):
    test_seq_spin.append(torch.flatten(_frame).data.numpy())
    test_labels_spin.append(_label.data)

test_seq_spin = np.vstack(test_seq_spin)
test_labels_spin = torch.concat(test_labels_spin).numpy()

# %%
cum_acc_train_spin, cum_acc_test_spin = linear_classifier(train_seq_spin, train_labels_spin, test_seq_spin, test_labels_spin)
print('avg accuracy over 10 runs for linear classifier on sequence train dataset %.4f' % cum_acc_train_spin)
print('avg accuracy over 10 runs for linear classifier on sequence test dataset %.4f' % cum_acc_test_spin)


# %%
# acc of linear classifier on images
images_all_spin, labels_all_spin, acc_knn_spin = linear_classifier_kfold(train_seq_spin, train_labels_spin, test_seq_spin, test_labels_spin)
print('cumulative accuracy over 5 folds for knn classifier on sequence dataset (train and test) %.4f' % acc_knn)


# %%
#######################
# visualisation of sequence statistics
#######################

# rdm of train images
# sort labels first
sorted_label, indices = torch.sort(torch.tensor(train_labels))
train_labels = train_labels[indices]
train_images = train_images[indices]

pair_dist_cosine = pairwise_distances(train_images, metric='cosine')

fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of train images')
plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'RDM coscience of train images'))

# %%
# rdm of train sequences
train_loader_seq = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_loader_spin.pth'))
seq_frames = []
seq_labels = []
plotting = []
for i, (_image, _label) in enumerate(train_loader_seq):
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
fig, axs = plt.subplots(1, 5, sharex=True)
for i in range(5):
    axs[i].imshow(plotting[i])
plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'example sequence'))


# %%
# rdm of one sequence
pair_dist_cosine = pairwise_distances(seq_frames[:5], metric='cosine')

fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames one sequence')
# plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames one sequence'))

# %%
# rdm of a class of sequence
pair_dist_cosine = pairwise_distances(seq_frames[:5*4*10], metric='cosine')
fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames one class')
# plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames one class'))


# %%
# rdm of all classes of sequences
pair_dist_cosine = pairwise_distances(seq_frames, metric='euclidean')
fig, ax = plt.subplots()
im = ax.imshow(pair_dist_cosine)
fig.colorbar(im, ax=ax)
ax.set_title('RDM cosine of frames all classes')
# plt.show()
fig.savefig(os.path.join(config['dataset_dir'], 'RDM cosine of frames all classes'))
