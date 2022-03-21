# this script returns the baseline classification acc on images of dataset
import os
import time

import pandas
import seaborn as sns
import torch.profiler
import yaml
from sklearn.manifold import TSNE
from torch.autograd import Variable

from evaluation import *

now = datetime.datetime.now()

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

# %%
# load images
train_loader = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_loader.pth'))
test_loader = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_loader.pth'))


# %%
# linear classifier fitted on all samples and tested with stratified kfold knn
def linear_classifier_kfold(data_loaders):
    rep_list = []
    labels_list = []
    for i in range(len(data_loaders)):
        for i, (_image, _label) in enumerate(data_loaders[i]):
            rep_list.append(_image.squeeze().flatten().numpy())
            labels_list.append(_label.numpy())
    rep_list = np.array(rep_list)
    labels_list = np.squeeze(np.array(labels_list))
    #     print(labels)

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
rep_list, labels_list, acc_knn = linear_classifier_kfold([train_loader, test_loader])
print(acc_knn)


# %%
# linear classifier fitted to train loader images and tested on test loader images
def linear_classifier(train_loader, test_loader):
    rep_train, rep_test = [], []
    labels_train, labels_test = [], []

    for i, (_image, _label) in enumerate(train_loader):
        rep_train.append(_image.squeeze().flatten().numpy())
        labels_train.append(_label.numpy())

    for i, (_image, _label) in enumerate(test_loader):
        rep_test.append(_image.squeeze().flatten().numpy())
        labels_test.append(_label.numpy())

    rep_train = np.array(rep_train)
    rep_test = np.array(rep_test)
    labels_train = np.squeeze(np.array(labels_train))
    labels_test = np.squeeze(np.array(labels_test))
    #     print(labels)

    # avg classification performance over 5 rounds
    cumulative_accuracy = 0
    for i in range(5):
        labels_train_vec = F.one_hot(torch.tensor(labels_train)).numpy()
        labels_test_vec = F.one_hot(torch.tensor(labels_test)).numpy()

        reg = linear_model.LinearRegression()
        reg.fit(rep_train, labels_train_vec)
        labels_predicted = reg.predict(rep_test)

        # Convert to one-hot
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)

        # Calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        accuracy = accuracy_score(labels_test_vec, labels_predicted)

        cumulative_accuracy += accuracy / 5

    return cumulative_accuracy


# %%
cum_acc = linear_classifier(train_loader, test_loader)
print('cumulative accuracy over 5 runs for linear classifier %.2f' % cum_acc)

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


classifier = LogisticRegression(784, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
classifier.to(device)

# %%
# train and test classifier
epochs = 200
iter = 0
final_acc = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 784)).to(device)
        labels = Variable(labels).to(device)

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
            for i, (_images, _labels) in enumerate(test_loader):
                _images = Variable(_images.view(-1, 784)).to(device)
                _labels = Variable(_labels).to(device)
                outputs = classifier(_images)
                _, predicted = torch.max(outputs.data, 1)
                total += _labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == _labels).sum()
            accuracy = 100 * correct / total
            final_acc = accuracy
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

print('final classification acc on image itself: %.2f' % final_acc)

# %%
# use clustering technique on flattened images to examine baseline clusters of dataset
print('tSNE clustering')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(rep_list)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

# %%
# visualisation
df = pandas.DataFrame()
df['tsne-one'] = tsne_results[:, 0]
df['tsne-two'] = tsne_results[:, 1]
df['y'] = labels_list
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
