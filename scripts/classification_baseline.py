# this script returns the baseline classification acc on images of dataset
import datetime
import os

import torch.profiler
import yaml
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
train_loader = torch.load(os.path.join(config['dataset_dir'], 'train_loader.pth'))
test_loader = torch.load(os.path.join(config['dataset_dir'], 'test_loader.pth'))


# %%
# linear classifier
def linear_classifier(data_loaders):
    rep_list = [];
    labels = []
    for i in range(len(data_loaders)):
        for i, (_image, _label) in enumerate(data_loaders[i]):
            rep_list.append(_image.squeeze().flatten().numpy())
            labels.append(_label.numpy())
    rep_list = np.array(rep_list)
    labels = np.squeeze(np.array(labels))
    #     print(labels)

    # Select two samples of each class as test set, classify with knn (k = 5)
    skf = StratifiedKFold(n_splits=5, shuffle=True)  # split into 5 folds
    skf.get_n_splits(rep_list, labels)
    # sample_size = len(data_loader)
    cumulative_accuracy = 0
    # Now iterate through all folds
    for train_index, test_index in skf.split(rep_list, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        reps_train, reps_test = rep_list[train_index], rep_list[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
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

    return cumulative_accuracy


# %%
# acc of linear classifier on images
acc = linear_classifier([train_loader, test_loader])
print(acc)


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
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))


# %%
