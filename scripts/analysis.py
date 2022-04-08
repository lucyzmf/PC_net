import os

import numpy as np
import pandas as pd

from evaluation import *

file_path = os.path.abspath('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs')

# %%
df_rep_false = pd.read_pickle(
    '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/reps_df_reset_per_frame_false_.4.pkl')

# %%
# rdms for each layer in train set
rep_false_layer0_train = np.vstack(df_rep_false[df_rep_false['layer'] == 0][df_rep_false['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)
rep_false_layer1_train = np.vstack(df_rep_false[df_rep_false['layer'] == 1][df_rep_false['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)
rep_false_layer2_train = np.vstack(df_rep_false[df_rep_false['layer'] == 2][df_rep_false['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)

rep_false_layer0_test = np.vstack(df_rep_false[df_rep_false['layer'] == 0][df_rep_false['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)
rep_false_layer1_test = np.vstack(df_rep_false[df_rep_false['layer'] == 1][df_rep_false['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)
rep_false_layer2_test = np.vstack(df_rep_false[df_rep_false['layer'] == 2][df_rep_false['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)

# %%
false_train_labels = df_rep_false[df_rep_false['is_train'] == 1][df_rep_false['layer'] == 0]['labels'].to_numpy().astype(np.int)
# turn label entries into int
# for l in range(len(false_train_labels)):
#     false_train_labels[l] = int(false_train_labels[l][0])

false_test_labels = df_rep_false[df_rep_false['is_train'] == 0][df_rep_false['layer'] == 0]['labels'].to_numpy().astype(np.int)
# turn label entries into int
# for l in range(len(false_test_labels)):
#     false_test_labels[l] = int(false_test_labels[l][0])

# %%
false_dir = '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/reset_per_frame_false'
idx = np.argsort(false_train_labels)
fig = rdm_w_rep_title(rep_false_layer2_train[idx], 'cosine', 'rdm r_out layer 2')
fig.show()
fig.savefig(os.path.join(false_dir, 'rdm r_out layer 2'))

# %%
_, layer0_acc = linear_regression(rep_false_layer0_train, false_train_labels, rep_false_layer0_test, false_test_labels)
_, layer1_acc = linear_regression(rep_false_layer1_train, false_train_labels, rep_false_layer1_test, false_test_labels)
_, layer2_acc = linear_regression(rep_false_layer2_train, false_train_labels, rep_false_layer2_test, false_test_labels)

# %%
acc_by_layer_false = [layer0_acc, layer1_acc, layer2_acc]

# %%
'''reset per frame true'''

df_rep_true = pd.read_pickle(
    '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/reps_df_reset_per_frame_true_.4.pkl')

# %%
# rdms for each layer in train set
rep_true_layer0_train = np.vstack(df_rep_true[df_rep_false['layer'] == 0][df_rep_true['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)
rep_true_layer1_train = np.vstack(df_rep_true[df_rep_false['layer'] == 1][df_rep_true['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)
rep_true_layer2_train = np.vstack(df_rep_true[df_rep_false['layer'] == 2][df_rep_true['is_train'] == 1]['r_out'].to_numpy()).astype(np.float)

rep_true_layer0_test = np.vstack(df_rep_true[df_rep_true['layer'] == 0][df_rep_true['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)
rep_true_layer1_test = np.vstack(df_rep_true[df_rep_true['layer'] == 1][df_rep_true['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)
rep_true_layer2_test = np.vstack(df_rep_true[df_rep_true['layer'] == 2][df_rep_true['is_train'] == 0]['r_out'].to_numpy()).astype(np.float)

# %%
true_train_labels = df_rep_true[df_rep_true['is_train'] == 1][df_rep_true['layer'] == 0]['labels'].to_numpy().astype(np.int)

true_test_labels = df_rep_true[df_rep_true['is_train'] == 0][df_rep_true['layer'] == 0]['labels'].to_numpy().astype(np.int)


# %%
true_dir = '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/reset_per_frame_true'
idx = np.argsort(true_train_labels)
fig = rdm_w_rep_title(rep_true_layer2_train[idx], 'cosine', 'rdm r_out layer 2')
fig.show()
fig.savefig(os.path.join(true_dir, 'rdm r_out layer 2'))

# %%
# %%
_, layer0_acc = linear_regression(rep_true_layer0_train, true_train_labels, rep_true_layer0_test, true_test_labels)
_, layer1_acc = linear_regression(rep_true_layer1_train, true_train_labels, rep_true_layer1_test, true_test_labels)
_, layer2_acc = linear_regression(rep_true_layer2_train, true_train_labels, rep_true_layer2_test, true_test_labels)

# %%
acc_by_layer_true = [layer0_acc, layer1_acc, layer2_acc]

# %%
df_acc_by_layer = pd.DataFrame()
df_acc_by_layer['acc'] = np.concatenate((acc_by_layer_false, acc_by_layer_true))
df_acc_by_layer['seq_info'] = np.concatenate((np.ones(3), np.zeros(3)))
df_acc_by_layer['layer'] = [0, 1, 2, 0, 1, 2]

df_acc_by_layer.head()

 # %%
import seaborn as sns
sns.barplot(data=df_acc_by_layer, x='layer', y='acc', hue='seq_info')
plt.title('generalisation to unseen examplars across layers in fc network')
# plt.show()
plt.savefig(os.path.join('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_6/80 epochs/', 'generalisation to unseen examplars across layers in fc network.4.png'))

# %%
# testing on still images

