# %%
from glob import glob

from evaluation import *

# %%
#######################
# plot metrics logged during training
#######################

# import metrics log files from all trained models
filedir = '/Users/lucyzhang/Documents/research/PC_net/results/morph_test_10/'
dataDir = '/trainesize160perclass'

conti_morph_log = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/trained_model/*metrics_log.pkl'))[0])
disconti_morph_log = pd.read_pickle(
    glob(os.path.join(filedir, 'discontinuous_morph/trained_model/*metrics_log.pkl'))[0])
conti_nomorph_log = pd.read_pickle(glob(os.path.join(filedir, 'continuous_nomorph/trained_model/*metrics_log.pkl'))[0])
still_control_log = pd.read_pickle(glob(os.path.join(filedir, 'still_control/trained_model/*metrics_log.pkl'))[0])

raise Exception('stop')
# %%
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
epochs = []
total_error = []
condition = []

for i in range(4):
    epochs.append(np.arange(40))
    condition.append(np.full(40, i))

total_error = np.hstack((conti_morph_log['total_error'],
                         disconti_morph_log['total_error'],
                         conti_nomorph_log['total_error'],
                         still_control_log['total_error']))

df_error_log = pd.DataFrame()
df_error_log['error outputs'] = total_error
df_error_log['epoch'] = np.hstack(epochs)
df_error_log['condition'] = np.hstack(condition)

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.despine()
sns.lineplot(data=df_error_log, x='epoch', y='error outputs', hue='condition', palette='tab10')
h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['continuous morph', 'discontinuous morph', 'continuous no morph', 'still control'],
          bbox_to_anchor=(1, 0.5),
          frameon=False)
plt.yscale('log')
plt.show()

# %%
#######################
# compare continuous morph vs no morph
#######################

# 1. decoding from representations from test set by layer
# load rep df
frame_rep_contimorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/**/frame_rep.pkl'))[0])
still_rep_continomorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_nomorph/**/still_rep.pkl'))[0])
still_rep_stillcontrol = pd.read_pickle(glob(os.path.join(filedir, 'still_control/**/still_rep.pkl'))[0])

# %%
df_genacc_morph = generate_acc_df([frame_rep_contimorph, still_rep_continomorph], [0, 1],
                                  'morph_or_no_morph', isGen=True)

# %%
# fig = plt.plot()
# sns.catplot(data=df_genacc_morph, col='accIsTest', x='layer', y='acc', hue='morph_or_no_morph', kind='bar')
# # plt.title('generalisation acc by layer morph vs no morph')
# plt.show()
# plt.close()

# %%
# plot acc
sns.set_context("paper", font_scale=1.5)
# %%
fig, ax = plt.subplots(figsize=(10,5))
sns.despine()
sns.barplot(data=df_genacc_morph[df_genacc_morph['accIsTest'] == 1][df_genacc_morph['layer'] != 0], x='layer', y='acc',
            hue='morph_or_no_morph', ax=ax, palette='tab10')
# ax.set_title('generalisation acc by layer morph vs no morph')

# add baseline classification on input
plt.axhline(y=.455, linestyle='dashed', color='black', label='linear classification with still data')
plt.axhline(y=0.8894, linestyle='dotted', color='black', label='linear classification with seq data')
plt.xlabel('area')
plt.ylabel('accuracy')
h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['classification with sampled fashion MNIST', 'classification with sequence data',
              'continuous morph', 'continuous no morph', ], frameon=False, loc='center left',
          bbox_to_anchor=(1, 0.5), ncol=1, fontsize=12)
plt.tight_layout()

plt.show()
plt.close()


# %%
# 2. rdm from layers
# top layer
isTrain = 1
fig1 = rdm_w_rep(np.vstack(
    seq_rep_contimorph[seq_rep_contimorph['is_train'] == isTrain][seq_rep_contimorph['layer'] == 3]['r_out'].to_numpy()),
    seq_rep_contimorph[seq_rep_contimorph['is_train'] == isTrain][seq_rep_contimorph['layer'] == 3][
        'labels'].to_numpy(),
    'cosine', ticklabel=640, title='continuous morph train reps')
plt.show()
# %%
fig2 = rdm_w_rep(np.vstack(
    still_rep_continomorph[still_rep_continomorph['is_train'] == isTrain][still_rep_continomorph['layer'] == 3][
        'r_out'].to_numpy()),
    still_rep_continomorph[still_rep_continomorph['is_train'] == isTrain][still_rep_continomorph['layer'] == 3][
        'labels'].to_numpy(),
    'cosine', ticklabel=160, title='continuous no morph train reps')
plt.show()
plt.close()

# %%
# rdm from still control
isTrain = 1
fig3 = rdm_w_rep(np.vstack(
    still_rep_stillcontrol[still_rep_stillcontrol['is_train'] == isTrain][still_rep_stillcontrol['layer'] == 3][
        'r_out'].to_numpy()),
    still_rep_stillcontrol[still_rep_stillcontrol['is_train'] == isTrain][still_rep_stillcontrol['layer'] == 3][
        'labels'].to_numpy(),
    'cosine', ticklabel=160, title='still control top area latent representations (train)')
plt.show()
plt.close()

# %%
# compute mean distance for within and between class
# contimorph
still_rep_contimorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/**/still_rep.pkl'))[0])
# %%
dist_contimorph = cosine_dis(np.vstack(
    still_rep_contimorph[still_rep_contimorph['layer'] == 3]['r_out'].to_numpy()),
    still_rep_contimorph[still_rep_contimorph['layer'] == 3][
        'labels'].to_numpy(),
    'cosine')

mean_dis_class_contimorph = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        mean_dis_class_contimorph[i, j] = dist_contimorph[i*200:(i+1)*200, j*200:(j+1)*200].mean()

sns.heatmap(mean_dis_class_contimorph, annot=True, vmax=1.0, vmin=0)
plt.xlabel('mean within class distance: %.3f, mean between class distance: %.3f' % (
        np.sum(np.identity(5)*mean_dis_class_contimorph)/5,
        np.sum((np.ones((5, 5))-np.identity(5))*mean_dis_class_contimorph)/20))
plt.show()

# %%
# continomorph
dist_continomorph = cosine_dis(np.vstack(
    still_rep_continomorph[still_rep_continomorph['layer'] == 3][
        'r_out'].to_numpy()),
    still_rep_continomorph[still_rep_continomorph['layer'] == 3][
        'labels'].to_numpy(),
    'cosine')

mean_dist_class_continomorph = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        mean_dist_class_continomorph[i, j] = np.mean(dist_continomorph[i*200:(i+1)*200, j*200:(j+1)*200])

sns.heatmap(mean_dist_class_continomorph, annot=True, vmax=1.0, vmin=0)
plt.xlabel('mean within class distance: %.3f, mean between class distance: %.3f' % (
        np.sum(np.identity(5)*mean_dist_class_continomorph)/5,
        np.sum((np.ones((5, 5))-np.identity(5))*mean_dist_class_continomorph)/20))
plt.show()

# %%
# still control
dist_stillcontrol = cosine_dis(np.vstack(
    still_rep_stillcontrol[still_rep_stillcontrol['layer'] == 3][
        'r_out'].to_numpy()),
    still_rep_stillcontrol[still_rep_stillcontrol['layer'] == 3][
        'labels'].to_numpy(),
    'cosine')

mean_dist_class_stillcontrol = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        mean_dist_class_stillcontrol[i, j] = np.mean(dist_stillcontrol[i*200:(i+1)*200, j*200:(j+1)*200])

sns.heatmap(mean_dist_class_stillcontrol, annot=True, vmax=1.0, vmin=0)
plt.xlabel('mean within class distance: %.3f, mean between class distance: %.3f' % (
        np.sum(np.identity(5)*mean_dist_class_stillcontrol)/5,
        np.sum((np.ones((5, 5))-np.identity(5))*mean_dist_class_stillcontrol)/20))
plt.show()


# %%
# 3. invariance from frames of a sequence for continuous morph
frame_rep_continmorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/**/frame_rep.pkl'))[0])

# %%
# get frame from one seq
train_spin_dataset = torch.load(os.path.join(filedir + dataDir, 'fashionMNISTtrain_set_spin.pt'))
one_seq, _ = train_spin_dataset[:9]

# %%
# compute distance
pair_dist_cosine_rep_contimorph = pairwise_distances(np.vstack(
    frame_rep_continmorph[frame_rep_continmorph['is_train'] == 1][frame_rep_continmorph['layer'] == 3]['r_out'][
    :9].to_numpy()), metric='cosine')

pair_dist_cosine_frame = pairwise_distances(torch.flatten(one_seq, start_dim=1), metric='cosine')

# %%
# plot rdm of frame and rep side by side
fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))
cbar_ax = fig.add_axes([.91, .3, .03, .4])
sns.heatmap(pair_dist_cosine_frame, xticklabels=1, yticklabels=1, ax=axs[0], cmap='viridis', vmin=0, vmax=.3,
            cbar_ax=cbar_ax)
axs[0].set_title('RDM frames one sequence')
sns.heatmap(pair_dist_cosine_rep_contimorph, xticklabels=1, yticklabels=1, ax=axs[1], cmap='viridis', vmin=0, vmax=.3,
            cbar_ax=cbar_ax)
axs[1].set_title('RDM reps one sequence contimorph')
# plt.tight_layout()
plt.show()

# %%
# plot tsne clustering of morph vs nomorph side by side
tsne_morph = plot_tsne(np.vstack(
    seq_rep_contimorph[seq_rep_contimorph['is_train'] == 1][seq_rep_contimorph['layer'] == 3]['r_out'].to_numpy()),
    seq_rep_contimorph[seq_rep_contimorph['is_train'] == 1][seq_rep_contimorph['layer'] == 3][
        'labels'].to_numpy(), 'tsne reps morph')
plt.show()

tsne_nomorph = plot_tsne(np.vstack(
    still_rep_continomorph[still_rep_continomorph['is_train'] == 1][still_rep_continomorph['layer'] == 3][
        'r_out'].to_numpy()),
    still_rep_continomorph[still_rep_continomorph['is_train'] == 1][still_rep_continomorph['layer'] == 3][
        'labels'].to_numpy(), 'tsne reps no morph')
plt.show()

# %%
#######################
# compare contimorph (big angle and small angle) vs discontimorph
#######################

frame_rep_discontinmorph = pd.read_pickle(glob(os.path.join(filedir, 'discontinuous_morph/**/frame_rep.pkl'))[0])
frame_rep_contimorph_sm = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph_sm/**/frame_rep.pkl'))[0])
frame_rep_contimorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/**/frame_rep.pkl'))[0])
# %%
df_genacc_conti1 = generate_acc_df([frame_rep_contimorph], [0],
                                  'conti_disconti', isGen=True)

df_genacc_conti2 = generate_acc_df([frame_rep_contimorph_sm], [1],
                                  'conti_disconti', isGen=True)

df_genacc_conti3 = generate_acc_df([frame_rep_discontinmorph], [2],
                                  'conti_disconti', isGen=True)

# %%
df_genacc_conti = pd.concat([df_genacc_conti1, df_genacc_conti2, df_genacc_conti3], ignore_index=True)

# %%
# 1. decoding acc
fig, ax = plt.subplots(figsize=(9, 5))
sns.despine()
sns.barplot(data=df_genacc_conti[df_genacc_conti['accIsTest'] == 1][df_genacc_conti['layer'] != 0], x='layer', y='acc', hue='conti_disconti', ax=ax,
            palette='tab10')
# ax.set_title('generalisation acc by layer continuous vs discontinuous')

# add baseline
# add baseline classification on input
plt.axhline(y=0.8894, linestyle='dotted', color='black', label='classification with frames')
plt.xlabel('area')
plt.ylabel('accuracy')
h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['classification with seq data', 'continuous (large rotation)', 'continuous (small rotation)',
              'discontinuous (large rotation)'], frameon=False, loc='center left',
          bbox_to_anchor=(1, 0.5), ncol=1, fontsize=12)
plt.tight_layout()

plt.show()
plt.close()

# %%
# 2. decoding acc delta from baseline
df_temp = df_genacc_conti.copy()
df_temp['acc'] = df_temp['acc'] - .8894

# %%
fig, ax = plt.subplots(figsize=(6, 5))
sns.despine()
sns.barplot(data=df_temp[df_temp['accIsTest'] == 1][df_temp['layer'] != 0], x='layer', y='acc', hue='conti_disconti', ax=ax,
            palette='tab10')
ax.set_title('generalisation acc by layer continuous vs discontinuous')

h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['continuous (large spin)', 'continuous (small spin)',
              'discontinuous (large spin)'], frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, -.1), ncol=2)
plt.tight_layout()

plt.tight_layout()
plt.show()
plt.close()

# %%
# rdm of contimorph vs discontimorph top layer
# fig1 = rdm_w_rep(np.vstack(
#     seq_rep_contimorph[seq_rep_contimorph['is_train'] == 1][seq_rep_contimorph['layer'] == 3]['r_out'].to_numpy()),
#     seq_rep_contimorph[seq_rep_contimorph['is_train'] == 1][seq_rep_contimorph['layer'] == 3][
#         'labels'].to_numpy(),
#     'cosine', istrain=True, ticklabel=640)
# plt.show()
reps = np.vstack(
    frame_rep_discontinmorph[frame_rep_discontinmorph['is_train'] == 1][frame_rep_discontinmorph['layer'] == 3][
        'r_out'].to_numpy())
labels = frame_rep_discontinmorph[frame_rep_discontinmorph['is_train'] == 1][frame_rep_discontinmorph['layer'] == 3][
        'labels'].to_numpy()

idx = np.argsort(labels)
reps = reps[idx]
pair_dist_cosine = pairwise_distances(reps[:9], metric='cosine')
# %%
# plot rdm of frame and rep side by side
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1, 1, 1.2]})
# cbar_ax = fig.add_axes([.91, .3, .03, .4])
# cbar_ax.set_in_layout(False)
sns.heatmap(pair_dist_cosine_frame, xticklabels=1, yticklabels=1, ax=axs[0], cmap='viridis', vmin=0, vmax=.8,
            cbar=False)
axs[0].set_title('frames')
axs[0].set_xlabel('mean cosine distance = %.3f' % (np.mean(pair_dist_cosine_frame)))
sns.heatmap(pair_dist_cosine_rep_contimorph, xticklabels=1, yticklabels=1, ax=axs[1], cmap='viridis', vmin=0, vmax=.8,
            cbar=False)
axs[1].set_title('latent representations \n (continuous morph)')
axs[1].set_xlabel('mean cosine distance = %.3f' % (np.mean(pair_dist_cosine_rep_contimorph)))
sns.heatmap(pair_dist_cosine, xticklabels=1, yticklabels=1, ax=axs[2], cmap='viridis', vmin=0, vmax=.8,
            cbar=True, cbar_kws={'label': 'cosine distance'})
axs[2].set_title('latent representations \n (discontinuous morph)')
axs[2].set_xlabel('mean cosine distance = %.3f' % (np.mean(pair_dist_cosine)))
plt.tight_layout()
plt.show()

# %%
# rdm by class

dist_discontimorph = cosine_dis(np.vstack(
    frame_rep_discontinmorph[frame_rep_discontinmorph['layer'] == 3][
        'r_out'].to_numpy()),
    frame_rep_discontinmorph[frame_rep_discontinmorph['layer'] == 3][
        'labels'].to_numpy(),
    'cosine')

mean_dist_class_discontimorph = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        mean_dist_class_discontimorph[i, j] = np.mean(dist_discontimorph[i*7200:(i+1)*7200, j*7200:(j+1)*7200])

sns.heatmap(mean_dist_class_discontimorph, annot=True, vmax=1.0, vmin=0)
plt.xlabel('mean within class distance: %.3f, mean between class distance: %.3f' % (
        np.sum(np.identity(5)*mean_dist_class_discontimorph)/5,
        np.sum((np.ones((5, 5))-np.identity(5))*mean_dist_class_discontimorph)/20))
plt.show()

# %%
#####
# additional analysis
########
# assessment of still rep acc from all networks
still_rep_discontimorph = pd.read_pickle(glob(os.path.join(filedir, 'discontinuous_morph/**/still_rep.pkl'))[0])
still_rep_contimorph = pd.read_pickle(glob(os.path.join(filedir, 'continuous_morph/**/still_rep.pkl'))[0])


df_genacc_stillcontivsdiconti = generate_acc_df([still_rep_stillcontrol, still_rep_contimorph,
                                                 still_rep_discontimorph, still_rep_continomorph],
                                                [0, 1, 2, 3], 'condition', True)

# %%
# plot
fig, ax = plt.subplots(figsize=(9, 5))
sns.despine()
sns.barplot(data=df_genacc_stillcontivsdiconti[df_genacc_stillcontivsdiconti['accIsTest'] == 1][df_genacc_stillcontivsdiconti['layer'] != 0], x='layer', y='acc', hue='condition', ax=ax,
            palette='tab10')
# ax.set_title('generalisation acc by layer still img')
plt.axhline(y=.455, linestyle='dashed', color='black', label='linear classification with still data')
plt.xlabel('area')
plt.ylabel('accuracy')

h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['classification with sampled \n fashion MNIST', 'still control', 'continuous morph', 'discontinuous morph', 'continuous no morph'],
          frameon=False, loc='center left',
          bbox_to_anchor=(1, 0.5), ncol=1, fontsize=12)

plt.tight_layout()
plt.show()
plt.close()

# %%
# examine if there are correlation between single neuron activity and spin angle
ex_seqrep_contimorph = np.vstack(frame_rep_continmorph[frame_rep_continmorph['is_train']==0][frame_rep_continmorph['layer']==3]['r_out'][9:18].to_numpy())

# %%
fig, axs = plt.subplots(10, 10, sharex=True, sharey=True)
x = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
for i, ax in enumerate(fig.axes):
    ax.plot(x, ex_seqrep_contimorph[:, i])
plt.show()

# %%
# plot gen acc and dataset size
corr_train_log = glob(os.path.join(filedir, 'correlation/**/trained_model/*metrics_log.pkl'))

# %%
condi = []
acc_corr = []
epoch_log = []
for i in range(len(corr_train_log)):
    condition = corr_train_log[i].split('/')[-3]  # get condition name as file
    condition = int(condition.split('_')[0])
    log = pd.read_pickle(corr_train_log[i])
    acc_corr.append(log['genAccStillImg'])
    condi.append(np.full(5, condition))
    epoch_log.append([1, 10, 20, 30, 40])

# %%
df_corr_gen = pd.DataFrame()
df_corr_gen['samples per class'] = np.array(condi).flatten()
df_corr_gen['acc'] = np.array(acc_corr).flatten()
df_corr_gen['epoch'] = np.array(epoch_log).flatten()

# %%
# plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.despine()
sns.lineplot(data=df_corr_gen, x='epoch', y='acc', hue='samples per class', palette='tab10')
h, _ = ax.get_legend_handles_labels()
ax.legend(h, ['20', '40', '100', '200', '500'], frameon=False, title='samples per class')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 5))
sns.despine()
sns.barplot(data=df_corr_gen[df_corr_gen['epoch']==40], x='samples per class', y='acc',
            palette='tab10', ci='sd')
plt.show()

# %%
# reconstruction
train_data = torch.load('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_10/trainesize160perclass/fashionMNISTtrain_image.pt')
img, label = train_data[0]

# %%
from fc_net import *
CONFIG_PATH = "../scripts/"

# %%
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

trained_still_network = FcDHPC([1936, 2500, 800, 100], [.07, .07, .05, .05], lr=.001, act_func=config['act_func'], device='cpu', dtype=torch.float)
trained_still_network.load_state_dict(torch.load('/Users/lucyzhang/Documents/research/PC_net/results/morph_test_10/still_control/trained_model/spin[1936, 2500, 800, 100][0.07, 0.07, 0.05, 0.05]Falsel2_0.0110.0end_trainingreadout.pth', map_location=torch.device('cpu')))

# %%
_, fig = trained_still_network.reconstruct(img, torch.tensor(label), 250)
plt.show()

# %%
sns.despine()
sns.heatmap(torch.squeeze(img), cmap='viridis', cbar_kws={'label': 'pixel value'})
plt.show()