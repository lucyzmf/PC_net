# %%
import pickle

import wandb
from torch import nn

wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')

# %%
"""
pytorch implementation of PC net
this branch tests the effects of architectural components on generalisation 
"""
from fc_net import FcDHPC
from hippo_net import FcDHPC_hippo
import torch.cuda as cuda
from evaluation import *
import yaml
import os
import numpy as np
import torch.profiler
import datetime
from torch.utils import data

now = datetime.datetime.now()

# load config
CONFIG_PATH = "../scripts/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")

if __name__ == '__main__':
    # seed the CPUs and GPUs
    torch.manual_seed(0)

    if torch.cuda.is_available():  # Use GPU if possible
        dev = "cuda:0"
        print("Cuda is available")
        cuda.manual_seed_all(0)

    else:
        dev = "cpu"
        print("Cuda not available")
    device = torch.device(dev)

    dtype = torch.float  # Set standard datatype

    # get config values
    dataset = config['dataset_type']
    pin_mem = config['pin_mem']
    batchSize = config['batch_size']
    n_workers = config['num_workers']
    inference_steps = config['infsteps']  # num infsteps per image
    epochs = config['epochs']  # total training epochs
    infrates = config['infrates']  # inf rates each layer
    lr = config['learning_rate']  # lr for weight updates
    arch = config['network_size']  # size of each layer
    per_seq_repeat = config['per_seq_repeat']  # num of repeats per image/sequence
    arch_type = config['architecture']
    morph_type = config['morph_type']
    frame_per_seq = config['frame_per_sequence']
    padding = config['padding_size']
    seq_train = config['seq_train']
    reg_strength = config['reg_strength']
    reg_type = config['reg_type']
    infstep_before_update = config['infstep_before_update']
    act_normalise = config['act_norm']
    norm_constant = config['norm_constant']
    reset_per_frame = config['reset_per_frame']

    # set up test name
    test_name = 'morph_test_10' + str(reset_per_frame) + '_seqtrain' + str(seq_train)
    print(test_name)


    # load datasets
    def stilldataset_to_dataloader(still_tensordataset):
        idx = still_tensordataset.indices
        img = still_tensordataset.dataset.data[idx]
        img = nn.functional.pad(img, (padding, padding, padding, padding))
        label = still_tensordataset.dataset.targets[idx]
        dataset = data.TensorDataset(img, label)
        loader = data.DataLoader(dataset, batch_size=config['batch_size'],
                                 num_workers=config['num_workers'], pin_memory=config['pin_mem'], shuffle=True)

        return loader


    # load train still data
    train_set_still = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtrain_image.pt'))
    train_still_img_loader = stilldataset_to_dataloader(train_set_still)
    # load test still images
    test_set_still = torch.load(os.path.join(config['dataset_dir'], 'fashionMNISTtest_image.pt'))
    test_still_img_loader = stilldataset_to_dataloader(test_set_still)

    # load train seq data
    if seq_train:
        train_set = torch.load(
            os.path.join(config['dataset_dir'], str(dataset) + 'train_set_' + str(morph_type) + '.pt'))
        # load test data
        test_set = torch.load(
            os.path.join(config['dataset_dir'], str(dataset) + 'test_set_' + str(morph_type) + '.pt'))
        if reset_per_frame:  # if reset per frame, shuffle spin dataset
            train_loader = data.DataLoader(train_set, batch_size=batchSize, num_workers=n_workers,
                                           pin_memory=pin_mem, shuffle=True)
            test_loader = data.DataLoader(test_set, batch_size=batchSize, num_workers=n_workers,
                                          pin_memory=pin_mem, shuffle=True)
        else:
            train_loader = data.DataLoader(train_set, batch_size=batchSize, num_workers=n_workers,
                                           pin_memory=pin_mem, shuffle=False)
            test_loader = data.DataLoader(test_set, batch_size=batchSize, num_workers=n_workers,
                                          pin_memory=pin_mem, shuffle=False)
    else:
        train_loader = train_still_img_loader

    # %%
    ###########################
    ### Training loop
    ###########################
    with torch.no_grad():  # turn off auto grad function
        # meta data on MNIST dataset
        numClass = config['num_classes']
        dataWidth = 28 + 2 * config['padding_size']

        # Hyperparameters for training logged with wandb
        wandb.init(project="DHPC_morph_test_10", entity="lucyzmf")  # , mode='disabled')

        wbconfig = wandb.config
        wbconfig.infstep = inference_steps
        wbconfig.epochs = epochs
        wbconfig.infrate = infrates
        wbconfig.network_size = arch
        wbconfig.architecture = arch_type
        wbconfig.lr = lr
        wbconfig.frame_per_seq = frame_per_seq
        wbconfig.morph = morph_type
        wbconfig.per_seq_repeat = per_seq_repeat
        wbconfig.batch_size = config['batch_size']
        wbconfig.train_size = config['train_size']
        wbconfig.test_size = config['test_size']
        wbconfig.act_func = config['act_func']
        wbconfig.seq_train = seq_train
        wbconfig.reg_strength = reg_strength
        wbconfig.reg_type = reg_type
        wbconfig.update_per_frame = inference_steps / infstep_before_update
        wbconfig.act_normalise = act_normalise
        wbconfig.norm_constant = norm_constant
        wbconfig.reset_per_frame = reset_per_frame

        #  network instantiation
        if arch_type == 'FcDHPC':
            net = FcDHPC(arch, infrates, lr=lr, act_func=config['act_func'], device=device, dtype=dtype)
        elif arch_type == 'FcDHPC_hippo':
            rf_sizes = config['rf_sizes']
            wbconfig.rf_sizes = rf_sizes
            net = FcDHPC_hippo(arch, infrates, lr=lr, act_func=config['act_func'], device=device, dtype=dtype)
        else:
            raise TypeError('network architecture not specified')
        net.to(device)
        wandb.watch(net)
        print('network instantiated')

        # %%
        # values logged during training
        total_errors = []
        total_errors_test = []
        last_layer_act_log = []
        clustering_acc = []
        test_acc_history_reg = []  # acc on test set at the end of each epoch
        # test_acc_history_knn = []
        best_gen_acc = 0

        # sample image and label to test reconstruction
        for test_images, test_labels in test_still_img_loader:
            sample_image = test_images[0]  # Reshape them according to your needs.
            sample_label = test_labels[0]

        print('start training')
        # prepare profiler
        profile_dir = "../results/" + test_name + str(now) + '/trace/'
        trained_model_dir = "../results/" + test_name + str(now) + '/trained_model/'
        os.makedirs(trained_model_dir)
        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(
        #             wait=1,
        #             warmup=1,
        #             active=2),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        #         record_shapes=True,
        #         profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        #         with_stack=True
        # ) as p:
        #     _image, target = train_loader.dataset[0]
        #     net.init_states()
        #     net(_image, inference_steps)
        #     net.learn()
        #     # profiler step boundary
        #     p.step()

        # finish profiling of one image, start training
        for epoch in range(epochs):
            net.init_states()
            errors = []  # log total error per sample in dataset
            last_layer_act = []  # log avg act of last layer neurons per sample
            rep_train, label_train = [], []  # log high level representations at the end of each sequence to test classification
            seq_rep_test, seq_label_test = [], []

            for i, (image, label) in enumerate(train_loader):
                net(image, inference_steps, istrain=True)
                net.learn()
                errors.append(net.total_error())

                if (i + 1) % 360 == 0:  # log at the end of 40 sequences
                    last_layer_act.append(torch.mean(net.states['r_output'][-1].detach().cpu()))
                    wandb.log({
                        # 'last layer activation distribution': wandb.Histogram(
                        #     net.states['r_activation'][-1].detach().cpu()),
                        'last layer output distribution': wandb.Histogram(
                            net.states['r_output'][-1].detach().cpu()),
                        'layer n-1 weights': wandb.Histogram(net.layers[-2].weights.detach().cpu()),
                        'layer n-1 output distribution': wandb.Histogram(net.states['r_output'][-2].detach().cpu()),
                        'layer n-1 error activation': wandb.Histogram(net.states['error'][-2].detach().cpu()),
                        # 'layer 0 error activation': wandb.Histogram(net.states['error'][0].detach().cpu())
                    })

                if (seq_train and not reset_per_frame) and (
                        (i + 1) % frame_per_seq == 0):  # if trained on sequences and reps taken at the end of seq
                    if (epoch % 10 == 0) or (epoch == epochs - 1):
                        rep_train.append(net.states['r_output'][-1].detach().cpu().numpy())
                        label_train.append(label)
                    net.init_states()

                if not seq_train or (
                        seq_train and reset_per_frame):  # if trained on still images or seq train but reset per frame
                    if (epoch % 10 == 0) or (epoch == epochs - 1):
                        rep_train.append(net.states['r_output'][-1].detach().cpu().numpy())
                        label_train.append(label)
                    net.init_states()

            # summary data
            total_errors.append(np.mean(errors))  # mean error per epoch
            last_layer_act_log.append(np.mean(last_layer_act))  # mean last layer activation per epoch

            wandb.log({
                'epoch': epoch,
                'train_error': total_errors[-1],
                'avg last layer act': last_layer_act_log[-1]
            })

            if (epoch % 10 == 0) or (epoch == epochs - 1):  # evaluation every 10 epochs
                # organise reps logged during training
                train_reps_dataset = data.TensorDataset(torch.tensor(rep_train), torch.tensor(label_train))
                torch.save(train_reps_dataset, trained_model_dir + 'epoch' + str(epoch) + 'train_rep.pt')
                rep_train = np.vstack(rep_train)
                label_train = torch.concat(label_train).numpy()

                # generate rdm with train reps
                fig_train = rdm_w_rep_title(rep_train, label_train, 'cosine',
                                            'RDM of training sequence representations')
                wandb.log({'rdm train data': wandb.Image(fig_train)})

                # get error on single frames
                errors_test = []
                rep_still_test = []  # log rep generated from still images for classification testing
                rep_still_labels = []
                net.init_states()

                for i, (_image, _label) in enumerate(
                        test_still_img_loader):  # iterate through still image loader to generate reps
                    net.init_states()
                    net(_image, inference_steps, istrain=False)
                    errors_test.append(net.total_error())
                    rep_still_test.append(net.states['r_output'][-1].detach().cpu().numpy())
                    rep_still_labels.append(_label)

                # log errors
                total_errors_test.append(np.mean(errors_test))
                print('epoch: %i, total error on train set: %.4f, avg last layer activation: %.4f' %
                      (epoch, total_errors[-1], last_layer_act_log[-1]))
                print('total error on test set: %.4f' % (total_errors_test[-1]))
                wandb.log({
                    'epoch': epoch,
                    'test_error on still images': total_errors_test[-1]
                })

                # organise arrays logging reps for test still imgs
                still_reps_dataset = data.TensorDataset(torch.tensor(rep_still_test), torch.tensor(rep_still_labels))
                torch.save(still_reps_dataset, trained_model_dir + 'epoch' + str(epoch) + 'test_still_reps.pt')
                rep_still_test = np.vstack(rep_still_test)  # representations
                rep_still_labels = torch.concat(rep_still_labels).numpy()  # labels

                # generate rdm with still test reps
                fig_test_still = rdm_w_rep_title(rep_still_test, rep_still_labels, 'cosine',
                                                 'RDM of still test image representations')
                wandb.log({'rdm test still data': wandb.Image(fig_test_still)})

                # assess clustering
                within_sample_acc = within_sample_classification_stratified(rep_train, label_train)
                clustering_acc.append(within_sample_acc)
                print('clustering: linear regression on high level reps from train set (stratified kfold) %.4f'
                      % within_sample_acc)
                wandb.log({
                    'epoch': epoch,
                    'clustering: linear regression on high level reps from train set (stratified kfold)': within_sample_acc
                })

                # assess generalisation that applies to both conditions
                # classification acc on still train reps and test still images
                _, acc_still_test_reg = linear_regression(rep_train, label_train, rep_still_test, rep_still_labels)
                print('generalisation: linear regression on still test images: %.4f' % acc_still_test_reg)

                # _, acc_still_test_knn = knn_classifier(rep_train, label_train, rep_still_test, rep_still_labels)
                # print('generalisation: knn on still test images: %.4f' % acc_still_test_knn)
                wandb.log({
                    'epoch': epoch,
                    'generalisation to still img (linear regression)': acc_still_test_reg
                })

                if seq_train:
                    net.init_states()
                    for i, (_image, _label) in enumerate(
                            test_loader):  # generate high level rep using spin seq test dataset
                        net(_image, inference_steps, istrain=False)
                        if ((i + 1) % frame_per_seq == 0) and (not reset_per_frame):  # at the end of each sequence
                            seq_rep_test.append(
                                net.states['r_output'][-1].detach().cpu().numpy())  # rep recorded at end of each seq
                            seq_label_test.append(_label)
                            net.init_states()
                        else:
                            seq_rep_test.append(
                                net.states['r_output'][-1].detach().cpu().numpy())  # rep recorded at end of each seq
                            seq_label_test.append(_label)
                            net.init_states()
                    # convert arrays
                    seq_reps_dataset = data.TensorDataset(torch.tensor(seq_rep_test), torch.tensor(seq_label_test))
                    torch.save(seq_reps_dataset, trained_model_dir + 'epoch' + str(epoch) + 'seq_rep_test.pt')
                    seq_rep_test = np.vstack(seq_rep_test)
                    seq_label_test = torch.concat(seq_label_test).numpy()

                    # generate rdm with train reps
                    fig_test = rdm_w_rep_title(seq_rep_test, seq_label_test, 'cosine',
                                               'RDM of test sequence representations')
                    wandb.log({'rdm test seq': wandb.Image(fig_test)})

                    # assess generalisation using all seq reps , use two types of classifiers
                    # linear regression
                    acc_train, acc_test = linear_regression(rep_train, label_train, seq_rep_test, seq_label_test)
                    test_acc_history_reg.append(acc_test)
                    print('epoch: %i: generalisation, seq rep to seq rep linear regression test acc %.4f' % (
                        epoch, acc_test))
                    if acc_test > best_gen_acc:  # save best gen acc model
                        best_gen_acc = acc_test
                        torch.save(net.state_dict(),
                                   trained_model_dir + str(config['morph_type']) + str(net.architecture) +
                                   str(net.inf_rates) + str(seq_train) + str(reg_type) + '_' + str(reg_strength) +
                                   str(wbconfig.update_per_frame) + 'updates_per_frame_acc' + str(
                                       best_gen_acc) + 'readout.pth')

                    # knn classification on train and test sequence
                    # _, cum_acc_knn = knn_classifier(rep_train, label_train, seq_rep_test, seq_label_test)
                    # test_acc_history_knn.append(cum_acc_knn)
                    # print('epoch: %i: generalisation, seq rep to seq rep knn test acc %.4f'
                    #       % (epoch, cum_acc_knn))

                    wandb.log({
                        'epoch': epoch,
                        'generalisation, seq rep to seq rep linear regression': acc_test
                    })

                # sample reconstruction
                recon_error, fig = net.reconstruct(sample_image, sample_label, inference_steps)
                wandb.log({
                    'epoch': epoch,
                    'reconstructed image': wandb.Image(fig)
                })

                # save trained models
                if epoch == epochs - 1:
                    torch.save(net.state_dict(),
                               trained_model_dir + str(config['morph_type']) + str(net.architecture) +
                               str(net.inf_rates) + str(seq_train) + str(reg_type) + '_' + str(reg_strength) +
                               str(wbconfig.update_per_frame) + 'end_training' + 'readout.pth')
                    print('end training')

        # %%
        # _, _, fig_train = generate_rdm(net, train_loader, inference_steps * 5)
        # wandb.log({'rdm train data': wandb.Image(fig_train)})
        # if seq_train:
        #     _, _, fig_test = generate_rdm(net, test_loader, inference_steps * 5)
        #     wandb.log({'rdm seq test data': wandb.Image(fig_test)})
        # _, _, fig_test = generate_rdm(net, test_still_img_loader, inference_steps * 5)
        # wandb.log({'rdm still image test data': wandb.Image(fig_test)})

        # %%
        # # inspect convergence of last layer
        # inf_step = np.arange(0, 5000)
        # high_layer_output = []
        # mid_layer_output = []
        # input_layer_output = []
        # error_intput = []
        # error_mid = []
        #
        # net.init_states()
        # for i in inf_step:
        #     net(sample_image, 1, istrain=False)
        #     high_layer_output.append(net.states['r_output'][-1].detach().cpu().numpy())
        #     mid_layer_output.append(net.states['r_output'][-2].detach().cpu().numpy())
        #     input_layer_output.append(net.states['r_output'][0].detach().cpu().numpy())
        #
        #     error_intput.append(net.states['error'][0].detach().cpu().numpy())
        #     error_mid.append(net.states['error'][-2].detach().cpu().numpy())
        #
        # # %%
        # fig, axs = plt.subplots(2, 3, figsize=(24, 10))
        # w_0_1 = net.layers[0].weights.cpu()
        # bu_error_to_hidden = np.matmul(torch.transpose(w_0_1, 0, 1).numpy(), np.transpose(np.vstack(error_intput)))
        #
        # axs[0][0].plot(inf_step, (np.transpose(bu_error_to_hidden) - error_mid))
        # axs[0][0].set_title('amount of update to hidden layer')
        # axs[0][1].plot(inf_step, mid_layer_output)
        # axs[0][1].set_title('hidden layer output')
        # axs[0][2].plot(inf_step, high_layer_output)
        # axs[0][2].set_title('highest layer output')
        #
        # mse_input = np.mean(np.vstack(error_intput), axis=1) ** 2
        # mse_mid = np.mean(np.vstack(error_mid), axis=1) ** 2
        #
        # axs[1][0].plot(inf_step, mse_input)
        # axs[1][0].set_title('input layer MSE')
        # axs[1][1].plot(inf_step, mse_mid)
        # axs[1][1].set_title('hidden layer MSE')
        #
        # # plot bu_error - e_act to layer 2
        # w_1_2 = net.layers[1].weights.cpu()
        # bu_error_hidden_to_last = np.matmul(torch.transpose(w_1_2, 0, 1).numpy(), np.transpose(np.vstack(error_mid)))
        # axs[1][2].plot(inf_step, (np.transpose(bu_error_hidden_to_last)))
        # axs[1][2].set_title('amount of update to last layer')
        #
        # plt.show()
        # plt.savefig(os.path.join(trained_model_dir, 'convergence.png'))

        # %%
        # save important datapoints for later plotting
        log = {'total_error': total_errors}

        fig, axs = plt.subplots()
        print('total errors', total_errors)
        plt.plot(np.arange(len(total_errors)), total_errors)
        plt.title('total MSE in network during training')
        plt.savefig(os.path.join(trained_model_dir, 'total_error.png'))

        # %%
        log['clustering acc'] = clustering_acc

        print('clustering acc', clustering_acc)
        fig, axs = plt.subplots()
        plt.plot(np.arange(len(clustering_acc)), clustering_acc)
        plt.title('clustering acc during training')
        plt.savefig(os.path.join(trained_model_dir, 'cluster_acc.png'))

        # %%
        # save gen acc log
        log['genAccReg'] = test_acc_history_reg
        # log['genAccKnn'] = test_acc_history_knn

        # %%
        with open(trained_model_dir + test_name + 'training_metrics_log.pkl', 'wb') as f:
            pickle.dump(log, f)

        # %%
        # generate and save representations for later evaluation
        # need three dataframes in total: seq representations, frame representations, still representations
        # each contain representations generated from corresponding data
        net.init_states()
        if seq_train:
            dataloaders = [train_loader, test_loader]
            dataloaders_still = [train_still_img_loader, test_still_img_loader]
            seq_reps = generate_reps(net, dataloaders, inference_steps, resetPerFrame=False)
            frame_reps = generate_reps(net, dataloaders, inference_steps, resetPerFrame=True)
            still_reps = generate_reps(net, dataloaders_still, inference_steps, resetPerFrame=True)
            seq_reps.to_pickle(os.path.join(trained_model_dir + 'seq_rep.pkl'))
            frame_reps.to_pickle(os.path.join(trained_model_dir + 'frame_rep.pkl'))
            still_reps.to_pickle(os.path.join(trained_model_dir + 'still_rep.pkl'))
        else:
            dataloaders = [train_loader, test_still_img_loader]
            still_reps = generate_reps(net, dataloaders, inference_steps, resetPerFrame=True)
            still_reps.to_pickle(os.path.join(trained_model_dir + 'still_rep.pkl'))
