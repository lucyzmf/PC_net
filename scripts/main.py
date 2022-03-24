# %%
import wandb

wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')

# %%
"""
pytorch implementation of PC net
this branch tests the effects of architectural components on generalisation 
"""
from fc_net import FcDHPC
import torch.cuda as cuda
from evaluation import *
import yaml
import os
import numpy as np
import torch.profiler
import datetime
from rf_net_cm import RfDHPC_cm

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

    # git config values
    dataset = config['dataset_type']
    inference_steps = config['infsteps']  # num infsteps per image
    epochs = config['epochs']  # total training epochs
    infrates = config['infrates']  # inf rates each layer
    lr = config['learning_rate']  # lr for weight updates
    arch = config['network_size']  # size of each layer
    per_seq_repeat = config['per_seq_repeat']  # num of repeats per image/sequence
    arch_type = config['architecture']
    morph_type = config['morph_type']
    frame_per_seq = config['frame_per_sequence']

    # load data
    train_loader = torch.load(
        os.path.join(config['dataset_dir'], str(dataset) + 'train_loader_' + str(morph_type) + '.pth'))
    test_loader = torch.load(
        os.path.join(config['dataset_dir'], str(dataset) + 'test_loader_' + str(morph_type) + '.pth'))

    # %%
    ###########################
    ### Training loop
    ###########################
    with torch.no_grad():  # turn off auto grad function
        # meta data on MNIST dataset
        numClass = 10
        dataWidth = 28 + 2*config['padding_size']

        # Hyperparameters for training logged with wandb
        wandb.init(project="DHPC_morph", entity="lucyzmf")  # , mode='disabled')

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

        #  network instantiation
        if arch_type == 'FcDHPC':
            net = FcDHPC(arch, infrates, lr=lr, act_func=config['act_func'], device=device, dtype=dtype)
        elif arch_type == 'RfDHPC_cm':
            rf_sizes = config['rf_sizes']
            wbconfig.rf_sizes = rf_sizes
            net = RfDHPC_cm(arch, rf_sizes, infrates, lr=lr, act_func=config['act_func'], device=device, dtype=dtype)
        else:
            raise TypeError('network architecture not specified')
        net.to(device)
        wandb.watch(net)
        print('network instantiated')

        # building memory storage
        mem = torch.rand([numClass, arch[-1]]).to(device)

        # %%
        # values logged during training
        total_errors = []
        total_errors_test = []
        last_layer_act_log = []
        train_acc_history = []
        test_acc_history = []  # acc on test set at the end of each epoch

        # sample image and label to test reconstruction
        for test_images, test_labels in test_loader:
            sample_image = test_images[0]  # Reshape them according to your needs.
            sample_label = test_labels[0]

        print('start training')
        # prepare profiler
        profile_dir = "../results/" + str(now) + '/trace/'
        trained_model_dir = "../results/" + str(now) + '/trained_model/'
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True
        ) as p:
            data, target = train_loader.dataset[0]
            net.init_states()
            net(data, inference_steps)
            net.learn()
            # profiler step boundary
            p.step()

        # finish profiling of one image, start training
        for epoch in range(epochs):
            errors = []  # log total error per sample in dataset
            last_layer_act = []  # log avg act of last layer neurons per sample

            for i, (image, label) in enumerate(train_loader):
                if i % frame_per_seq == 0:
                    net.init_states()
                for j in range(per_seq_repeat):
                    net(image, inference_steps)
                net.learn()
                errors.append(net.total_error())

                if i % 50 == 0:  # log every 50 steps
                    last_layer_act.append(torch.mean(net.states['r_activation'][-1].detach().cpu()))
                    wandb.log({
                        'last layer activation distribution': wandb.Histogram(
                            net.states['r_activation'][-1].detach().cpu()),
                        'last layer output distribution': wandb.Histogram(
                            net.states['r_output'][-1].detach().cpu()),
                        'layer n-1 weights': wandb.Histogram(net.layers[-2].weights.detach().cpu()),
                        'layer n-1 output distribution': wandb.Histogram(net.states['r_output'][-2].detach().cpu())
                    })

            # summary data
            total_errors.append(np.mean(errors))  # mean error per epoch
            last_layer_act_log.append(np.mean(last_layer_act))  # mean last layer activation per epoch

            wandb.log({
                'epoch': epoch,
                'train_error': total_errors[-1],
                'avg last layer act': last_layer_act_log[-1]
            })

            if epoch == epochs - 1:
                print('end training, saving trained model')
                os.mkdir(trained_model_dir)
                torch.save(net.state_dict(), trained_model_dir + str(arch_type) + str(net.architecture) + str(
                    net.inf_rates) + 'readout.pth')

            if (epoch % 10 == 0) and (epoch != 0):
                # test classification
                errors_test = []
                for i, (image, label) in enumerate(test_loader):
                    net.init_states()
                    net(image, inference_steps)
                    errors_test.append(net.total_error())
                total_errors_test.append(np.mean(errors_test))
                print('epoch: %i, total error on train set: %.4f, avg last layer activation: %.4f' %
                      (epoch, total_errors[-1], last_layer_act_log[-1]))
                print('total error on test set: %.4f' % (total_errors_test[-1]))
                # reg_acc_train = test_accuracy(net, train_loader)
                # reg_acc_test = test_accuracy(net, test_loader)

                wandb.log({
                    'test_error': total_errors_test[-1],
                    # 'reg acc on train set': reg_acc_train,
                    # 'reg acc on test set': reg_acc_test
                })

                # sample reconstruction
                recon_error, fig = net.reconstruct(sample_image, sample_label, 100)
                wandb.log({'reconstructed image': wandb.Image(fig)})

            # if (epoch == 0) or (epoch == epochs/2) or (epoch == epochs - 1):
            #     # train classifier using training data
            #     train_acc = train_classifier(net, classifier, train_loader, 1)
            #     print(train_acc)
            #
            #     # test classification acc at the end of each epoch
            #     test_acc = test_classifier(net, classifier, test_loader)  # test classifier on test set (unseen data)
            #     train_acc_history.append(train_acc)
            #     test_acc_history.append(test_acc)
            #
            #     print('epoch: ', epoch, '. classifier training acc: ', train_acc, '. classifier test acc: ', test_acc)
            #     wandb.log({
            #         'classifier train acc': train_acc,
            #         'classifier test acc': test_acc
            #     })

        # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # axs[0].plot(total_errors)
        # axs[0].set_title('Total Errors on training set')
        # axs[1].plot(total_errors_test)
        # axs[1].set_title('Total Errors on test set')
        # plt.tight_layout()
        # plt.show()

        # %%
        _, _, fig_train = generate_rdm(net, train_loader, inference_steps * 5)
        wandb.log({'rdm train data': wandb.Image(fig_train)})
        _, _, fig_test = generate_rdm(net, test_loader, inference_steps * 5)
        wandb.log({'rdm test data': wandb.Image(fig_test)})
