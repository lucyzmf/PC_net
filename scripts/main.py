# %%
import numpy as np
import torch
import wandb
import os

wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')
os.environ["WANDB_MODE"] = "offline"

# %%
"""pytorch implementation of deep hebbian predictive coding(DHPC) net that enables relatively flexible maniputation 
of network architecture code inspired by Matthias's code of PCtorch 

this version implements new learning paradigm. an array stores representations formed from the output of highest layer 
after seeing sample from that category 
"""
import torchvision
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from network import DHPC, sigmoid
import torch.cuda as cuda
from evaluation import *
import torch.profiler
import glob

if __name__ == '__main__':
    # seed the CPUs and GPUs
    torch.manual_seed(0)

    if torch.cuda.is_available():  # Use GPU if possible
        dev = "cuda:0"
        print("Cuda is available")
        cuda.manual_seed_all(0)
        n_workers = 2
        batchSize = 1
        machine = 'LISA cluster'

    else:
        dev = "cpu"
        print("Cuda not available")
        n_workers = 0
        batchSize = 1
        machine = 'cpu'

    device = torch.device(dev)

    dtype = torch.float  # Set standard datatype

    #  data preprocessing
    full_mnist = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # %%

    # genearte train test files for digit classification
    indices = np.arange(len(full_mnist))
    train_indices, test_indices = train_test_split(indices, train_size=50 * 10, test_size=10 * 10,
                                                   stratify=full_mnist.targets)

    # Warp into Subsets and DataLoaders
    train_dataset = Subset(full_mnist, train_indices)
    test_dataset = Subset(full_mnist, test_indices)

    dataWidth = train_dataset[0][0].shape[1]
    numClass = 10  # number of classes in mnist
    # %%

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=n_workers, pin_memory=True, batch_size=batchSize)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=n_workers, pin_memory=True, batch_size=batchSize)

    torch.save(train_loader, 'data/train_loader.pth')
    torch.save(test_loader, 'data/test_loader.pth')

    # %%
    ###########################
    ### Training loop
    ###########################

    with torch.no_grad():  # turn off auto grad function
        wandb.init(project="DHPC", entity="lucyzmf")

        config = wandb.config
        config.infstep = 100
        config.epoch = 1
        config.infrate = [.1, .07, .05]
        config.lr = .05
        config.arch = [dataWidth ** 2, 1000, 50]
        config.batchSize = batchSize
        config.num_workers = n_workers
        config.machine = machine

        # Hyperparameters for training
        inference_steps = config.infstep
        epochs = config.epoch

        #  network instantiation
        network_architecture = config.arch
        inf_rates = config.infrate
        per_im_repeat = 1

        net = DHPC(network_architecture, inf_rates, lr=config.lr, act_func=sigmoid, device=device, dtype=dtype)
        net.to(device)
        wandb.watch(net)
        print('network instantiated')

        # building memory storage
        mem = torch.rand([numClass, network_architecture[-1]]).to(device)

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
        profile_dir = "trace"
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
            for epoch in range(epochs):

                errors = []  # log total error per sample in dataset
                last_layer_act = []  # log avg act of last layer neurons per sample

                for i, (image, label) in enumerate(train_loader):
                    net.init_states()
                    cat_mem = mem[label.item()]  # retrieve category rep
                    for j in range(per_im_repeat):
                        cat_mem = net(torch.flatten(image), inference_steps, cat_mem)  # returns output of highest layer
                    net.learn()
                    mem[label.item()] = cat_mem  # update cat mem with new highest layer output
                    # print(cat_mem, mem[label.item()])
                    errors.append(net.total_error())
                    last_layer_act.append(torch.mean(net.states['r_activation'][-1].detach().cpu()))
                    wandb.log({
                        'last layer activation distribution': wandb.Histogram(
                            net.states['r_activation'][-1].detach().cpu()),
                        'last layer output distribution': wandb.Histogram(net.states['r_output'][-1].detach().cpu()),
                        'layer n-1 weights': wandb.Histogram(net.layers[-2].weights.detach().cpu()),
                        'layer n-1 output distribution': wandb.Histogram(net.states['r_output'][-2].detach().cpu())
                    })

                    # log mem storage as wandb table
                    my_table = wandb.Table(columns=np.arange(net.architecture[-1]).tolist(),
                                           data=mem.detach().cpu().numpy())
                    wandb.log({'catemory mem': my_table})

                    # profiler step boundary
                    p.step()

                # summary data
                total_errors.append(np.mean(errors))  # mean error per epoch
                last_layer_act_log.append(np.mean(last_layer_act))  # mean last layer activation per epoch

                wandb.log({
                    'epoch': epoch,
                    'train_error': total_errors[-1],
                    'avg last layer act': last_layer_act_log[-1]
                })

                if (epoch % 10 == 0) and (epoch != 0):
                    # test classification
                    errors_test = []
                    for i, (image, label) in enumerate(test_loader):
                        net.init_states()
                        net(torch.flatten(image), inference_steps)
                        errors_test.append(net.total_error())
                    total_errors_test.append(np.mean(errors_test))
                    print('epoch: %i, total error on train set: %.4f, avg last layer activation: %.4f' % (
                    epoch, total_errors[-1],
                    last_layer_act_log[-1]), )
                    print('total error on test set: %.4f' % (total_errors_test[-1]))
                    reg_acc_train = test_accuracy(net, train_loader)
                    reg_acc_test = test_accuracy(net, test_loader)

                    wandb.log({
                        'test_error': total_errors_test[-1],
                        'reg acc on train set': reg_acc_train,
                        'reg acc on test set': reg_acc_test
                    })

                    # sample reconstruction
                    recon_error, fig = net.reconstruct(sample_image, sample_label, 10)
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

                if epoch == epochs - 1:
                    print('end training, saving trained model')
                    torch.save(net.state_dict(), str(net.architecture) + str(net.inf_rates) + 'readout.pth')
                # profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
                # # add the pt.trace.json files to the Artifact
                # profile_art.add_file(glob.glob(profile_dir + "/*.pt.trace.json")[0], "trace.pt.trace.json")
                # # log the artifact
                # wandb.log_artifact(profile_art)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(total_errors)
        axs[0].set_title('Total Errors on training set')
        axs[1].plot(total_errors_test)
        axs[1].set_title('Total Errors on test set')
        plt.tight_layout()
        plt.show()

        # %%
        _, _, fig_train = generate_rdm(net, train_loader, 10, mem)
        wandb.log({'rdm train data': wandb.Image(fig_train)})
        _, _, fig_test = generate_rdm(net, test_loader, 10, mem)
        wandb.log({'rdm test data': wandb.Image(fig_test)})
