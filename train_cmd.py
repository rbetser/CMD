############################################# IMPORTS #######################################################
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import argparse
import torch
from sklearn.decomposition import PCA

# utils
from utils.base_utils import *
from utils.cmd_utils import basic_cmd, sample_represented_weights, create_AB_DF, update_AB_vec
from utils.pbfgs_utils import train_pbfgs, get_model_param_vec


############################################## MAIN #########################################################

def main(args):

    # Initialization of device and loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    # Loading CIFAR10 dataset (for ViT-b-16 different image size)
    if args.model == 'vit':
        trainloader, testloader = get_cifar10_loaders_vit(args.batch_size)

    else:
        trainloader, testloader = get_cifar10_loaders(args.batch_size)
        
    # Results Directory
    results_dir = args.results_dir
    dir_name = results_dir + '/' + args.model
    
    dir_path = './' + dir_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_name)

    # Initialization of model and optimizer
    net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp = get_model(args.model, device)

    # 'Theta' is a list of model checkpoints
    Theta_path = dir_name + '/Theta.pt'

    # Perform training or load weights
    if not os.path.exists(Theta_path):

        print('Starting Full Training')
        Theta = []
        for epoch in range(epochs):

            _, _ = train(net, epoch, trainloader, device, criterion, optimizer)

            if use_scheduler:
                scheduler.step()

            theta = net.state_dict()
            Theta += [copy.deepcopy(theta)]

        torch.save(Theta, Theta_path)

    else:
        print('Loading Weights')
        Theta = torch.load(Theta_path)

    # Hold checkpoints in matrix N (model parameters) x E (training epochs) -
    # each row represents a model parameter, each column represents a training epoch.
    all_weights = matricize_list_of_model_params(Theta, convert_to_numpy=True)

    # Calculate SGD test accuracy
    if args.sgd:

        print('Calculating SGD Test Accuracy')
        fname_sgd = dir_name + '/sgd_test_acc.pt'                # Test accuracy array is saved in this file

        if not os.path.exists(fname_sgd):

            # Initialization of model and accuracy array
            net, optimizer, scheduler, use_scheduler, epochs, _, _ = get_model(args.model, device)
            Test_acc = []

            for epoch in range(epochs):

                # Load epoch weights to the model
                epoch_weights = all_weights[:, epoch]
                epoch_state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), epoch_weights)
                net.load_state_dict(epoch_state_dict)

                # Test model accuracy and add to accuracy array
                _, test_acc = test(net, device, testloader, criterion)
                print('SGD Epoch {} Test Acc {}'.format(epoch, test_acc))
                Test_acc += [test_acc]

            # Save accuracy array
            torch.save(Test_acc, fname_sgd)

    # Calculate Online CMD test accuracy
    if args.cmd_PostHoc:

        print('Calculating Post-hoc CMD Test Accuracy ')
        fname_posthoc = dir_name + '/cmd{}_posthoc_test_acc.pt'.format(cmd_modes)    # Test accuracy array is saved in this file
        cmd_name = dir_name + '/cmd{}_posthoc.pt'.format(cmd_modes)                 # CMD parameters are saved in this file

        if not os.path.exists(fname_posthoc):

            # Initialization of model and accuracy array
            net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp = get_model(args.model, device)
            Test_acc = []

            # Calculate CMD parameters or load
            if not os.path.exists(cmd_name):

                print('Calculating CMD parameters')
                samp_ind = sample_represented_weights(all_weights, net, cmd_samp)          # index of sampled weights for CMD
                a_vec, b_vec, ref_idx, ref_weights, related_mode = \
                    basic_cmd(all_weights, samp_ind, args.start, epochs, cmd_modes)

                torch.save((a_vec, b_vec, ref_idx, ref_weights, related_mode), cmd_name)

            else:
                print('Loading CMD parameters')
                a_vec, b_vec, ref_idx, ref_weights, related_mode = torch.load(cmd_name)

            # Calculate accuracy per epoch
            for epoch in range(epochs):

                # Calculate CMD weights and load to model
                r_w = ref_weights[:, epoch]            # find reference weights of last epoch
                weights_cmd = r_w[related_mode] * a_vec + b_vec

                state_dict_cmd = assign_vectorized_params_to_state_dict(net.state_dict(), weights_cmd)
                net.load_state_dict(state_dict_cmd)

                # Test model accuracy
                _, test_acc = test(net, device, testloader, criterion)
                print('Post-hoc CMD Epoch {} Test Acc {}'.format(epoch, test_acc))
                Test_acc += [test_acc]

            # Save accuracy array
            torch.save(Test_acc, fname_posthoc)

    # Calculate Online CMD test accuracy
    if args.cmd_online:

        print('Calculating Online CMD Test Accuracy ')
        fname_online = dir_name + '/cmd{}_online_warmup_{}_test_acc.pt'.format(cmd_modes, args.warmup_cmd)  # Test accuracy array is saved in this file
        cmd_name = dir_name + '/cmd{}_warmup_{}.pt'.format(cmd_modes, args.warmup_cmd)  # CMD parameters are saved in this file

        if not os.path.exists(fname_online):

            # Initialization of model and accuracy array
            net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp = get_model(args.model, device)
            Test_acc = []

            for epoch in range(epochs):

                # warm-up phase - use SGD model
                if epoch < args.warmup_cmd:

                    # Load epoch weights to the model
                    epoch_weights = all_weights[:, epoch]
                    epoch_state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), epoch_weights)
                    net.load_state_dict(epoch_state_dict)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                # End of warm-up phase - calculate CMD parameters
                elif epoch == args.warmup_cmd:

                    # If CMD parameters do not exist, calculate them. Else - load.
                    if not os.path.exists(cmd_name):
                        print('Calculating CMD parameters')
                        samp_ind = sample_represented_weights(all_weights, net, cmd_samp)          # index of sampled weights for CMD
                        a_vec, b_vec, ref_idx, ref_weights, related_mode = \
                            basic_cmd(all_weights, samp_ind, args.start, epoch + 1, cmd_modes)

                        torch.save((a_vec, b_vec, ref_idx, ref_weights, related_mode), cmd_name)

                    else:
                        print('Loading CMD parameters')
                        a_vec, b_vec, ref_idx, ref_weights, related_mode = torch.load(cmd_name)

                    # Calculate CMD weights and load to model
                    r_w = ref_weights[:, -1]  # find reference weights of last epoch
                    weights_cmd = r_w[related_mode] * a_vec + b_vec

                    state_dict_cmd = assign_vectorized_params_to_state_dict(net.state_dict(), weights_cmd)
                    net.load_state_dict(state_dict_cmd)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                # Post warm-up phase - Update CMD parameters
                else:

                    # Load epoch weights
                    epoch_weights = all_weights[:, epoch]

                    # Update reference weights arrays and A,B vectors
                    new_ref_weights = epoch_weights[np.asarray(ref_idx).astype(int)]
                    ref_weights = np.concatenate((ref_weights, np.expand_dims(new_ref_weights, axis=1)), axis=1)
                    weights_info = create_AB_DF(epoch_weights, related_mode)
                    a_vec, b_vec = update_AB_vec(a_vec, b_vec, ref_weights, np.expand_dims(epoch_weights, axis=1),
                                                 weights_info)

                    # Calculate CMD weights and load to model
                    weights_cmd = new_ref_weights[related_mode] * a_vec + b_vec
                    state_dict_cmd = assign_vectorized_params_to_state_dict(net.state_dict(), weights_cmd)
                    net.load_state_dict(state_dict_cmd)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                # Add accuracy to accuracy array
                print('CMD Iter Epoch {} Test Acc {}'.format(epoch, test_acc))
                Test_acc += [test_acc]

            # Save accuracy array
            torch.save(Test_acc, fname_online)

    # Calculate Embedded CMD test accuracy
    if args.cmd_embed:

        print('Calculating Embedded CMD Test Accuracy')
        fname_embed = dir_name + '/cmd{}_embed_warmup_{}_L_{}_P_{}_rel_{}_test_acc.pt'\
            .format(cmd_modes, args.warmup_cmd, args.embed_window, args.embed_perc, args.perc_rel) # Test accuracy array is saved in this file

        cmd_name = dir_name + '/cmd{}_warmup_{}.pt'.format(cmd_modes, args.warmup_cmd)  # CMD parameters are saved in this file

        if not os.path.exists(fname_embed):

            # Initialization of model and accuracy array
            net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp = get_model(args.model, device)
            Test_acc = []
            sum_all = 0                     # Counter for embedded weights percentage
            sum_embed = 0                   # Counter for embedded weights percentage
            embed_idx = []
            embed_idx_arr = np.asarray(embed_idx)       # array of weight indexes for embedded weights

            for epoch in range(epochs):

                # warm-up phase - use SGD model
                if epoch <= args.warmup_cmd:

                    # Load epoch weights to the model
                    epoch_weights = all_weights[:, epoch]
                    epoch_state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), epoch_weights)
                    net.load_state_dict(epoch_state_dict)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                    if use_scheduler:
                        scheduler.step()

                    # Counter for embedded weights percentage
                    sum_all += len(epoch_weights)

                    # End of warm-up phase - calculate CMD parameters
                    if epoch == args.warmup_cmd:

                        # If CMD parameters do not exist, calculate them. Else - load.
                        if not os.path.exists(cmd_name):

                            print('Calculating CMD parameters')
                            samp_ind = sample_represented_weights(all_weights, net, cmd_samp)  # index of sampled weights for CMD
                            a_vec, b_vec, ref_idx, ref_weights, related_mode = \
                                basic_cmd(all_weights, samp_ind, args.start, epoch + 1, cmd_modes)
                            torch.save((a_vec, b_vec, ref_idx, ref_weights, related_mode), cmd_name)

                        else:

                            print('Loading CMD parameters')
                            a_vec, b_vec, ref_idx, ref_weights, related_mode = torch.load(cmd_name)

                        # Save current A,B values as old
                        a_vec_old = copy.deepcopy(a_vec)
                        b_vec_old = copy.deepcopy(b_vec)

                # Post warm-up phase - Train embedded model and update CMD parameters
                else:

                    # If no weights are embedded continue with SGD weights. Else - train embedded model.
                    if len(embed_idx_arr) == 0:

                        epoch_weights = all_weights[:, epoch]

                    else:
                        _, _ = train(net, epoch, trainloader, device, criterion, optimizer)
                        theta = net.state_dict()
                        epoch_weights = vectorize_model_params(theta).data.cpu().numpy()

                    if use_scheduler:
                        scheduler.step()

                    # Update reference weights array and A,B vectors
                    new_ref_weights = epoch_weights[np.asarray(ref_idx).astype(int)]
                    ref_weights = np.concatenate((ref_weights, np.expand_dims(new_ref_weights, axis=1)), axis=1)
                    weights_info = create_AB_DF(epoch_weights, related_mode)
                    a_vec, b_vec = update_AB_vec(a_vec, b_vec, ref_weights, np.expand_dims(epoch_weights, axis=1),
                                                 weights_info)

                    # Restore fixed A,B values for embedded weights and calculate the embedded model
                    if len(embed_idx_arr) > 0:
                        a_vec[embed_idx_arr] = a_vec_old[embed_idx_arr]
                        b_vec[embed_idx_arr] = b_vec_old[embed_idx_arr]

                    weights_embed = copy.deepcopy(epoch_weights)
                    weights_cmd = new_ref_weights[related_mode] * a_vec + b_vec
                    if len(embed_idx_arr) > 0:
                        weights_embed[embed_idx_arr] = weights_cmd[embed_idx_arr]

                    state_dict_embed = assign_vectorized_params_to_state_dict(net.state_dict(), weights_embed)
                    net.load_state_dict(state_dict_embed)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                    # This part is only for embedding epochs
                    if epoch % args.embed_window == 0:

                        # Calculate diff array
                        a_diff = np.abs(np.asarray(a_vec_old - a_vec))
                        b_diff = np.abs(np.asarray(b_vec_old - b_vec))
                        diff = np.sqrt(a_diff * a_diff + b_diff * b_diff)

                        # Save current A,B values as old (note that the A,B values of embedded weights were already restored)
                        a_vec_old = copy.deepcopy(a_vec)
                        b_vec_old = copy.deepcopy(b_vec)

                        # Remove all frozen weights from the calculation by making them max(diff) + 1
                        if len(embed_idx_arr) > 0:
                            diff[embed_idx_arr] = diff.max() + 1

                        # Calculate number of weights to embedded
                        if args.perc_rel:
                            tot_w = len(diff) - len(embed_idx_arr)

                        else:
                            tot_w = len(diff)

                        num = np.ceil(tot_w * (args.embed_perc / 100)).astype(int)

                        # New embedded indexes
                        idx = np.argsort(diff)[:num]

                        # Update the embedded weights index array
                        embed_idx_arr = np.concatenate((embed_idx_arr, idx)).astype(int)
                        embed_idx_arr = np.unique(embed_idx_arr)                    # make sure each index exists once

                        # Make sure reference weights are not embedded
                        for ref in ref_idx:
                            ref = ref.astype(int)
                            embed_idx_arr = embed_idx_arr[embed_idx_arr != ref]

                    # Counters for embedded weights percentage
                    sum_all += len(epoch_weights)
                    sum_embed += len(embed_idx_arr)

                # Add accuracy to accuracy array and print accuracy and embedded percentage
                print('Embedded CMD Epoch {} Test Acc {}'.format(epoch, test_acc))
                Test_acc += [test_acc]

                print('Embedded CMD saved {}% of weights'.format(100 * (sum_embed / sum_all)))

            # Save accuracy array
            torch.save(Test_acc, fname_embed)

    # Calculate P-BFGS test accuracy
    if args.pbfgs:

        print('Calculating P-BFGS Test Accuracy')
        fname_pbfgs = dir_name + '/pbfgs_warmup_{}_dims_{}_test_acc.pt'.format(args.warmup_pbfgs, args.pbfgs_num)    # Test accuracy array is saved in this file

        if not os.path.exists(fname_pbfgs):

            # Initialization of model and accuracy array
            net, optimizer, scheduler, use_scheduler, epochs, _, _ = get_model(args.model, device)
            optimizer = optim.SGD(net.parameters(), lr=1, momentum=0)       # optimizer used for P-BFGS training in the low dimensional space
            Test_acc = []

            # P-BFGS requires saving a temporary checkpoint of the model
            tmp_name = dir_name + '/pbfgs_first_{}_dims_{}'.format(args.warmup_pbfgs, args.pbfgs_num)

            # Calculaate number of parameters with gradients
            model_params = len(get_model_param_vec(net))
            all_weights = all_weights[:, :args.warmup_pbfgs + 1]
            W = np.zeros((args.warmup_pbfgs + 1, model_params))             # P-BFGS warmup phase array of checkpoints

            for epoch in range(epochs):

                # warm-up phase - use SGD model
                if epoch < args.warmup_pbfgs:

                    # Load epoch weights to the model
                    epoch_weights = all_weights[:, 0]
                    epoch_state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), epoch_weights)
                    net.load_state_dict(epoch_state_dict)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                    # Remove column from all weights array for memory saving
                    all_weights = all_weights[:, 1:]

                    # Insert current model weights to W array
                    W[epoch, :] = get_model_param_vec(net)

                # End of warm-up phase - calculate P-BFGS model
                elif epoch == args.warmup_pbfgs:

                    # Load epoch weights to the model
                    epoch_weights = all_weights[:, 0]
                    all_weights = []            # this array is not further required
                    epoch_state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), epoch_weights)
                    net.load_state_dict(epoch_state_dict)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                    # Insert current model weights to W array
                    W[epoch, :] = get_model_param_vec(net)

                    # Calculate P array using PCA on W
                    pca = PCA(n_components=args.pbfgs_num)
                    pca.fit_transform(W)
                    W = []          # this array is not further required
                    P = np.array(pca.components_)
                    P = torch.from_numpy(P).cuda()

                    # Initialization of P-BFGS parameters
                    gk_last = None
                    sk = None
                    grad_res_momentum = 0
                    Bk = torch.eye(args.pbfgs_num).cuda()

                # Post warm-up - Train model using P-BFGS training in the low dimensional space
                else:

                    # P-BFGS training epoch
                    cudnn.benchmark = True
                    optimizer.zero_grad()
                    gk_last, sk, grad_res_momentum = train_pbfgs(args, trainloader, net, criterion, optimizer, P,
                                                                 gk_last, sk, Bk, grad_res_momentum, tmp_name)

                    # Test model accuracy
                    _, test_acc = test(net, device, testloader, criterion)

                # Add accuracy to accuracy array
                print('P-BFGS Epoch {} Test Acc {}'.format(epoch, test_acc))
                Test_acc += [test_acc]

            # Save accuracy array
            torch.save(Test_acc, fname_pbfgs)
            P = []                      # this array is not further required

    # Create Test Accuracy graph
    plot_name = dir_name + '/all_methods_test_acc.png'               # This will be the plot name
    plt.figure()

    # Plot a curve for each method
    if args.sgd:
        test_acc_sgd = torch.load(fname_sgd)
        plt.plot(test_acc_sgd, lw=1, color='darkorange', label='SGD - {:.2f}%'.format(100 * test_acc_sgd[-1]))

    if args.cmd_PostHoc:
        test_acc_posthoc = torch.load(fname_posthoc)
        plt.plot(test_acc_posthoc, lw=1, color='red', label='CMD Post-hoc - {:.2f}%'.format(100 * test_acc_posthoc[-1]))

    if args.cmd_online:
        test_acc_online = torch.load(fname_online)
        plt.plot(test_acc_online, lw=1, color='blue', label='CMD Online - {:.2f}%'.format(100 * test_acc_online[-1]))

    if args.cmd_embed:
        test_acc_embed = torch.load(fname_embed)
        plt.plot(test_acc_embed, lw=1, color='green', label='Embedded CMD - {:.2f}%'.format(100 * test_acc_embed[-1]))

    if args.pbfgs:
        test_acc_pbfgs = torch.load(fname_pbfgs)
        plt.plot(test_acc_pbfgs, lw=1, color='purple', label='P-BFGS - {:.2f}%'.format(100 * test_acc_pbfgs[-1]))

    # Figure parameters
    fs = 20
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel('Test Accuracy', fontsize=fs)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # plt.grid()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

############################################## Arguments ####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results',
                        help='results directory - all model checkpoints, CMD parameters and test accuracy arrays will be saved here.')
    parser.add_argument('--model', type=str, default='res18',
                        help='Model name. Options: res18, preres164, wideres, lenet, googlenet, vit')
    parser.add_argument('--start', type=int, default=0,
                        help='starting epoch of the warm-up phase for CMD')
    parser.add_argument('--warmup_cmd', type=int, default=20,
                        help='ending epoch of warm-up phase for CMD')
    parser.add_argument('--warmup_pbfgs', type=int, default=80,
                        help='Number of warm up epochs fore P-BFGS')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training')

    # parser.add_argument('--cmd_modes', type=int, default=10,
    #                     help='number of CMD modes')
    # parser.add_argument('--cmd_samp', type=int, default=1000,
    #                     help='number of CMD sampling weights')

    parser.add_argument('--embed_window', type=int, default=10,
                        help='Number of epochs between each embedding epoch')
    parser.add_argument('--embed_perc', type=int, default=10,
                        help='Percentage of weights to be embedded')
    parser.add_argument('--perc_rel', type=bool, default=False,
                        help='Percentage of all model weights (False) or not embedded weights (True)')

    parser.add_argument('--pbfgs_num', default=40, type=int, metavar='N',
                        help='number of components for PCA')
    parser.add_argument('--accumulate', default=1, type=int, metavar='N',
                        help='how many times accumulate for gradients')

    parser.add_argument('--sgd', type=bool, default=True,
                        help='Perform SGD when True')
    parser.add_argument('--cmd_PostHoc', type=bool, default=True,
                        help='Perform post-hoc CMD when True')
    parser.add_argument('--cmd_online', type=bool, default=True,
                        help='Perform online CMD when True')
    parser.add_argument('--cmd_embed', type=bool, default=False,
                        help='Perform embedded CMD when True')
    parser.add_argument('--pbfgs', type=bool, default='',
                        help='Perform P-BFGS when True')

    args = parser.parse_args()

    main(args)
