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
        _, testloader = get_cifar10_loaders_vit(args.batch_size)

    else:
        _, testloader = get_cifar10_loaders(args.batch_size)

    # Results Directory
    results_dir = args.results_dir
    dir_name = results_dir + '/' + args.model

    dir_path = './' + dir_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_name)

    # Initialization of model and optimizer
    net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp = get_model(args.model, device)

    # Loading CMD parameters
    cmd_name = os.path.join(dir_name, args.cmd_path)
    print('Loading CMD parameters')
    a_vec, b_vec, ref_idx, ref_weights, related_mode = torch.load(cmd_name)

    # Using the last values of the reference weights from training
    r_w = np.asarray(ref_weights)[:, -1]

    # Calculating reconstructed full model using CMD
    theta_cmd = r_w[related_mode] * a_vec + b_vec
    state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), theta_cmd)
    net.load_state_dict(state_dict)
    
    # 2D Landscape calculation
    table_name = os.path.join(dir_name, args.table_path)
    if not os.path.isfile(table_name):
        print("Calculating 2D Landscape")
        grid, Jout, r_w_star = calc_2d_landscape(r_w, a_vec, b_vec, related_mode, net, test_theta,
                                                 testloader, criterion, device)
        torch.save((grid, Jout), table_name)

    else:
        print("Loading 2D Landscape")
        grid, Jout = torch.load(table_name)
        
        # Finding the optimal r_w values
        r_w_star_ind = np.asarray(np.where(np.asarray(Jout) == np.asarray(Jout).min()))
        if len(r_w_star_ind.shape) > 1:
            r_w_star_ind = r_w_star_ind[:, 0]

        x, y = grid
        r_w_star = np.asarray([x[r_w_star_ind[0], 0], y[0, r_w_star_ind[1]]])
    
    # Test accuracy for starting and optimal points
    theta_star = r_w_star[related_mode] * a_vec + b_vec
    _, test_acc_rw = test_theta(net, np.asarray(theta_cmd), device, testloader, criterion)
    _, test_acc_star = test_theta(net, np.asarray(theta_star), device, testloader, criterion)
    
    # Saving the 2D landscape plot
    plot_name = os.path.join(dir_name, args.plot_path)
    if not os.path.isfile(plot_name):

        fig = plt.figure()
        print('\nSaving 2d plot')
        x, y = grid
        ax = fig.add_subplot(111)
        img = ax.pcolormesh(x, y, (-1 * Jout))
        ax.contour(x, y, (-1 * Jout), 7, colors='k')
        ax.set_xlabel('w_0')
        ax.set_ylabel('w_1')
        plt.plot(r_w_star[0], r_w_star[1], '*', label=f'Optimal - {test_acc_star}%')
        plt.plot(r_w[0], r_w[1], 'o', label=f'Original - {test_acc_rw}%')
        plt.colorbar(img, ax=ax)
        plt.legend()
        plt.title('Loss Landscape')
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    ############################################## Arguments ####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results',
                        help='results directory - all model checkpoints, CMD parameters and test accuracy arrays will be saved here.')
    parser.add_argument('--model', type=str, default='res18',
                        help='Model name. Options: res18, preres164, wideres, lenet, googlenet, vit')
    parser.add_argument('--cmd_path', type=str, default='cmd2_posthoc.pt',
                        help='path to cmd parameters, inside the results directory')
    parser.add_argument('--table_path', type=str, default='cmd2_posthoc_2dlandscape.pt',
                        help='path to table of 2D landscape calculation')
    parser.add_argument('--plot_path', type=str, default='cmd2_posthoc_2dlandscape.png',
                        help='path to plot of 2D landscape calculation')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training')

    args = parser.parse_args()

    main(args)
    