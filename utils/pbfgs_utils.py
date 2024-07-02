import numpy as np
import torch.nn as nn
import os
import torch

"""
 All of the functions in this file are taken from:
 https://github.com/nblt/DLDR
 official implementation of 'Low Dimensional Trajectory Hypothesis is True: DNNs can be Trained in Tiny Subspaces' (TPAMI 2022)
"""


def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)


def get_model_grad_vec(model):
    """
        Return the model grad as a vector
    """

    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)


def update_grad(model, grad_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx + size].reshape(arr_shape)
        idx += size


def update_param(model, param_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.data.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.data = param_vec[idx:idx + size].reshape(arr_shape)
        idx += size


def P_plus_BFGS(args, model, optimizer, grad, oldf, X, y, P, gk_last, sk, Bk, grad_res_momentum, fname):
    """
        P_plus_BFGS algorithm
    """
    alpha = 0
    gamma = 0.9
    rho = 0.4
    sigma = 0.4
    gk = torch.mm(P.to(torch.float64), grad.reshape(-1, 1).to(torch.float64))

    grad_proj = torch.mm(P.transpose(0, 1), gk)
    grad_res = grad - grad_proj.reshape(-1)

    # Quasi-Newton update
    if gk_last is not None:
        yk = gk - gk_last
        g = (torch.mm(yk.transpose(0, 1), sk))[0, 0]
        if (g > 1e-20):
            pk = 1. / g
            t1 = torch.eye(args.pbfgs_num).cuda() - torch.mm(pk * yk, sk.transpose(0, 1))
            Bk = torch.mm(torch.mm(t1.transpose(0, 1), Bk.to(torch.double)), t1) + torch.mm(pk * sk, sk.transpose(0, 1))

    gk_last = gk
    dk = -torch.mm(Bk.to(torch.float64), gk.to(torch.float64))

    # Backtracking line search
    m = 0
    search_times_MAX = 20
    descent = torch.mm(gk.transpose(0, 1), dk)[0, 0]

    # Copy the original parameters
    model_name = fname + '_temporary.pt'
    torch.save(model.state_dict(), model_name)

    sk = dk
    while (m < search_times_MAX):
        update_grad(model, torch.mm(P.transpose(0, 1), -sk).reshape(-1))
        optimizer.step()
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        newf = loss.item()
        model.load_state_dict(torch.load(model_name))

        if (newf < oldf + sigma * descent):
            break

        m = m + 1
        descent *= rho
        sk *= rho

    # SGD + momentum for the remaining part of gradient
    grad_res_momentum = grad_res_momentum * gamma + grad_res

    # Update the model grad and do a step
    update_grad(model, torch.mm(P.transpose(0, 1), -sk).reshape(-1) + grad_res_momentum * alpha)
    optimizer.step()

    return gk_last, sk, grad_res_momentum


def train_pbfgs(args, train_loader, model, criterion, optimizer, P, gk_last, sk, Bk, grad_res_momentum, fname):

    # Switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        # Load batch data to cuda
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Compute gradient and do SGD step
        loss.backward()

        # Do P_plus_BFGS update
        gk = (get_model_grad_vec(model) / args.accumulate)
        gk_last, sk, grad_res_momentum = P_plus_BFGS(args, model, optimizer, gk, loss.item(), input_var, target_var, P,
                                                     gk_last, sk, Bk, grad_res_momentum, fname)
        optimizer.zero_grad()

        return gk_last, sk, grad_res_momentum

