import copy
import math
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import shutil
import scipy

# models
from models.preresnet import preresnet
from models.wide_resnet import Wide_ResNet
from models.Lenet import LeNet5
from models.googlenet import GoogLeNet
from models.resnet import ResNet18


def matricize_list_of_model_params(state_dicts_list, convert_to_numpy=True):
    """
    recieves:
    state_dicts_list, a list of state_dicts. All state_dicts in state_dicts_list have K parameters. len(state_dicts_list) = T
    returns:
    Theta, a [K X T] matrix (numpy array), where each column is a vectorized state_dict, where the i_th column is constructed from state_dicts_list[i]
    """

    assert len(state_dicts_list) > 1  # assert that there is more than one time step in Theta..

    vectorized_param_list = [vectorize_model_params(p).data.cpu() for p in state_dicts_list]  # each of the T elements in vectorized_param_list is converted to a K-element pytorch vector
    if convert_to_numpy:
        vectorized_param_list = [p.numpy() for p in vectorized_param_list]  # each of the T elements in vectorized_param_list is converted to a K-element numpy vector
        return np.stack(vectorized_param_list, axis=1)  # K X T
    else:
        return torch.stack(vectorized_param_list, dim=1)  # K X T


def assign_vectorized_params_to_state_dict(state_dict, param_vec):
    """
    recieves:
    param_vec is a vector of numpy parameters as returned by function "vectorize_model_params" above
    state_dict, a pytorch model's state_dict compatible with param_vec
    modifies:
    Update state_dict's parameters to be param_vec.
    """
    start_layer_idx = 0
    layer = 0
    for paramname in state_dict:
        param = state_dict[paramname]
        layer_size = param.numel()
        if len(param.size()) == 0:  # skip empty tensors
            continue
        end_layer_idx = start_layer_idx + layer_size
        layer_shape = tuple(param.data.shape)
        # print(param_vec[start_layer_idx: end_layer_idx].shape)
        # print(layer_shape)
        param.data = torch.reshape(torch.from_numpy(param_vec[start_layer_idx: end_layer_idx]), layer_shape)
        start_layer_idx = end_layer_idx
        layer += 1
    return state_dict


def vectorize_model_params(state_dict):
    """
    recieves:
    state_dict of the net
    returns:
    param_vec: parameters of net as a vector (deep copy, pytorch tensor)""""""
    """
    param_vec = []
    for param in state_dict.values():
        if len(param.size()) == 0:  # skip empty tensors
            continue
        param_vec.append(torch.flatten(param.data.detach().clone()))
    param_vec = torch.cat(param_vec, dim=0)
    return copy.deepcopy(param_vec)


last_time = time.time()
begin_time = last_time


def format_time(seconds):
    """
        This function is used for the progress_bar() function below
        """
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def progress_bar(current, total, msg=None):
    """
        function that presents a progress bar of a training epoch. Used by the train() function below
        """
    _, term_width = shutil.get_terminal_size()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 65.
    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

############give credit on this function##################
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
        Taken from: https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_cifar10_loaders(batch_size=128):
    """
        recieves:
        batch size
        returns:
        CIFAR10 training set and test set as pytorch data loaders
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def get_cifar10_loaders_vit(batch_size=128):
    """
            recieves:
            batch size
            returns:
            CIFAR10 training set and test set as pytorch data loaders with image size of 224x224
        """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


def get_model(model_name, device):
    """
                recieves:
                model name and device to load model to.
                returns:
                net - initialized model, optimizer for training, scheduler for training,
                scheduler indicator (bool) and number of epochs (int).
            """

    if model_name == 'res18':
        net = ResNet18(num_classes=10, softmax_output=False)
        net = nn.DataParallel(net).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=0)
        scheduler = []
        use_scheduler = False
        epochs = 100
        cmd_modes = 2
        cmd_samp = 1000

    if model_name == 'wideres':
        net = Wide_ResNet(28, 10, 0.3, 10)
        net = nn.DataParallel(net).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
        scheduler = []
        use_scheduler = False
        epochs = 120
        cmd_modes = 5
        cmd_samp = 1000

    if model_name == 'preres164':
        net = preresnet(depth=164)
        net = nn.DataParallel(net).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
        scheduler = []
        use_scheduler = False
        epochs = 150
        cmd_modes = 2
        cmd_samp = 10000

    if model_name == 'lenet':
        net = LeNet5(num_classes=10, grayscale=False)
        net = nn.DataParallel(net).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        scheduler = []
        use_scheduler = False
        epochs = 150
        cmd_modes = 10
        cmd_samp = 1000

    if model_name == 'googlenet':
        net = GoogLeNet()
        net = nn.DataParallel(net).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        scheduler = []
        use_scheduler = False
        epochs = 120
        cmd_modes = 7
        cmd_samp = 5000

    if model_name == 'vit':
        net = torchvision.models.vit_b_16(pretrained=True)
        net = nn.DataParallel(net).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.03, momentum=0.9, weight_decay=0)
        epoch_steps = np.ceil(50000 / 128).astype(int)                                 # change '128' if batch size is different, 50000 is CIFAR10 training set size
        warmup_epochs = 5
        warmup_steps = warmup_epochs * epoch_steps
        epochs = 90
        total_steps = epochs * epoch_steps
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
        use_scheduler = True
        cmd_modes = 10
        cmd_samp = 1000

    return net, optimizer, scheduler, use_scheduler, epochs, cmd_modes, cmd_samp


def train(net, epoch, loader, device, criterion, optimizer):
    """
            This function performs a training epoch
    """
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_accuracy = correct / total

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%%'
                     % (train_loss / (batch_idx + 1), 100. * train_accuracy))

    return train_loss / total, train_accuracy


def test(net, device, loader, criterion):
    """
            recieves:
            net, device, loader and criterion
            returns:
            accuracy and loss of net on the data in loader using the given device and the criterion.
        """

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    torch.cuda.empty_cache()
    return test_loss / total, test_acc


def test_theta(net, theta, device, loader, criterion):
    """
    Load theta, a numpy vector, into the state_dict of net and test
    """
    state_dict = assign_vectorized_params_to_state_dict(net.state_dict(), theta)
    net.load_state_dict(state_dict)
    test_loss, test_acc = test(net, device, loader, criterion=criterion)
    return test_loss, test_acc


def test_reference_weights_constructor(r_w, a_vec, b_vec, related_mode, net, test_theta, testloader, criterion, device):
    """
    :return: A function that tests performance on a given set of reference weights inputs
    """

    def test_reference_weights(r_w, return_mode='loss'):
        """
        hard-coded: a_vec, b_vec, ref_weights, related_mode, net, test_function, test_loader, device
        """
        # calculate parameters using cmd, and latest reference weights
        theta = r_w[related_mode] * a_vec + b_vec

        # test
        test_loss, test_acc = test_theta(net, theta, device, testloader, criterion)
        if return_mode == 'both':
            return test_loss, test_acc
        elif return_mode == 'loss':
            return test_loss
        elif return_mode == 'acc':
            return -1 * test_acc

    return test_reference_weights


def calc_2d_landscape(r_w, a_vec, b_vec, related_mode, net, test_theta, val_loader, criterion, device):

    # Defining a function that tests performance on a given set of reference weights inputs
    val_reference_weights = test_reference_weights_constructor(r_w, a_vec, b_vec, related_mode, net, test_theta,
                                                               val_loader, criterion, device)

    # creating grid of reference weights values
    r_w_range = []
    for r_w_cur in r_w:

        r_w_cur_abs = np.abs(r_w_cur)
        r_w_range += [slice(r_w_cur - 0.5 * r_w_cur_abs, r_w_cur + 0.5 * r_w_cur_abs, 0.02 * r_w_cur_abs)]

    r_w_range = tuple(r_w_range)

    # exhaustive search + numerical grad. desc.
    r_w_star, loss_star, grid, Jout = scipy.optimize.brute(val_reference_weights, r_w_range, full_output=True,
                                                           finish=None)
    return grid, Jout, r_w_star
