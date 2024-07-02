import copy
import math
from collections import Counter
import scipy.cluster.hierarchy as sch
import itertools
import os
import random
import torch
import numpy as np
from numpy.linalg import pinv
from tqdm import tqdm
from ordered_set import OrderedSet
import pandas as pd


def cluster_corr(corr_array, inplace=False, clust_method='distance', clust_t=2):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to each other

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    t = pairwise_distances.max()/clust_t if clust_method == 'distance' else clust_t if clust_method == 'maxclust' else None
    print('clustering criterion is %s, t=%f' % (clust_method, t))
    idx_to_cluster_array = sch.fcluster(linkage, t, criterion=clust_method)

    idx = np.argsort(idx_to_cluster_array)
    modes = idx_to_cluster_array[idx]
    corr_in_cluster = [corr_array.iloc[idx[modes == mode], :].T.iloc[idx[modes == mode], :] for mode in range(min(modes), max(modes)+1)]

    modes_count = Counter(modes)
    modes_n, count = modes_count.keys(), modes_count.values()
    modes_counter_df = pd.DataFrame(count, index=modes_n)
    sorted_modes = modes_counter_df.sort_values(by=0, ascending=False).index.values
    mode_covert_dict = {k: i for i, k in enumerate(sorted_modes)}

    sorted_idx_to_cluster_array = np.array([mode_covert_dict[idx_to_cluster_array[i]] for i in range(len(idx_to_cluster_array))])
    sorted_idx = np.argsort(sorted_idx_to_cluster_array)
    sorted_corr_in_cluster = [corr_in_cluster[i-1] for i in sorted_modes]
    sorted_modes_per_weight = sorted_idx_to_cluster_array[sorted_idx]

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        sorted_clustered_corr = corr_array.iloc[sorted_idx, :].T.iloc[sorted_idx, :]
        return sorted_clustered_corr, sorted_idx_to_cluster_array, sorted_modes_per_weight, sorted_corr_in_cluster
        #return corr_array.iloc[sorted_idx, :].T.iloc[sorted_idx, :], corr_array.iloc[idx, :].T.iloc[idx, :], idx_to_cluster_array, modes, corr_in_cluster
    return corr_array[idx, :][:, idx], idx_to_cluster_array


def compute_modes(samp_weights, samp_ind, clust_method='distance', clust_t=2):
    """
        Divides the sampled weights to modes.
    """

    w_corr = np.corrcoef(samp_weights)
    w_corr[np.isnan(w_corr)] = 0
    samp_corr = pd.DataFrame(np.abs(w_corr), index=samp_ind, columns=samp_ind)
    sorted_clustered_corr, sorted_idx_to_cluster_array, sorted_modes_per_weight, sorted_corr_in_cluster = cluster_corr(
        samp_corr, clust_method=clust_method, clust_t=clust_t)
    return sorted_clustered_corr, sorted_idx_to_cluster_array, sorted_modes_per_weight, sorted_corr_in_cluster, samp_corr


def find_ref_w_per_mode(corr_in_cluster, all_weights):
    """
        Returns the reference weight in the given mode of weights
    """
    ref_w_idx = [cluster.index[cluster.mean().argmax()] for cluster in corr_in_cluster]
    ref_weight = all_weights[ref_w_idx, :]
    return ref_w_idx, ref_weight


def handle_constants(weights_corr):
    """
        Returns 0 for cells where the correlation is NaN.
    """
    i, j = np.where(np.isnan(np.array(weights_corr)))
    for k, m in zip(i, j):
        weights_corr[k][m] = 0
    return weights_corr


def get_struct_from_net(net):
    """
        Returns name of layer per weight in the model for the given model
    """
    detailed_model_layers = {k: math.prod(v.shape) for k, v in net.state_dict().items() if
                             'num_batches_tracked' not in k}
    detailed_w_to_layer_list = []
    for k, v in detailed_model_layers.items():
        detailed_w_to_layer_list += [k] * v

    return np.array(detailed_w_to_layer_list)


def sample_represented_weights(all_weights, net, k=1000, sampling='U_B'):
    """
            Returns indexes of sampled K weights from the model
    """
    w_to_layer = get_struct_from_net(net)

    layers_set = OrderedSet(w_to_layer)
    layers_ranges = [(min(np.where(w_to_layer == layer)[0]), max(np.where(w_to_layer == layer)[0])) for layer in
                     layers_set]
    layers_size = [val for val in list(Counter(w_to_layer).values())]
    print('%d layers' % len(layers_size))
    print(layers_size)
    if sampling == 'var':
        print('=======')
        probs = np.exp(np.std(all_weights.numpy(), axis=1))
        probs /= sum(probs)
        samp_ind = list(np.random.choice(np.arange(all_weights.shape[0]), size=k, replace=False, p=probs))
    elif sampling == 'U' or k / 2 < len(layers_size):
        samp_ind = np.sort(random.sample(range(all_weights.shape[0]), k))
        samps = [sum((samp_ind >= w_range[0]) & (samp_ind <= w_range[1])) for i, w_range in enumerate(layers_ranges)]
        print(sum(samps))
        print('samples from layers:')
        print(samps)
    elif sampling == 'U_B':
        samps = [min(int(k / (2 * len(layers_set))), l_size) for l_size in layers_size]
        samps = [samp + int((layers_size[i] / sum(layers_size)) * k / 2) if int(
            (layers_size[i] / sum(layers_size)) * k / 2) <= layers_size[i] - samp else samp for i, samp in
                 enumerate(samps)]
        if k - sum(samps) > 0:
            for i, samp_idx in enumerate(np.argsort(samps)):
                if k - sum(samps) <= layers_size[samp_idx] - samps[samp_idx]:
                    samps[samp_idx] += k - sum(samps)
                    break

        samp_ind = [random.sample(range(w_range[0], w_range[1] + 1), samps[i]) for i, w_range in
                    enumerate(layers_ranges)]
        samp_ind = list(itertools.chain.from_iterable(samp_ind))

    return samp_ind


def calc_related_mode(all_weights, ref_w_idx):
    """
        Description: This function receives all the weights and the center weights index and returnes the related mode of each weight.

        INPUTS
        all_weights: Matrix of size - number of all the weights X number of epochs - contains the value of each weight at each epoch.
        ref_w_idx: Array of size - number of clusters X 1 - contains the index of each center weight.

        OUTPUTS
        weights_modes: Array of size - number of all the weights X 1 - contains the mode each weight is related to.
    """
    # extract the center weights values per epoch
    ref_w_arr = np.array(all_weights[ref_w_idx, :])  # Size: number_of_modes X number of epochs
    weights_corr_w_ref = np.zeros((all_weights.shape[0], len(ref_w_idx)))

    for idx in range(len(ref_w_idx)):

        ref_w = ref_w_arr[idx, :]
        weights_corr_w_ref[:, idx] = np.abs((len(ref_w) * np.sum(all_weights * ref_w[None, :], axis=-1) - (np.sum(all_weights, axis=-1) * np.sum(ref_w))) / (np.sqrt(np.abs((len(ref_w) * np.sum(all_weights ** 2, axis=-1) - np.sum(all_weights, axis=-1) ** 2) * (len(ref_w) * np.sum(ref_w ** 2) - np.sum(ref_w) ** 2)))))

    # fix Nan's
    weights_corr_w_ref = handle_constants(weights_corr_w_ref)
    weights_modes = np.argmax(np.asarray(weights_corr_w_ref), axis=1)

    return weights_modes


def create_AB_DF(all_weights, related_mode):
    """
        Description: This function receives all the weights and the related mode of each weight and returnes a data frame containing the related mode and two additional columns for the a,b vectors.

        INPUTS
        all_weights: Matrix of size - number of all the weights X number of epochs - contains the value of each weight at each epoch.
        related_mode: Array of size - number of all the weights X 1 - contains the related mode of each weight.

        OUTPUTS
        DF: Dataframe of size - number of all the weights X 3 - contains the mode each weight is related to and two empty columns.
    """

    # extract each weight's index
    all_w_idx = list(OrderedSet(range(all_weights.shape[0])))

    # cretae a Dataframe with the related mode for each weight
    DF = pd.DataFrame({'related_mode': related_mode}, index=all_w_idx)

    # add two empty columns
    DF['a'] = [np.nan] * DF.shape[0]
    DF['b'] = [np.nan] * DF.shape[0]

    return DF


def calc_AB_vec(weights_info, all_weights, ref_w_idx):
    """
        Description: This function receives all the weights and the related mode of each weight and returnes the a,b vectors.

        INPUTS
        weights_info: Dataframe of size - number of all the weights X 3 - contains the mode each weight is related to and two empty columns.
        all_weights: Matrix of size - number of all the weights X number of epochs - contains the value of each weight at each epoch.
        ref_w_idx - Array of size - number of clusters X 1 - contains the index of each center weight.

        OUTPUTS
        a_vec: Array of size - number of all the weights X 1 - contains the value of the a vector for each weight (all clusters)
        b_vec: Array of size - number of all the weights X 1 - contains the value of the b vector for each weight (all clusters)
    """
    # find number of epochs
    n_epochs = all_weights.shape[1]

    # extract the center weights values per epoch
    ref_weights = np.array(all_weights[np.array(ref_w_idx).astype(int), :])  # Size: number_of_modes X number of epochs

    # a loop for each mode, using the different center weight for the a,b calculation
    for mode in tqdm(range(len(set(weights_info['related_mode'].values)))):
        # find the index of all the weights in the current mode
        weights_in_mode_idx = weights_info.index[np.where(weights_info['related_mode'].values == mode)[0]]

        # find the values of all the weights in the current mode
        weights_in_mode = all_weights[weights_in_mode_idx, :]

        # the current mode center weight values per epoch
        ref_weight = ref_weights[mode]

        # current mode a, b calculations according to CMD method
        w_r_tilde = np.concatenate((np.expand_dims(ref_weight, 0), np.ones((1, n_epochs))), axis=0)
        A_tilde = weights_in_mode.dot(w_r_tilde.T).dot(pinv(w_r_tilde.dot(w_r_tilde.T)))
        a_vec = np.expand_dims(A_tilde[:, 0], 1)
        b_vec = np.expand_dims(A_tilde[:, 1], 1)

        # insert the current mode a,b values in to the all weights a,b vectors
        weights_info['a'].iloc[weights_in_mode_idx] = a_vec.squeeze(1)
        weights_info['b'].iloc[weights_in_mode_idx] = b_vec.squeeze(1)

    return np.asarray(weights_info['a'].values), np.asarray(weights_info['b'].values)


def update_AB_vec(a_vec, b_vec, ref_weights, new_all_weights, weights_info):
    """
        Description: This function updates the a,b vectors.

        INPUTS
        a_vec: Array of size - number of all the weights X 1 - Previous A vector
        b_vec: Array of size - number of all the weights X 1 - Previous B vector
        weights_info: Dataframe of size - number of all the weights X 3 - contains the mode each weight is related to and two empty columns.
        new_all_weights: Matrix of size - number of all the weights X 1 - contains the value of each weight at the last epoch.
        ref_weights - Array of size - number of modes X number of epochs - contains the value of each reference weight in each epoch.

        OUTPUTS
        a_vec: Array of size - number of all the weights X 1 - contains the value of the a vector for each weight (all modes)
        b_vec: Array of size - number of all the weights X 1 - contains the value of the b vector for each weight (all modes)
    """
    # find number of new epochs
    n_epochs_new = new_all_weights.shape[1]

    full_n_epochs = ref_weights.shape[1]

    n_epochs_old = full_n_epochs - n_epochs_new

    # print(n_epochs_old, n_epochs_new, full_n_epochs)

    # a loop for each mode, using the different center weight for the a,b calculation
    for mode in tqdm(range(len(set(weights_info['related_mode'].values)))):
        # find the index of all the weights in the current mode
        weights_in_mode_idx = weights_info.index[np.where(weights_info['related_mode'].values == mode)[0]]

        # find the values of all the weights in the current mode
        weights_in_mode = np.asarray(new_all_weights[weights_in_mode_idx, :])

        # the current mode center weight values per epoch
        ref_weight_full = ref_weights[mode, :]
        ref_weight_old = ref_weight_full[0 : n_epochs_old]
        ref_weight_new = np.asarray(ref_weight_full[-n_epochs_new])
        ref_weight_new_ = np.zeros((1, n_epochs_new))
        ref_weight_new_[0, :] = ref_weight_new

        # current mode a, b calculations according to CMD method
        w_r_tilde_full = np.concatenate((np.expand_dims(ref_weight_full, 0), np.ones((1, n_epochs_new + n_epochs_old))),
                                        axis=0)
        w_r_tilde_old = np.concatenate((np.expand_dims(ref_weight_old, 0), np.ones((1, n_epochs_old))), axis=0)
        w_r_tilde_new = np.concatenate((ref_weight_new_, np.ones((1, n_epochs_new))), axis=0)

        A_tilde_old = np.asarray([a_vec[weights_in_mode_idx], b_vec[weights_in_mode_idx]]).T
        # print(A_tilde_old.shape)
        A_tilde_old_norm = A_tilde_old.dot(np.asarray(w_r_tilde_old.dot(w_r_tilde_old.T)))
        new_A_tilde = A_tilde_old_norm + weights_in_mode.dot(np.asarray(w_r_tilde_new.T))
        A_tilde = new_A_tilde.dot(pinv(w_r_tilde_full.dot(w_r_tilde_full.T)))

        a_vec_new = np.expand_dims(A_tilde[:, 0], 1)
        b_vec_new = np.expand_dims(A_tilde[:, 1], 1)

        # insert the current mode a,b values in to the all weights a,b vectors
        weights_info['a'].iloc[weights_in_mode_idx] = a_vec_new.squeeze(1)
        weights_info['b'].iloc[weights_in_mode_idx] = b_vec_new.squeeze(1)

    return np.asarray(weights_info['a'].values), np.asarray(weights_info['b'].values)


def basic_cmd(all_weights, samp_ind, start_E, end_E, clust_t=2, rate=1, clust_method='maxclust'):
    """
        Description: This function receives all the weights and the sampled weights index's and performs the full CMD algorithm.

        INPUTS
        all_weights: Matrix of size - number of all the weights X number of epochs - contains the value of each weight at each epoch.
        samp_ind: Matrix of size - number of sampled weights X 1 - contains the index of each sampled weight.
        start_E: number of epoch to start the process from.
        end_E: number of epoch to end the process at.
        clust_t: if clust_method is 'distance' then clust_t is the factor, if clust_method is 'maxclust' then clust_t is the maximum number of clusters'.
        clust_method: Clustering method. Can get 'distance' or 'maxclust'.

        OUTPUTS
        a_vec: Matrix of size - number of all the weights X 1 - contains the values of the A vector.
        b_vec: Matrix of size - number of all the weights X 1 - contains the values of the B vector.
        ref_idx: Matrix of size - number of modes X 1 - contains the index's of the center weights.
        ref_weights: Matrix of size - number of modes X number of epochs - contains the values of the center weights at each epoch.
        ref_weights: Matrix of size - number of weights X 1 - contains the value of the selceted mode for each weight.
    """

    # organize data
    tmp_all_weights = all_weights[:, start_E:end_E]
    tmp_all_weights = tmp_all_weights[:, ::rate]
    tmp_samp_weights = tmp_all_weights[samp_ind, :]

    print('number of epoch - %d' % tmp_all_weights.shape[1])

    print('searching for modes..')
    clustered_corr, idx_to_cluster_array, modes, corr_in_cluster, unclust_corr = compute_modes(tmp_samp_weights,
                                                                                               samp_ind, clust_method,
                                                                                               clust_t)
    ref_w_idx, ref_weights = find_ref_w_per_mode(corr_in_cluster, tmp_all_weights)
    print('%d modes were extracted.' % len(ref_w_idx))

    print('associating rest of the weights to the modes and perform modeling..')
    if clust_t > 1:
        related_mode = calc_related_mode(tmp_all_weights, ref_w_idx)
    else:
        related_mode = np.zeros(all_weights.shape[0])

    print('calculating AB vetors')
    weights_info = create_AB_DF(tmp_all_weights, related_mode)
    a_vec, b_vec = calc_AB_vec(weights_info, np.asarray(tmp_all_weights), ref_w_idx)

    # create outputs
    a_vec = a_vec.astype(np.float32)
    b_vec = b_vec.astype(np.float32)
    ref_idx = np.array(ref_w_idx).astype(np.float32)
    ref_weights = np.array(ref_weights).astype(np.float32)
    related_mode = weights_info['related_mode'].values.astype(np.uint8)

    print('CMD Done!')
    return a_vec, b_vec, ref_idx, ref_weights, related_mode


