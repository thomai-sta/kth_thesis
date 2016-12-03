#!/usr/local/bin/python
#  -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def plot_transition(n_components, groups, labels, sequences, lengths):
    """ Produce Transition maps """
    MCI_idx = np.where(groups == 1)
    MCI_labels = labels[MCI_idx[0]]
    MCI_seq = sequences[MCI_idx[0]][:]
    MCI_len = lengths[MCI_idx[0]]
    MCI_total_trans = 0
    MCI_transitions_AD = np.zeros((n_components, n_components))
    MCI_transitions_nonAD = np.zeros((n_components, n_components))
    MCI_seq = MCI_seq.astype(int)
    for i, state_seq in enumerate(MCI_seq):
        for step in np.arange(1, MCI_len[i]):
            MCI_total_trans += 1
            if MCI_labels[i] == 0:
                MCI_transitions_nonAD[MCI_seq[i, step - 1], MCI_seq[i, step]] += 1
            else:
                MCI_transitions_AD[MCI_seq[i, step - 1], MCI_seq[i, step]] += 1

    MCI_transitions_nonAD /= MCI_total_trans
    MCI_transitions_AD /= MCI_total_trans

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Transition Maps for the MCI subjects', fontsize=20)

    ax1.set_title('non-AD end-group')
    im1 = ax1.matshow(MCI_transitions_nonAD, cmap='BrBG')
    plt.colorbar(im1, ax=ax1)

    ax2.set_title('AD end-group')
    im2 = ax2.matshow(MCI_transitions_AD, cmap='BrBG')
    plt.colorbar(im2, ax=ax2)

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())

    to_save = plt.gcf()
    to_save.savefig('transitions_mci/transitions_%d_states.png' %(n_components))

    non_MCI_idx = np.where(groups == 0)
    non_MCI_labels = labels[non_MCI_idx[0]]
    non_MCI_seq = sequences[non_MCI_idx[0]][:]
    non_MCI_len = lengths[non_MCI_idx[0]]
    non_MCI_total_trans = 0
    non_MCI_transitions_AD = np.zeros((n_components, n_components))
    non_MCI_transitions_nonAD = np.zeros((n_components, n_components))
    non_MCI_seq = non_MCI_seq.astype(int)
    for i, state_seq in enumerate(non_MCI_seq):
        for step in np.arange(1, non_MCI_len[i]):
            non_MCI_total_trans += 1
            if non_MCI_labels[i] == 0:
                non_MCI_transitions_nonAD[non_MCI_seq[i, step - 1], non_MCI_seq[i, step]] += 1
            else:
                non_MCI_transitions_AD[non_MCI_seq[i, step - 1], non_MCI_seq[i, step]] += 1

    non_MCI_transitions_nonAD /= non_MCI_total_trans
    non_MCI_transitions_AD /= non_MCI_total_trans

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Transition Maps for the non-MCI subjects', fontsize=20)

    ax1.set_title('non-AD end-group')
    im1 = ax1.matshow(non_MCI_transitions_nonAD, cmap='BrBG')
    plt.colorbar(im1, ax=ax1)

    ax2.set_title('AD end-group')
    im2 = ax2.matshow(non_MCI_transitions_AD, cmap='BrBG')
    plt.colorbar(im2, ax=ax2)

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())

    to_save = plt.gcf()
    # plt.show()

    to_save.savefig('transitions_non-mci/transitions_%d_states.png' %(n_components))