#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def plot_states(sequences, lengths, groups, labels, n_components):
    """ Print the MCI """
    MCI_idx = np.where(groups == 1)
    MCI_groups = labels[MCI_idx[0]]
    MCI_sequences = sequences[MCI_idx[0], :]
    MCI_lengths = lengths[MCI_idx[0]]

    colours = ['g', 'r']

    for i, seq in enumerate(MCI_sequences):
        x = np.arange(MCI_lengths[i])
        if MCI_groups[i] == 0:
            # non-AD group
            plt.subplot(211)
            plt.axis([0, 3, 0, n_components - 1])
            plt.title('non-AD end-group: %d' %(len(np.where(MCI_groups == 0)[0])))
            plt.plot(x, seq[:MCI_lengths[i]], colours[MCI_groups[i]])
        elif MCI_groups[i] == 1:
            # AD group
            plt.subplot(212)
            plt.axis([0, 3, 0, n_components - 1])
            plt.title('AD end-group: %d' %(len(np.where(MCI_groups == 1)[0])))
            plt.plot(x, seq[:MCI_lengths[i]], colours[MCI_groups[i]])

    plt.suptitle("MCI Group state Sequences\nTotal Subjects: %d" %(len(MCI_groups)))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())

    to_save = plt.gcf()

    to_save.savefig('mci/states_%d.png' %(n_components))

    """ Print the non-MCI """
    plt.figure()
    NON_MCI_idx = np.where(groups != 1)
    NON_MCI_groups = labels[NON_MCI_idx[0]]
    NON_MCI_sequences = sequences[NON_MCI_idx[0], :]
    NON_MCI_lengths = lengths[NON_MCI_idx[0]]

    colours = ['g', 'r']

    for i, seq in enumerate(NON_MCI_sequences):
        x = np.arange(NON_MCI_lengths[i])
        if NON_MCI_groups[i] == 0:
            # non-AD group
            plt.subplot(211)
            plt.axis([0, 3, 0, n_components - 1])
            plt.title('non-AD end-group: %d' %(len(np.where(NON_MCI_groups == 0)[0])))
            plt.plot(x, seq[:NON_MCI_lengths[i]], colours[NON_MCI_groups[i]])
        elif NON_MCI_groups[i] == 1:
            # AD group
            plt.subplot(212)
            plt.axis([0, 3, 0, n_components - 1])
            plt.title('AD end-group: %d' %(len(np.where(NON_MCI_groups == 1)[0])))
            plt.plot(x, seq[:NON_MCI_lengths[i]], colours[NON_MCI_groups[i]])

    plt.suptitle("non-MCI Group state Sequences\nTotal Subjects: %d" %(len(NON_MCI_groups)))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())

    to_save = plt.gcf()

    to_save.savefig('non-mci/states_%d.png' %(n_components))
