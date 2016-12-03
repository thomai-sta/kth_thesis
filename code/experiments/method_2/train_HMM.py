#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import common_functions as cf
from common_functions import iter_from_X_lengths


""" CONCEPT:
        Train HMMs
        Use state sequences to train SVM
        Test how well it predicts end group """


def make_data():
    """ Read in data """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data/for_hmm.pickle"), "rb") as f:
        [observations, lengths, subject_group, subject_end_group, _, _,
         subject_idxs] = pickle.load(f)

    """ Prepare training/testing data """
    train_idx = np.where(subject_group != 1)  # Don't use MCI for training
    train_obs = np.zeros((0, observations.shape[1]))
    train_len = lengths[train_idx[0]]
    train_labels = subject_end_group[train_idx[0]]
    train_labels[train_labels == 1] = 0
    train_labels[train_labels == 2] = 1
    for i in train_idx[0]:
        temp =\
            observations[subject_idxs[i]:(subject_idxs[i] + lengths[i])][:]
        train_obs = np.vstack((train_obs, temp))


    test_idx = np.where(subject_group == 1)  # Use MCI for testing
    test_obs = np.zeros((0, observations.shape[1]))
    test_len = lengths[test_idx[0]]
    test_labels = subject_end_group[test_idx[0]]
    test_labels[test_labels == 1] = 0
    test_labels[test_labels == 2] = 1
    for i in test_idx[0]:
        temp =\
            observations[subject_idxs[i]:(subject_idxs[i] + lengths[i])][:]
        test_obs = np.vstack((test_obs, temp))

    return train_obs, train_len, train_labels, test_obs, test_len, test_labels


def train_HMM(n_components, train_obs, train_len, test_obs, test_len, verbose=False,
              cov_type='spherical', iter=500):

    h0 = cf.setup_and_train(n_components, train_obs, train_len, cov_type,
                            verbose, iter)
    train_state_seq = h0.predict(train_obs, train_len)
    train_seq = cf.vecs2mats(train_state_seq, train_len)

    test_state_seq = h0.predict(test_obs, test_len)
    test_seq = cf.vecs2mats(test_state_seq, test_len)

    return train_seq, test_seq

