#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import common_functions as cf

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

    labels = subject_end_group
    labels[labels == 1] = 0
    labels[labels == 2] = 1

    groups = subject_group
    groups[groups == 2] = 0

    """ Training data: non-MCI """
    train_idx = np.where(subject_group != 1)  # Don't use MCI for training
    train_obs = np.zeros((0, observations.shape[1]))
    train_len = lengths[train_idx[0]]
    for i in train_idx[0]:
        temp =\
            observations[subject_idxs[i]:(subject_idxs[i] + lengths[i])][:]
        train_obs = np.vstack((train_obs, temp))

    return train_obs, train_len, observations, lengths, labels, groups


def train_HMM(n_comp, train_obs, train_len, observations, lengths, verbose=True,
              cov_type='spherical', iter=500):
    h0 = cf.setup_and_train(n_comp, train_obs, train_len, cov_type,
                            verbose, iter)

    sequences = h0.predict(observations, lengths)
    seq = cf.vec2mat(sequences, lengths)

    return seq

