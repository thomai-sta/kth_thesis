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


def make_data(max_components):
    """ Read in data """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data/for_hmm.pickle"), "rb") as f:
        [observations, lengths, subject_group, subject_end_group, _, _,
         subject_idxs] = pickle.load(f)

    labels = subject_end_group
    labels[labels == 1] = 0
    labels[labels == 2] = 1

    """ Create Bagged dataset """
    lengths_new = []
    lengths_new.append(lengths[0])

    """ Implementing bagging to check """
    start = 0
    end = lengths[0]
    new_observations = np.empty(shape=(0, observations.shape[1]))
    for i in range(len(lengths)):
        curr_obs = observations[start:end, :]
        curr_len = lengths[i]
        start += curr_len
        if i < len(lengths) - 1:
            next_len = lengths[i + 1]
            end += next_len

        new_obs = np.random.permutation(curr_obs)
        new_observations = np.vstack((new_observations, new_obs))


    return observations, new_observations, lengths, labels


def train_HMM(n_components, observations, lengths, verbose=False,
              cov_type='spherical', iter=500, params="mcts"):

    h0 = cf.setup_and_train(n_components, observations, lengths, cov_type,
                            verbose, iter, params=params)
    return h0

