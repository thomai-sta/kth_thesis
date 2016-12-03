#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import common_functions as cf
from common_functions import iter_from_X_lengths


""" CONCEPT:
        Train HMMs for the different "end" groups (AD - nonAD)
        Test how well they specialize """


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

def train_HMM(n_components, observations, lengths, labels, verbose=False,
              cov_type='spherical', iter=500):
    AD_idx = np.where(labels == 1)
    AD_lengths = lengths[AD_idx[0]]

    nonAD_idx = np.where(labels != 1)
    nonAD_lengths = lengths[nonAD_idx[0]]

    l = np.nditer(labels)
    nonAD_observations = np.zeros((0, observations.shape[1]))
    AD_observations = np.zeros((0, observations.shape[1]))
    for s, e, in iter_from_X_lengths(observations, lengths):
        temp = observations[s:e, :]
        if l.next() == 0:
            nonAD_observations = np.vstack((nonAD_observations, temp))
        else:
            AD_observations = np.vstack((AD_observations, temp))

    AD_hmm = cf.setup_and_train(n_components, AD_observations, AD_lengths,
                                cov_type, verbose, iter)

    nonAD_hmm = cf.setup_and_train(n_components, nonAD_observations,
                                   nonAD_lengths, cov_type, verbose, iter)

    return AD_hmm, nonAD_hmm


def test_HMM(ad_hmm, non_ad_hmm, observations, lengths):
    """ Test observation probabilities for both HMMs and decide on label """
    predicted = []
    for start, end in iter_from_X_lengths(observations, lengths):
        obs = observations[start:end, :]
        ad_score = ad_hmm.score(obs)
        non_ad_score = non_ad_hmm.score(obs)

        if ad_score > non_ad_score:
            predicted.append(1)
        else:
            predicted.append(0)

    return predicted