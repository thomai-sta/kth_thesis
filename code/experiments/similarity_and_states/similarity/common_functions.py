#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from hmmlearn import hmm


def vec2mat(vector, lengths):
    matrix = np.zeros((0, np.max(lengths)))
    start = 0
    end = lengths[0]
    for i in range(len(lengths)):
        matrix= np.vstack((matrix, -np.ones((1, np.max(lengths)))))
        matrix[-1, :len(vector[start:end])] = vector[start:end]

        start += lengths[i]
        if i < len(lengths) - 1:
            end += lengths[i + 1]

    return matrix


def setup_and_train(n_components, train_obs, train_len, cov_type, verbose, iter, params):
    """ Fully connected """
    initial_transmat = np.random.random((n_components, n_components))
    initial_transmat = initial_transmat / np.sum(initial_transmat, axis=1)[:,
                                          None]

    """ Start at random state """
    initial_startprob = np.zeros(n_components)
    initial_startprob[:] = 1.0 / n_components

    h = hmm.GaussianHMM(n_components=n_components, covariance_type=cov_type,
                        params=params, init_params="mc", n_iter=iter,
                        verbose=verbose)
    h.transmat_ = initial_transmat.copy()
    h.startprob_ = initial_startprob.copy()
    h.fit(train_obs, train_len)

    return h


def vote_states(sequences, test_len, n_components):
    iterations = sequences.shape[1]

    votes = -np.ones((len(test_len), n_components, np.max(test_len)))
    print votes.shape
    for i in np.arange(iterations):
        state_seq = vec2mat(sequences[:, i], test_len)
        for sub, seq in enumerate(state_seq):
            for step in np.arange(test_len[sub]):
                votes[sub, seq[step], step] += 1

    """ All have voted """
    final_sequences = - np.ones((len(test_len), np.max(test_len)))
    for sub in np.arange(len(test_len)):
        temp = votes[sub, :, :test_len[sub]]
        final_sequences[sub, :test_len[sub]] = np.argmax(temp, axis=0)

    return final_sequences
