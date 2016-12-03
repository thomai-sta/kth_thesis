#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from hmmlearn import hmm


def random_ints_with_sum(n):
    np.random.seed(42)
    """
    Generate non-negative random integers summing to `n`.
    """
    while n > 0:
        if n == 1:
            yield 1
        else:
            r = np.random.randint(1, n)
            yield r
        n -= r


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


def vecs2mats(vector, lengths):
    matrix = np.zeros((0, np.max(lengths)))

    for start, end in  iter_from_X_lengths(vector, lengths):
        matrix= np.vstack((matrix, -np.ones((1, np.max(lengths)))))
        matrix[-1, :(end - start)] = vector[start:end]

    return matrix


def setup_and_train(n_components, train_obs, train_len, cov_type, verbose, iter):
    """ Fully connected """
    initial_transmat = np.random.random((n_components, n_components))
    initial_transmat = initial_transmat / np.sum(initial_transmat, axis=1)[:,
                                          None]

    """ Start at random state """
    initial_startprob = np.zeros(n_components)
    initial_startprob[:] = 1.0 / n_components

    h = hmm.GaussianHMM(n_components=n_components, covariance_type=cov_type,
                        params="mcts", init_params="mc", n_iter=iter,
                        verbose=verbose)
    h.transmat_ = initial_transmat.copy()
    h.startprob_ = initial_startprob.copy()
    print("Training HMM...")
    h.fit(train_obs, train_len)
    print("Converged: %r" % h.monitor_.converged)

    return h


def vote_states(sequences, test_len, n_components):
    iterations = sequences.shape[1]

    votes = -np.ones((len(test_len), n_components, np.max(test_len)))
    for i in np.arange(iterations):
        state_seqs = vecs2mats(sequences[:, i], test_len)
        for sub, seq in enumerate(state_seqs):
            for step in np.arange(test_len[sub]):
                votes[sub, seq[step], step] += 1

    """ All have voted """
    final_sequences = - np.ones((len(test_len), np.max(test_len)))
    for sub in np.arange(len(test_len)):
        temp = votes[sub, :, :test_len[sub]]
        final_sequences[sub, :test_len[sub]] = np.argmax(temp, axis=0)

    return final_sequences
