# -*- coding: utf-8 -*-

# http://www.ece.tamu.edu/~bjyoon/journal/IEEE-SPL-2010-HMM-similarity.pdf

import numpy as np
import kld
from discreteMarkovChain import markovChain


def get_stationary(transmat):
    # P = np.array([[0.5, 0.5], [0.6, 0.4]])
    # mc = markovChain(transmat)
    # mc.computePi('eigen')  # We can also use 'power', 'krylov' or 'eigen'
    # # print(mc.pi)
    # return mc.pi
    w, v = np.linalg.eig(transmat.T)

    j_stationary = np.argmin(abs(w - 1.0))
    p_stationary = v[:, j_stationary].real
    p_stationary /= p_stationary.sum()
    return p_stationary

    # S, U = np.linalg.eig(transmat.T)
    # stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    # stationary = stationary / np.sum(stationary)
    # return np.real(stationary)


def gini(v):
    N = len(v)
    if N == 1:
        return 0
    v_sorted = np.sort(v)
    v_norm = np.linalg.norm(v, ord=1)
    gini_sum = 0

    for k in np.arange(N):
        # k-th smallest element
        gini_sum += v_sorted[k] / v_norm * (N - (k + 1) + 1 / 2) / (N- 1)

    return N / (N - 1) - 2 * gini_sum


def hmm_similarity(h1, h2):
    n_components = h1.n_components

    # 1. Compute stationary distributions π and π'
    stationary1 = get_stationary(h1.transmat_)
    stationary2 = get_stationary(h2.transmat_)

    # 2. Compute D(bi||bi') for every pair of states
    # 3. Evaluate Se(vi, vi')
    D = np.zeros((n_components, n_components))
    Se = np.zeros((n_components, n_components))
    if hasattr(h1, 'means_'):
        # Gaussian HMMs
        for state_1 in np.arange(n_components):
            m1 = h1.means_[state_1][:]
            v1 = h1.covars_[state_1][:][:]
            for state_2 in np.arange(n_components):
                m2 = h2.means_[state_2][:]
                v2 = h2.covars_[state_2][:][:]
                D[state_1][state_2] = \
                    0.5 * (kld.kld_continuous(m1, v1, m2, v2) +
                           kld.kld_continuous(m2, v2, m1, v1))
                Se[state_1][state_2] = 1 / D[state_1][state_2]
    else:
        # Multinomial HMMs
        for state_1 in np.arange(n_components):
            pk = h1.emissionprob_[state_1][:]
            for state_2 in np.arange(n_components):
                qk = h2.emissionprob_[state_2][:]
                D[state_1][state_2] = \
                    0.5 * (kld.kld_discrete(pk, qk) + kld.kld_discrete(qk, pk))
                Se[state_1][state_2] = 1 / D[state_1][state_2]

    # 4. Estimate the state correspondence matrix Q
    # Calculate ES(λ, λ')
    ES = 0
    for state_1 in np.arange(n_components):
        for state_2 in np.arange(n_components):
            ES += stationary1[state_1] * stationary2[state_2] \
                  * Se[state_1][state_2]

    # Fill Q
    Q = np.zeros((n_components, n_components))
    for state_1 in np.arange(n_components):
        for state_2 in np.arange(n_components):
            Q[state_1][state_2] = \
                stationary1[state_1] * stationary2[state_2] \
                * Se[state_1][state_2] / ES

    # 5. Compute the HMM similarity measure S(λ||λ')
    # Calculate Gini's for rows and columns
    rows_sum = 0
    cols_sum = 0
    for idx in np.arange(n_components):
        rows_sum += gini(Q[:][idx])
        cols_sum += gini(Q[idx][:])

    S = 0.5 * (rows_sum / n_components + cols_sum / n_components)
    return S


def find_duplicates(records_array):
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                       return_index=True)

    # sets of indices
    res = np.split(idx_sort, idx_start[1:])
    # filter them with respect to their size, keeping only items occurring more than
    # once

    # vals = vals[count > 1] # duplicate values
    res = filter(lambda x: x.size > 1, res)
    return res


# def state_correspondence(h0, h, previous_votes=0):
def state_correspondence(h0, h):
    """ Calculate correspondence of h's states to h0's states"""
    n_components = h0.n_components

    """ 1. Compute stationary distributions π and π' """
    # stationary1 = get_stationary(h0.transmat_)
    stationary1 = get_stationary(h0.transmat_)
    stationary2 = get_stationary(h.transmat_)

    """ 2. Compute D(bi||bi') for every pair of states """
    """ 3. Evaluate Se(vi, vi') """
    D = np.zeros((n_components, n_components))
    Se = np.zeros((n_components, n_components))
    if hasattr(h0, 'means_'):
        # Gaussian HMMs
        for state_1 in np.arange(n_components):
            m1 = h0.means_[state_1][:]
            v1 = h0.covars_[state_1][:][:]
            for state_2 in np.arange(n_components):
                m2 = h.means_[state_2][:]
                v2 = h.covars_[state_2][:][:]
                D[state_1][state_2] = \
                    0.5 * (kld.kld_continuous(m1, v1, m2, v2) +
                           kld.kld_continuous(m2, v2, m1, v1))
                Se[state_1][state_2] = 1 / D[state_1][state_2]
    else:
        # Multinomial HMMs
        for state_1 in np.arange(n_components):
            pk = h0.emissionprob_[state_1][:]
            for state_2 in np.arange(n_components):
                qk = h.emissionprob_[state_2][:]
                D[state_1][state_2] = \
                    0.5 * (kld.kld_discrete(pk, qk) + kld.kld_discrete(qk, pk))
                Se[state_1][state_2] = 1 / D[state_1][state_2]

    """ 4. Estimate the state correspondence matrix Q """
    # Calculate ES(λ, λ')
    ES = 0
    for state_1 in np.arange(n_components):
        for state_2 in np.arange(n_components):
            ES += stationary1[state_1] * stationary2[state_2] \
                  * Se[state_1][state_2]

    # Fill Q
    Q = np.zeros((n_components, n_components))
    for state_1 in np.arange(n_components):
        for state_2 in np.arange(n_components):
            Q[state_1][state_2] = \
                stationary1[state_1] * stationary2[state_2] \
                * Se[state_1][state_2] / ES

    # Q: rows are states of h0 nd cols are states of h, so transpose
    Q = Q.T
    print Q
    """ Get state correspondences """
    # Correspondence[i]: state i of h connects to correspondence[i] of h0
    correspondence = np.zeros(n_components)
    for i, row in enumerate(Q):
        correspondence[i] = np.argmax(row)

    # duplicates = find_duplicates(correspondence)
    # if duplicates:
    #     for duplicate in duplicates:
    #
    return correspondence