#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import train_HMM
import hmm_similarity
from matplotlib import pyplot as plt

start_time = datetime.now()

np.random.seed()
max_comp = 41
n_components = np.arange(1, max_comp)
target_names = ["non-AD", "AD"]
iterations = 20

""" Make train/test data """
observations, bag_obs, lengths, labels = train_HMM.make_data(max_comp)

observations_rand = observations[:lengths[0]][:]
lengths_rand = []
lengths_rand.append(lengths[0])
l = 0

similarities = np.zeros(max_comp - 1)
counter = np.zeros(max_comp - 1)
similarities_bag = np.zeros(max_comp - 1)
bag_counter = np.zeros(max_comp - 1)
similarities_bag_rand = np.zeros(max_comp - 1)
bag_rand_counter = np.zeros(max_comp - 1)

for iteration in np.arange(iterations):
    for n_comp in n_components:
        print("Iteration: %d, Components: %d" %(iteration, n_comp))
        """ Create normal HMM """
        print("Training Regular")
        h = train_HMM.train_HMM(n_comp, observations, lengths)
        # Check for nan's
        check = np.isnan(h.startprob_)
        if check.any():
            print("GOT NANS!!!!!!")

        """ Create Bagged HMM """
        print("Training Bagged")
        h_bag = train_HMM.train_HMM(n_comp, bag_obs, lengths)  # h is an hmm trained on the "randomized data"
        # Check for nan's
        check_bag = np.isnan(h_bag.startprob_)
        if check_bag.any():
            print("GOT NANS!!!!!!")

        """ Create a random HMM for comparison """
        print("Training Random")
        if np.sum(lengths_rand) < n_comp:
            # Get a new observation
            l += 1
            temp = np.sum(lengths[:(l + 1)])
            observations_rand = observations[:temp][:]
            lengths_rand.append(lengths[l])

        h_rand = train_HMM.train_HMM(n_comp, observations_rand, lengths_rand, iter=1, params="mc")
        # Check for nan's
        check_rand = np.isnan(h_rand.startprob_)
        if check_rand.any():
            print("GOT NANS!!!!!!")

        """ Measure similarity """
        s_reg_rand = hmm_similarity.hmm_similarity(h, h_rand)
        if ~np.isnan(s_reg_rand):
            similarities[n_comp - 1] += s_reg_rand
            counter[n_comp - 1] += 1

        s_bag_rand = hmm_similarity.hmm_similarity(h_bag, h_rand)
        if ~np.isnan(s_bag_rand):
            similarities_bag_rand[n_comp - 1] += s_bag_rand
            bag_rand_counter[n_comp - 1] += 1

        s_reg_bag = hmm_similarity.hmm_similarity(h, h_bag)
        if ~np.isnan(s_reg_bag):
            similarities_bag[n_comp - 1] += s_reg_bag
            bag_counter[n_comp - 1] += 1


similarities /= (counter)

X = np.arange(1, max_comp)
plt.plot(X, similarities, label="regular with random", linewidth=2.0)

similarities_bag /= (bag_counter)
plt.plot(X, similarities_bag, label="regular with bagged", linewidth=2.0)

similarities_bag_rand /= (bag_rand_counter)
plt.plot(X, similarities_bag_rand, label="bagged with random", linewidth=2.0)

plt.title("Similarities Between Different HMMs")
plt.legend(loc=0)

plt.grid()
plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
