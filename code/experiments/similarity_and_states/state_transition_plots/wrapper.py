#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import train_HMM
import pickle


start_time = datetime.now()

np.random.seed()
max_comp = 21
n_components = np.arange(2, max_comp)

sequences = {}

""" Make train/test data """
# train HMM only on non-MCI
train_obs, train_len, observations, lengths, labels, groups =\
    train_HMM.make_data()

for n_comp in n_components:
    print("Components: %d" % n_comp)
    sequences[n_comp] = train_HMM.train_HMM(n_comp, train_obs, train_len,
                                            observations, lengths)

with open("from_hmm.pickle", "wb") as f:
    pickle.dump([n_components, train_obs, train_len, observations, lengths,
                 labels, groups, sequences], f)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
