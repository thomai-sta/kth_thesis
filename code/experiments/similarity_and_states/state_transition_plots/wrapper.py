#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import train_HMM
import plot_states
import plot_transitions

start_time = datetime.now()

np.random.seed()
max_comp = 21
n_components = np.arange(2, max_comp)

""" Make train/test data """
# train HMM only on non-MCI
train_obs, train_len, observations, lengths, labels, groups = train_HMM.make_data()

for n_comp in n_components:
    print("Components: %d" % n_comp)
    sequences = train_HMM.train_HMM(n_comp, train_obs, train_len, observations, lengths)
    plot_states.plot_states(sequences, lengths, groups, labels, n_comp)
    plot_transitions.plot_transition(n_comp, groups, labels, sequences, lengths)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))