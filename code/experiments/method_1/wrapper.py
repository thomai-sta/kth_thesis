#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import train_HMM
import pickle
import os
from matplotlib import pyplot as plt
import metrics
from sklearn.cross_validation import StratifiedKFold
from common_functions import iter_from_X_lengths

start_time = datetime.now()

HMM_train = "CF"

np.random.seed()
n_components = np.arange(2, 41)
target_names = ["non-AD", "AD"]
iterations = 10
folds = 3

""" Create and initialize metrics """
F1_global   = {}
F1_weighted = {}
sensitivity = {}
specificity = {}
CM = {}
fpr = {}
tpr = {}

for f in np.arange(folds):
    F1_global[f] = [0] * (n_components[-1] + 1)
    F1_weighted[f] = [0] * (n_components[-1] + 1)
    sensitivity[f] = [0] * (n_components[-1] + 1)
    specificity[f] = [0] * (n_components[-1] + 1)
    CM[f] = {}
    for n_comp in n_components:
        CM[f][n_comp] = np.zeros((2, 2))

""" Create and initialize metrics for 3 FU """
F1_global_3   = {}
F1_weighted_3 = {}
sensitivity_3 = {}
specificity_3 = {}
CM_3 = {}
fpr_3 = {}
tpr_3 = {}

for f in np.arange(folds):
    F1_global_3[f] = [0] * (n_components[-1] + 1)
    F1_weighted_3[f] = [0] * (n_components[-1] + 1)
    sensitivity_3[f] = [0] * (n_components[-1] + 1)
    specificity_3[f] = [0] * (n_components[-1] + 1)
    CM_3[f] = {}
    for n_comp in n_components:
        CM_3[f][n_comp] = np.zeros((2, 2))

""" Make train/test data """
train_obs_init, train_len_init, train_labels_init, MCI_obs, MCI_len, MCI_labels = \
    train_HMM.make_data()

starts = []
ends = []
for s, e in iter_from_X_lengths(MCI_obs, MCI_len):
    starts.append(s)
    ends.append(e)

if folds != 1:
    skf = StratifiedKFold(MCI_labels, 3)

for iteration in np.arange(iterations):
    f = 0
    for train, test in skf:
        train_len = np.append(train_len_init, MCI_len[train])
        train_labels = np.append(train_labels_init, MCI_labels[train])
        train_obs = train_obs_init.copy()
        for i in train:
            temp = MCI_obs[starts[i]:ends[i], :]
            train_obs = np.vstack((train_obs, temp))

        test_len = MCI_len[test]
        test_labels = MCI_labels[test]
        test_obs = np.zeros((0, MCI_obs.shape[1]))
        for i in test:
            temp = MCI_obs[starts[i]:ends[i], :]
            test_obs = np.vstack((test_obs, temp))

        fu_3_idx = np.where(test_len == 4)
        test_labels_3 = test_labels[fu_3_idx[0]]

        for n_comp in n_components:
            print("Iteration: %d, Components: %d, Fold: %d"
                  % (iteration, n_comp, f))
            AD_hmm, non_AD_hmm =\
                train_HMM.train_HMM(n_comp, train_obs, train_len, train_labels)

            predicted =\
                train_HMM.test_HMM(AD_hmm, non_AD_hmm, test_obs, test_len)
            predicted = np.array(predicted)
            predicted_3 = predicted[fu_3_idx[0]]

            """ Get Metrics """
            f1_global, f1_weighted = \
                metrics.get_F1_score(test_labels, predicted)
            F1_global[f][n_comp] += f1_global
            F1_weighted[f][n_comp] += f1_weighted

            cm = metrics.get_confusion_matrix(test_labels, predicted)
            CM[f][n_comp] = np.add(cm, CM[f][n_comp])

            sens = metrics.get_sensitivity(test_labels, predicted)
            sensitivity[f][n_comp] += sens

            spec = metrics.get_specificity(test_labels, predicted)
            specificity[f][n_comp] += spec

            """ Get Metrics 3 FU """
            f1_global_3, f1_weighted_3 = \
                metrics.get_F1_score(test_labels_3, predicted_3)
            F1_global_3[f][n_comp] += f1_global_3
            F1_weighted_3[f][n_comp] += f1_weighted_3

            cm_3 = metrics.get_confusion_matrix(test_labels_3, predicted_3)
            CM_3[f][n_comp] = np.add(cm_3, CM_3[f][n_comp])

            sens_3 = metrics.get_sensitivity(test_labels_3, predicted_3)
            sensitivity_3[f][n_comp] += sens_3

            spec_3 = metrics.get_specificity(test_labels_3, predicted_3)
            specificity_3[f][n_comp] += spec_3
        f += 1

""" Average all metrics """
""" make folder for CMs """
script_dir = os.path.dirname(__file__)
results_dir =\
    os.path.join(script_dir, ('CMs_' + HMM_train +  '_iters_%d_CF/') % iterations)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for f in np.arange(folds):
    F1_global[f] = [i / iterations for i in F1_global[f]]
    F1_weighted[f] = [i / iterations for i in F1_weighted[f]]
    sensitivity[f] = [i / iterations for i in sensitivity[f]]
    specificity[f] = [i / iterations for i in specificity[f]]
    for n_comp in n_components:
        CM[f][n_comp] = np.divide(CM[f][n_comp], float(iterations))
        title = ("st_%d_it_%d_HMM_" + HMM_train + "_f_%d" )\
                % (n_comp, iterations, f)

        metrics.plot_confusion_matrix(CM[f][n_comp], target_names, HMM_train,
                                      iterations, title=title)
    fpr[f] = [1 - x for x in specificity[f][n_components[0]:n_components[-1] + 1]]

    tpr[f] = [x for (y, x) in
           sorted(zip(fpr[f], sensitivity[f][n_components[0]:n_components[-1] + 1]))]

    fpr[f] = sorted(fpr[f])

    F1_global_3[f] = [i / iterations for i in F1_global_3[f]]
    F1_weighted_3[f] = [i / iterations for i in F1_weighted_3[f]]
    sensitivity_3[f] = [i / iterations for i in sensitivity_3[f]]
    specificity_3[f] = [i / iterations for i in specificity_3[f]]
    for n_comp in n_components:
        CM_3[f][n_comp] = np.divide(CM_3[f][n_comp], float(iterations))
        title_3 = ("st_%d_it_%d_fu_3_HMM_" + HMM_train + "_f_%d") %(n_comp, iterations, f)
        metrics.plot_confusion_matrix(CM_3[f][n_comp], target_names, HMM_train,
                                      iterations, title=title_3)
    fpr_3[f] = [1 - x for x in specificity_3[f][n_components[0]:n_components[-1] + 1]]

    tpr_3[f] = [x for (y, x) in
             sorted(zip(fpr_3[f], sensitivity_3[f][n_components[0]:n_components[-1] + 1]))]

    fpr_3[f] = sorted(fpr_3[f])

# with open("metrics.pickle", "wb") as f:
#     pickle.dump(
#         [HMM_train, SVM_train, volumetric, n_components, folds, iterations, fpr,
#          tpr, F1_training, F1_global, F1_weighted, sensitivity, specificity,
#          F1_global_3, F1_weighted_3, sensitivity_3, specificity_3, fpr_3,
#          tpr_3], f)
with open("metrics.pickle", "wb") as f:
    pickle.dump([HMM_train, n_components, iterations, fpr, tpr, F1_global,
         F1_weighted, sensitivity, specificity, F1_global_3, F1_weighted_3,
         sensitivity_3, specificity_3, fpr_3, tpr_3, folds], f)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))