#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import train_HMM
import pickle
import os
import metrics
from sklearn.cross_validation import StratifiedKFold
from common_functions import iter_from_X_lengths
from svm_stuff import svm_cross_validation

start_time = datetime.now()

HMM_train = "CV"

np.random.seed()
n_components = np.arange(2, 41)
target_names = ["non-AD", "AD"]
iterations = 4
SVM_folds = [5, 7, 10]
folds = 3

""" Create and initialize metrics """
sensitivity = {}
specificity = {}
CM = {}
fpr = {}
tpr = {}

for f in SVM_folds:
    sensitivity[f] = [0] * (n_components[-1] + 1)
    specificity[f] = [0] * (n_components[-1] + 1)
    CM[f] = {}
    for n_comp in n_components:
        CM[f][n_comp] = np.zeros((2, 2))

""" Create and initialize metrics for 3 FU """
sensitivity_3 = {}
specificity_3 = {}
CM_3 = {}
fpr_3 = {}
tpr_3 = {}

for f in SVM_folds:
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
else:
    skf = zip([-1], [-1])

for iteration in np.arange(iterations):
    f = 0
    for train, test in skf:
        if np.any(train == -1):
            train_len = train_len_init
            train_labels = train_labels_init
            train_obs = train_obs_init.copy()

            test_len = MCI_len
            test_labels = MCI_labels
            test_obs = MCI_obs.copy()
        else:
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
            train_data, test_data =\
                train_HMM.train_HMM(n_comp, train_obs, train_len, test_obs,
                                    test_len)

            for svm_f in SVM_folds:
                print("Iteration: %d, Components: %d, Fold: %d, SVM Fold: %d"
                      % (iteration, n_comp, f, svm_f))
                predicted = svm_cross_validation(train_data, train_labels, test_data, svm_f)
                predicted = np.array(predicted)
                predicted_3 = predicted[fu_3_idx[0]]

                """ Get Metrics """
                cm = metrics.get_confusion_matrix(test_labels, predicted)
                CM[svm_f][n_comp] = np.add(cm, CM[svm_f][n_comp])

                sens = metrics.get_sensitivity(test_labels, predicted)
                sensitivity[svm_f][n_comp] += sens

                spec = metrics.get_specificity(test_labels, predicted)
                specificity[svm_f][n_comp] += spec

                """ Get Metrics 3 FU """
                cm_3 = metrics.get_confusion_matrix(test_labels_3, predicted_3)
                CM_3[svm_f][n_comp] = np.add(cm_3, CM_3[svm_f][n_comp])

                sens_3 = metrics.get_sensitivity(test_labels_3, predicted_3)
                sensitivity_3[svm_f][n_comp] += sens_3

                spec_3 = metrics.get_specificity(test_labels_3, predicted_3)
                specificity_3[svm_f][n_comp] += spec_3
        f += 1

""" Average all metrics """
""" make folder for CMs """
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, ('results' + HMM_train + '_iters_%d/')
                           % iterations)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for svm_f in SVM_folds:
    sensitivity[svm_f] = [i / (iterations * folds) for i in sensitivity[svm_f]]
    specificity[svm_f] = [i / (iterations * folds) for i in specificity[svm_f]]
    for n_comp in n_components:
        CM[svm_f][n_comp] = np.divide(CM[svm_f][n_comp], float(iterations * folds))
        title = ("st_%d_it_%d_HMM_" + HMM_train + "_svmf_%d" )\
                % (n_comp, iterations, svm_f)

        metrics.plot_confusion_matrix(CM[svm_f][n_comp], target_names,
                                      iterations, results_dir=results_dir,
                                      title=title)
    fpr[svm_f] = [1 - x for x in specificity[svm_f][n_components[0]:n_components[-1] + 1]]

    tpr[svm_f] = [x for (y, x) in
           sorted(zip(fpr[svm_f], sensitivity[svm_f][n_components[0]:n_components[-1] + 1]))]

    fpr[svm_f] = sorted(fpr[svm_f])

    sensitivity_3[svm_f] = [i / (iterations * folds) for i in sensitivity_3[svm_f]]
    specificity_3[svm_f] = [i / (iterations * folds) for i in specificity_3[svm_f]]
    for n_comp in n_components:
        CM_3[svm_f][n_comp] = np.divide(CM_3[svm_f][n_comp], float(iterations * folds))
        title_3 = ("st_%d_it_%d_fu_3_HMM_" + HMM_train + "_svmf_%d")\
                  %(n_comp, iterations, svm_f)
        metrics.plot_confusion_matrix(CM_3[svm_f][n_comp], target_names,
                                      iterations, results_dir=results_dir,
                                      title=title_3)
    fpr_3[svm_f] = [1 - x for x in specificity_3[svm_f][n_components[0]:n_components[-1] + 1]]

    tpr_3[svm_f] = [x for (y, x) in
             sorted(zip(fpr_3[svm_f], sensitivity_3[svm_f][n_components[0]:n_components[-1] + 1]))]

    fpr_3[svm_f] = sorted(fpr_3[svm_f])

with open(results_dir + "metrics.pickle", "wb") as f:
    pickle.dump([HMM_train, n_components, iterations, fpr, tpr, sensitivity,
                 specificity, sensitivity_3, specificity_3, fpr_3, tpr_3,
                 folds, SVM_folds], f)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))