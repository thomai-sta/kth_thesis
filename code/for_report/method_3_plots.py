#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from scipy.spatial import distance

""" Check if we already have the metrics file """
file = "method_3.pickle"
if os.path.isfile(file):
    [_, n_components, _, fpr, tpr, sensitivity,
     specificity, sensitivity_3, specificity_3, fpr_3, tpr_3,
     folds, SVM_folds] =\
        pickle.load(open(file, "rb"))

file_CV = "method_3_CV.pickle"
if os.path.isfile(file_CV):
    [_, n_components, _, fpr_CV, tpr_CV, sensitivity_CV,
     specificity_CV, sensitivity_3_CV, specificity_3_CV, fpr_3_CV, tpr_3_CV,
     folds, SVM_folds] = pickle.load(open(file_CV, "rb"))

""" Results are already averaged """
print("CV\t\tSVMf\t\t1 - fpr\t\ttpr\t\t\tdist\t\tAUC\t\t\t3FU")

"""  Plot ROC curves """
roc = plt.figure()
roc.hold(True)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.grid()
x = [0, 1]
y = [0, 1]
plt.plot(x, y, '--', label="Random", linewidth=2.0)
# no CV
for svmf in SVM_folds:
    fpr[svmf].insert(0, 0.0)
    fpr[svmf].append(1.0)
    tpr[svmf].insert(0, 0.0)
    tpr[svmf].append(1.0)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_tpr = np.interp(mean_fpr, fpr[svmf], tpr[svmf])
    roc_auc = auc(mean_fpr, mean_tpr)
    # Get best point
    dist = 100
    best_x = best_y = 0
    for f, t in zip(mean_fpr, mean_tpr):
        d = distance.euclidean((0, 1), (f, t))
        if d < dist:
            dist = d
            best_x = f
            best_y = t
    plt.plot(mean_fpr, mean_tpr, label="no CV, %d SVM folds: %f" % (svmf, roc_auc), linewidth=2.0)
    print("NO\t\t%d\t\t\t%f\t%f\t%f\t%f\tNO" % (svmf, 1 - best_x, best_y, dist, roc_auc))

    # CV
    fpr_CV[svmf].insert(0, 0.0)
    fpr_CV[svmf].append(1.0)
    tpr_CV[svmf].insert(0, 0.0)
    tpr_CV[svmf].append(1.0)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_tpr = np.interp(mean_fpr, fpr_CV[svmf], tpr_CV[svmf])
    roc_auc = auc(mean_fpr, mean_tpr)
    # Get best point
    dist = 100
    best_x_CV = best_y_CV = 0
    for f, t in zip(mean_fpr, mean_tpr):
        d = distance.euclidean((0, 1), (f, t))
        if d < dist:
            dist = d
            best_x_CV = f
            best_y_CV = t
    plt.plot(mean_fpr, mean_tpr, label="with CV, %d SVM folds: %f" % (svmf, roc_auc), linewidth=2.0)
    print("YES\t\t%d\t\t\t%f\t%f\t%f\t%f\tNO" % (svmf, 1 - best_x_CV, best_y_CV, dist, roc_auc))

plt.legend(loc=0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Method 3 with & without Cross-Validation")


"""  Plot ROC curves for 3 Follow-Ups """
roc = plt.figure()
roc.hold(True)
plt.axis('scaled')
plt.axis([0, 1, 0, 1])
plt.grid()
x = [0, 1]
y = [0, 1]
plt.plot(x, y, '--', label="Random", linewidth=2.0)
# no CV
for svmf in SVM_folds:
    fpr_3[svmf].insert(0, 0.0)
    fpr_3[svmf].append(1.0)
    tpr_3[svmf].insert(0, 0.0)
    tpr_3[svmf].append(1.0)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_tpr = np.interp(mean_fpr, fpr_3[svmf], tpr_3[svmf])
    roc_auc = auc(mean_fpr, mean_tpr)
    # Get best point
    dist = 100
    best_x_3 = best_y_3 = 0
    for f, t in zip(mean_fpr, mean_tpr):
        d = distance.euclidean((0, 1), (f, t))
        if d < dist:
            dist = d
            best_x_3 = f
            best_y_3 = t
    plt.plot(mean_fpr, mean_tpr, label="no CV, %d SVM folds: %f" % (svmf, roc_auc), linewidth=2.0)
    print("NO\t\t%d\t\t\t%f\t%f\t%f\t%f\tYES" % (svmf, 1 - best_x_3, best_y_3, dist, roc_auc))
    # CV
    fpr_3_CV[svmf].insert(0, 0.0)
    fpr_3_CV[svmf].append(1.0)
    tpr_3_CV[svmf].insert(0, 0.0)
    tpr_3_CV[svmf].append(1.0)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_tpr = np.interp(mean_fpr, fpr_3_CV[svmf], tpr_3_CV[svmf])
    roc_auc = auc(mean_fpr, mean_tpr)
    # Get best point
    dist = 100
    best_x_CV_3 = best_y_CV_3 = 0
    for f, t in zip(mean_fpr, mean_tpr):
        d = distance.euclidean((0, 1), (f, t))
        if d < dist:
            dist = d
            best_x_CV_3 = f
            best_y_CV_3 = t
    plt.plot(mean_fpr, mean_tpr, label="with CV, %d SVM folds: %f" % (svmf, roc_auc), linewidth=2.0)
    print("YES\t\t%d\t\t\t%f\t%f\t%f\t%f\tYES" % (svmf, 1 - best_x_CV_3, best_y_CV_3, dist, roc_auc))

plt.legend(loc=0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Method 3 with & without Cross-Validation 3 Follow-Ups")




""" Plot Sensitivity/Specificity """
min_sensitivity_1 = [0.7] * len(n_components)
min_sensitivity_2 = [0.6] * len(n_components)

# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

for svmf in SVM_folds:
    plt.plot(n_components, sensitivity[svmf][n_components[0]:n_components[-1] + 1], label="Sensitivity, %d folds" %svmf, linewidth=2.0)
    plt.plot(n_components, specificity[svmf][n_components[0]:n_components[-1] + 1], label="Specificity, %d folds" %svmf, linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 3 without Cross-Validation")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

for svmf in SVM_folds:
    plt.plot(n_components, sensitivity_CV[svmf][n_components[0]:n_components[-1] + 1], label="Sensitivity, %d folds" %svmf, linewidth=2.0)
    plt.plot(n_components, specificity_CV[svmf][n_components[0]:n_components[-1] + 1], label="Specificity, %d folds" %svmf, linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 3 with Cross-Validation")


""" Plot Sensitivity/Specificity 3 Follow-Ups """
# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

for svmf in SVM_folds:
    plt.plot(n_components, sensitivity_3[svmf][n_components[0]:n_components[-1] + 1], label="Sensitivity, %d folds" %svmf, linewidth=2.0)
    plt.plot(n_components, specificity_3[svmf][n_components[0]:n_components[-1] + 1], label="Specificity, %d folds" %svmf, linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 3 without Cross-Validation 3 Follow-Ups")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

for svmf in SVM_folds:
    plt.plot(n_components, sensitivity_3_CV[svmf][n_components[0]:n_components[-1] + 1], label="Sensitivity, %d folds" %svmf, linewidth=2.0)
    plt.plot(n_components, specificity_3_CV[svmf][n_components[0]:n_components[-1] + 1], label="Specificity, %d folds" %svmf, linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 3 with Cross-Validation 3 Follow-Ups")


# plt.show()
