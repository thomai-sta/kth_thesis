#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')

cmap = plt.get_cmap('jet')
colors1 = cmap(np.linspace(0, 1.0, 9))

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

# print("CV\tSVMf\tdor\t\tf1\t\t3FU")

""" Results are already averaged """
# Get F1 scores and dor
f1 = {}
f1_3 = {}
f1_CV = {}
f1_CV_3 = {}
dor = {}
dor_3 = {}
dor_CV = {}
dor_CV_3 = {}
for svmf in SVM_folds:
    f1[svmf] = [2 * x * y / (x + y) for x, y in
                zip(sensitivity[svmf][n_components[0]:n_components[-1] + 1],
                    specificity[svmf][n_components[0]:n_components[-1] + 1])]
    f1_3[svmf] = [2 * x * y / (x + y) for x, y in
                  zip(sensitivity_3[svmf][n_components[0]:n_components[-1] + 1],
                      specificity_3[svmf][n_components[0]:n_components[-1] + 1])]
    f1_CV[svmf] = [2 * x * y / (x + y) for x, y in
                   zip(sensitivity_CV[svmf][n_components[0]:n_components[-1] + 1],
                       specificity_CV[svmf][n_components[0]:n_components[-1] + 1])]
    f1_CV_3[svmf] = [2 * x * y / (x + y) for x, y in
                     zip(sensitivity_3_CV[svmf][n_components[0]:n_components[-1] + 1],
                         specificity_3_CV[svmf][n_components[0]:n_components[-1] + 1])]

    dor[svmf] = [x * y / ((1 - x) * (1 - y)) for x, y in
                 zip(sensitivity[svmf][n_components[0]:n_components[-1] + 1],
                     specificity[svmf][n_components[0]:n_components[-1] + 1])]
    dor_3[svmf] = [x * y / ((1 - x) * (1 - y)) for x, y in
                   zip(sensitivity_3[svmf][n_components[0]:n_components[-1] + 1],
                       specificity_3[svmf][n_components[0]:n_components[-1] + 1])]
    dor_CV[svmf] = [x * y / ((1 - x) * (1 - y)) for x, y in
                    zip(sensitivity_CV[svmf][n_components[0]:n_components[-1] + 1],
                        specificity_CV[svmf][n_components[0]:n_components[-1] + 1])]
    dor_CV_3[svmf] = [x * y / ((1 - x) * (1 - y)) for x, y in
                      zip(sensitivity_3_CV[svmf][n_components[0]:n_components[-1] + 1],
                          specificity_3_CV[svmf][n_components[0]:n_components[-1] + 1])]


""" Plot Sensitivity/Specificity """
min_sensitivity = [0.5] * len(n_components)
min_specificity = [0.62] * len(n_components)
min_f1 = [0.554] * len(n_components)

# no CV
plt.figure()
ax = plt.subplot(111)
ax.hold(True)
ax.axis([0, n_components[-1] + 1, 0, 1])
ax.plot(n_components, min_sensitivity, "--", label="Min. Sensitivity",
        linewidth=2.0)
ax.plot(n_components, min_specificity, "--", label="Min. Specificity",
        linewidth=2.0)
ax.plot(n_components, min_f1, "--", label="Min. F1-Score", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    # avg_dor = np.sum(dor[svmf]) / len(dor[svmf])
    # avg_f1 = np.sum(f1[svmf]) / len(dor[svmf])
    # print("NO\t%d\t%f\t%f\tNO" % (svmf, avg_dor, avg_f1))
    ax.plot(n_components,
            sensitivity[svmf][n_components[0]:n_components[-1] + 1],
            label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
            color=colors1[idx])
    idx += 1
    ax.plot(n_components,
            specificity[svmf][n_components[0]:n_components[-1] + 1],
            label="Specificity, %d Folds" % svmf, linewidth=2.0,
            color=colors1[idx])
    idx += 1
    ax.plot(n_components, f1[svmf], "o-", label="F1 Score, %d Folds" % svmf,
            linewidth=2.0, color=colors1[idx])
    idx += 1

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=4)
ax.grid()
plt.xlabel('Number of States')
plt.title("Method 3 Blind Experiments")

# CV
plt.figure()
ax = plt.subplot(111)
ax.hold(True)
ax.axis([0, n_components[-1] + 1, 0, 1])
ax.plot(n_components, min_sensitivity, "--", label="Min. Sensitivity",
        linewidth=2.0)
ax.plot(n_components, min_specificity, "--", label="Min. Specificity",
        linewidth=2.0)
ax.plot(n_components, min_f1, "--", label="Min. F1-Score", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    # avg_dor = np.sum(dor_CV[svmf]) / len(dor[svmf])
    # avg_f1 = np.sum(f1_CV[svmf]) / len(dor[svmf])
    # print("YES\t%d\t%f\t%f\tNO" % (svmf, avg_dor, avg_f1))
    ax.plot(n_components,
            sensitivity_CV[svmf][n_components[0]:n_components[-1] + 1],
            label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
            color=colors1[idx])
    idx += 1
    ax.plot(n_components,
            specificity_CV[svmf][n_components[0]:n_components[-1] + 1],
            label="Specificity, %d Folds" % svmf, linewidth=2.0,
            color=colors1[idx])
    idx += 1
    ax.plot(n_components, f1_CV[svmf], "o-",
            label="F1 Score, %d Folds" % svmf, linewidth=2.0,
            color=colors1[idx])
    idx += 1

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=4)
ax.grid()
plt.xlabel('Number of States')
plt.title("Method 3 Semi-Blind Experiments")


""" Plot Sensitivity/Specificity 3 Follow-Ups """
# no CV
plt.figure()
ax = plt.subplot(111)
ax.hold(True)
ax.axis([0, n_components[-1] + 1, 0, 1])
ax.plot(n_components, min_sensitivity, "--", label="Min. Sensitivity",
        linewidth=2.0)
ax.plot(n_components, min_specificity, "--", label="Min. Specificity",
        linewidth=2.0)
ax.plot(n_components, min_f1, "--", label="Min. F1-Score", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    # avg_dor = np.sum(dor_3[svmf]) / len(dor[svmf])
    # avg_f1 = np.sum(f1_3[svmf]) / len(dor[svmf])
    # print("NO\t%d\t%f\t%f\tYES" % (svmf, avg_dor, avg_f1))
    plt.plot(n_components,
             sensitivity_3[svmf][n_components[0]:n_components[-1] + 1],
             label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components,
             specificity_3[svmf][n_components[0]:n_components[-1] + 1],
             label="Specificity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components, f1_3[svmf], "o-", label="F1 Score, %d Folds" % svmf,
             linewidth=2.0, color=colors1[idx])
    idx += 1

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=4)
ax.grid()
plt.xlabel('Number of States')
plt.title("Method 3 Blind Experiments 3 Follow-Ups")

# CV
plt.figure()
ax = plt.subplot(111)
ax.hold(True)
ax.axis([0, n_components[-1] + 1, 0, 1])
ax.plot(n_components, min_sensitivity, "--", label="Min. Sensitivity",
        linewidth=2.0)
ax.plot(n_components, min_specificity, "--", label="Min. Specificity",
        linewidth=2.0)
ax.plot(n_components, min_f1, "--", label="Min. F1-Score", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    # avg_dor = np.sum(dor_CV_3[svmf]) / len(dor[svmf])
    # avg_f1 = np.sum(f1_CV_3[svmf]) / len(dor[svmf])
    # print("YES\t%d\t%f\t%f\tYES" % (svmf, avg_dor, avg_f1))
    plt.plot(n_components,
             sensitivity_3_CV[svmf][n_components[0]:n_components[-1] + 1],
             label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components,
             specificity_3_CV[svmf][n_components[0]:n_components[-1] + 1],
             label="Specificity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components, f1_CV_3[svmf], "o-",
             label="F1 Score, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=4)
ax.grid()
plt.xlabel('Number of States')
plt.title("Method 3 Semi-Blind Experiments 3 Follow-Ups")


""" Plot DOR """
min_dor = [1] * len(n_components)

colors2 = cmap(np.linspace(0, 1.0, 4))

for svmf in SVM_folds:
    idx = 0
    plt.figure()
    plt.hold(True)
    plt.axis([0, n_components[-1] + 1, 0, 10])
    plt.plot(n_components, min_dor, "--", label="Min. DOR", linewidth=2.0)
    plt.plot(n_components, dor[svmf], label="DOR Blind",
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_3[svmf], label="DOR Blind 3 FU",
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_CV[svmf], label="DOR Semi-Blind",
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_CV_3[svmf],
             label="DOR Semi-Blind 3 FU", linewidth=2.0,
             color=colors2[idx])
    plt.legend(loc=0)
    plt.grid()
    plt.xlabel('Number of States')
    plt.title("Method 3 Diagnostic odds ratio, Folds: %d" % svmf)


plt.show()
