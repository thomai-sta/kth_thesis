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
file = "method_2.pickle"
if os.path.isfile(file):
    [_, n_components, _, fpr, tpr, sensitivity,
     specificity, sensitivity_3, specificity_3, fpr_3, tpr_3,
     folds, SVM_folds] =\
        pickle.load(open(file, "rb"))

file_CV = "method_2_CV.pickle"
if os.path.isfile(file_CV):
    [_, n_components, _, fpr_CV, tpr_CV, sensitivity_CV,
     specificity_CV, sensitivity_3_CV, specificity_3_CV, fpr_3_CV, tpr_3_CV,
     folds, SVM_folds] = pickle.load(open(file_CV, "rb"))

print("CV\tSVMf\tdor\t\tf1\t\t3FU")

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
min_sensitivity_1 = [0.7] * len(n_components)
min_sensitivity_2 = [0.6] * len(n_components)

# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    avg_dor = np.sum(dor[svmf]) / len(dor[svmf])
    avg_f1 = np.sum(f1[svmf]) / len(dor[svmf])
    print("NO\t%d\t%f\t%f\tNO" % (svmf, avg_dor, avg_f1))
    plt.plot(n_components,
             sensitivity[svmf][n_components[0]:n_components[-1] + 1],
             label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components,
             specificity[svmf][n_components[0]:n_components[-1] + 1],
             label="Specificity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components, f1[svmf], "o-", label="F1 Score, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 2 without Cross-Validation")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    avg_dor = np.sum(dor_CV[svmf]) / len(dor[svmf])
    avg_f1 = np.sum(f1_CV[svmf]) / len(dor[svmf])
    print("YES\t%d\t%f\t%f\tNO" % (svmf, avg_dor, avg_f1))
    plt.plot(n_components,
             sensitivity_CV[svmf][n_components[0]:n_components[-1] + 1],
             label="Sensitivity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components,
             specificity_CV[svmf][n_components[0]:n_components[-1] + 1],
             label="Specificity, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1
    plt.plot(n_components, f1_CV[svmf], "o-", label="F1 Score, %d Folds" % svmf, linewidth=2.0,
             color=colors1[idx])
    idx += 1

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 2 with Cross-Validation")


""" Plot Sensitivity/Specificity 3 Follow-Ups """
# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    avg_dor = np.sum(dor_3[svmf]) / len(dor[svmf])
    avg_f1 = np.sum(f1_3[svmf]) / len(dor[svmf])
    print("NO\t%d\t%f\t%f\tYES" % (svmf, avg_dor, avg_f1))
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

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 2 without Cross-Validation 3 Follow-Ups")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)

idx = 0
for svmf in SVM_folds:
    avg_dor = np.sum(dor_CV_3[svmf]) / len(dor[svmf])
    avg_f1 = np.sum(f1_CV_3[svmf]) / len(dor[svmf])
    print("YES\t%d\t%f\t%f\tYES" % (svmf, avg_dor, avg_f1))
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

plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 2 with Cross-Validation 3 Follow-Ups")


# cmap = plt.get_cmap('jet')
colors2 = cmap(np.linspace(0, 1.0, 4))

for svmf in SVM_folds:
    idx = 0
    plt.figure()
    plt.hold(True)
    plt.axis([0, n_components[-1] + 1, 0, 10])

    plt.plot(n_components, dor[svmf], label="DOR, %d Folds" % svmf,
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_3[svmf], label="DOR 3 FU, %d Folds" % svmf,
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_CV[svmf], label="DOR with CV, %d Folds" % svmf,
             linewidth=2.0, color=colors2[idx])
    idx += 1
    plt.plot(n_components, dor_CV_3[svmf],
             label="DOR with CV 3 FU, %d Folds" % svmf, linewidth=2.0,
             color=colors2[idx])
    plt.legend(loc=0)
    plt.grid()
    plt.xlabel('Number of States')
    plt.title("Method 2 Diagnostic odds ratio")


plt.show()
