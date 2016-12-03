#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from scipy.spatial import distance

""" Check if we already have the metrics file """
file = "method_1.pickle"
if os.path.isfile(file):
    [_, n_components, _, fpr, tpr, _, _, sensitivity, specificity, _, _,
     sensitivity_3, specificity_3, fpr_3, tpr_3] =\
        pickle.load(open(file, "rb"))

file_CV = "method_1_CV.pickle"
if os.path.isfile(file_CV):
    [_, n_components_CV, _, fpr_CV, tpr_CV, _, _, sensitivity_CV,
     specificity_CV, _, _, sensitivity_3_CV, specificity_3_CV, fpr_3_CV,
     tpr_3_CV, fold] = pickle.load(open(file_CV, "rb"))


print("CV\t\t1 - fpr\t\ttpr\t\t\tdist\t\tAUC\t\t\t3FU")

""" Average Cross-Validation results """
avg_fpr_CV = [0] * len(fpr_CV[0])
for key, values in fpr_CV.items():
    avg_fpr_CV = [x + y for x, y in zip(avg_fpr_CV, values)]
avg_fpr_CV = [x / fold for x in avg_fpr_CV]

avg_tpr_CV = [0] * len(tpr_CV[0])
for key, values in tpr_CV.items():
    avg_tpr_CV = [x + y for x, y in zip(avg_tpr_CV, values)]
avg_tpr_CV = [x / fold for x in avg_tpr_CV]
# Re-arrange fpr and tpr, in fpr's increasing order
avg_tpr_CV = [x for (y, x) in sorted(zip(avg_fpr_CV, avg_tpr_CV))]
avg_fpr_CV = sorted(avg_fpr_CV)

# 3 FU
avg_fpr_CV_3 = [0] * len(fpr_3_CV[0])
for key, values in fpr_3_CV.items():
    avg_fpr_CV_3 = [x + y for x, y in zip(avg_fpr_CV_3, values)]
avg_fpr_CV_3 = [x / fold for x in avg_fpr_CV_3]

avg_tpr_CV_3 = [0] * len(tpr_3_CV[0])
for key, values in tpr_3_CV.items():
    avg_tpr_CV_3 = [x + y for x, y in zip(avg_tpr_CV_3, values)]
avg_tpr_CV_3 = [x / fold for x in avg_tpr_CV_3]
# Re-arrange fpr and tpr, in fpr's increasing order
avg_tpr_CV_3 = [x for (y, x) in sorted(zip(avg_fpr_CV_3, avg_tpr_CV_3))]
avg_fpr_CV_3 = sorted(avg_fpr_CV_3)


avg_sensitivity_CV = [0] * len(sensitivity_CV[0])
for key, values in sensitivity_CV.items():
    avg_sensitivity_CV = [x + y for x, y in zip(avg_sensitivity_CV, values)]
avg_sensitivity_CV = [x / fold for x in avg_sensitivity_CV]

avg_specificity_CV = [0] * len(specificity_CV[0])
for key, values in specificity_CV.items():
    avg_specificity_CV = [x + y for x, y in zip(avg_specificity_CV, values)]
avg_specificity_CV = [x / fold for x in avg_specificity_CV]


avg_sensitivity_CV_3 = [0] * len(sensitivity_3_CV[0])
for key, values in sensitivity_3_CV.items():
    avg_sensitivity_CV_3 = [x + y for x, y in zip(avg_sensitivity_CV_3, values)]
avg_sensitivity_CV_3 = [x / fold for x in avg_sensitivity_CV_3]

avg_specificity_CV_3 = [0] * len(specificity_3_CV[0])
for key, values in specificity_3_CV.items():
    avg_specificity_CV_3 = [x + y for x, y in zip(avg_specificity_CV_3, values)]
avg_specificity_CV_3 = [x / fold for x in avg_specificity_CV_3]


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
fpr.insert(0, 0.0)
fpr.append(1.0)
tpr.insert(0, 0.0)
tpr.append(1.0)
mean_fpr = np.linspace(0, 1, 1000)
mean_tpr = np.interp(mean_fpr, fpr, tpr)
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
plt.plot(mean_fpr, mean_tpr, label="w/o CV: %f" % (roc_auc), linewidth=2.0)
print("NO\t\t%f\t%f\t%f\t%f\tNO" % (1 - best_x, best_y, dist, roc_auc))

# CV
avg_fpr_CV.insert(0, 0.0)
avg_fpr_CV.append(1.0)
avg_tpr_CV.insert(0, 0.0)
avg_tpr_CV.append(1.0)
mean_fpr = np.linspace(0, 1, 1000)
mean_tpr = np.interp(mean_fpr, avg_fpr_CV, avg_tpr_CV)
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
plt.plot(mean_fpr, mean_tpr, label="with CV: %f" % (roc_auc), linewidth=2.0)
print("YES\t\t%f\t%f\t%f\t%f\tNO" % (1 - best_x_CV, best_y_CV, dist, roc_auc))


plt.legend(loc=0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Method 1 with & without Cross-Validation")


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
fpr_3.insert(0, 0.0)
fpr_3.append(1.0)
tpr_3.insert(0, 0.0)
tpr_3.append(1.0)
mean_fpr = np.linspace(0, 1, 1000)
mean_tpr = np.interp(mean_fpr, fpr_3, tpr_3)
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
plt.plot(mean_fpr, mean_tpr, label="w/o CV: %f" % (roc_auc), linewidth=2.0)
print("NO\t\t%f\t%f\t%f\t%f\tYES" % (1 - best_x_3, best_y_3, dist, roc_auc))

# CV
avg_fpr_CV_3.insert(0, 0.0)
avg_fpr_CV_3.append(1.0)
avg_tpr_CV_3.insert(0, 0.0)
avg_tpr_CV_3.append(1.0)
mean_fpr = np.linspace(0, 1, 1000)
mean_tpr = np.interp(mean_fpr, avg_fpr_CV_3, avg_tpr_CV_3)
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
plt.plot(mean_fpr, mean_tpr, label="with CV: %f" % (roc_auc), linewidth=2.0)
print("YES\t\t%f\t%f\t%f\t%f\tYES" % (1 - best_x_CV_3, best_y_CV_3, dist, roc_auc))


plt.legend(loc=0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Method 1 with & without Cross-Validation 3 Follow-Ups")




""" Plot Sensitivity/Specificity """
min_sensitivity_1 = [0.7] * len(n_components)
min_sensitivity_2 = [0.6] * len(n_components)

# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])

plt.plot(n_components, sensitivity[n_components[0]:n_components[-1] + 1], label="Sensitivity", linewidth=2.0)
plt.plot(n_components, specificity[n_components[0]:n_components[-1] + 1], label="Specificity", linewidth=2.0)
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 without Cross-Validation")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])

plt.plot(n_components, avg_sensitivity_CV[n_components[0]:n_components[-1] + 1], label="Sensitivity", linewidth=2.0)
plt.plot(n_components, avg_specificity_CV[n_components[0]:n_components[-1] + 1], label="Specificity", linewidth=2.0)
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 with Cross-Validation")


""" Plot Sensitivity/Specificity 3 Follow-Ups """
# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])

plt.plot(n_components, sensitivity_3[n_components[0]:n_components[-1] + 1], label="Sensitivity", linewidth=2.0)
plt.plot(n_components, specificity_3[n_components[0]:n_components[-1] + 1], label="Specificity", linewidth=2.0)
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 without Cross-Validation 3 Follow-Ups")

# CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])

plt.plot(n_components, avg_sensitivity_CV_3[n_components[0]:n_components[-1] + 1], label="Sensitivity", linewidth=2.0)
plt.plot(n_components, avg_specificity_CV_3[n_components[0]:n_components[-1] + 1], label="Specificity", linewidth=2.0)
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 with Cross-Validation 3 Follow-Ups")


plt.show()
