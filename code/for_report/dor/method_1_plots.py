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
file = "method_1.pickle"
if os.path.isfile(file):
    [_, n_components, _, fpr, tpr, _, _, sensitivity, specificity, _, _,
     sensitivity_3, specificity_3, fpr_3, tpr_3] =\
        pickle.load(open(file, "rb"), encoding='latin1')

file_CV = "method_1_CV.pickle"
if os.path.isfile(file_CV):
    [_, n_components_CV, _, fpr_CV, tpr_CV, _, _, sensitivity_CV,
     specificity_CV, _, _, sensitivity_3_CV, specificity_3_CV, fpr_3_CV,
     tpr_3_CV, fold] = pickle.load(open(file_CV, "rb"), encoding='latin1')


print("CV\tdor\t\tf1\t\t3FU")

""" Average Cross-Validation results """
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
    avg_sensitivity_CV_3 =\
        [x + y for x, y in zip(avg_sensitivity_CV_3, values)]
avg_sensitivity_CV_3 = [x / fold for x in avg_sensitivity_CV_3]

avg_specificity_CV_3 = [0] * len(specificity_3_CV[0])
for key, values in specificity_3_CV.items():
    avg_specificity_CV_3 =\
        [x + y for x, y in zip(avg_specificity_CV_3, values)]
avg_specificity_CV_3 = [x / fold for x in avg_specificity_CV_3]


# Get F1 scores and dor
f1 = [2 * x * y / (x + y) for x, y in
      zip(sensitivity[n_components[0]:n_components[-1] + 1],
          specificity[n_components[0]:n_components[-1] + 1])]
f1_3 = [2 * x * y / (x + y) for x, y in
        zip(sensitivity_3[n_components[0]:n_components[-1] + 1],
            specificity_3[n_components[0]:n_components[-1] + 1])]
f1_CV = [2 * x * y / (x + y) for x, y in
         zip(avg_sensitivity_CV[n_components[0]:n_components[-1] + 1],
             avg_specificity_CV[n_components[0]:n_components[-1] + 1])]
f1_CV_3 = [2 * x * y / (x + y) for x, y in
           zip(avg_sensitivity_CV_3[n_components[0]:n_components[-1] + 1],
               avg_specificity_CV_3[n_components[0]:n_components[-1] + 1])]

dor = [x * y / ((1 - x) * (1 - y)) for x, y in
       zip(sensitivity[n_components[0]:n_components[-1] + 1],
           specificity[n_components[0]:n_components[-1] + 1])]
dor_3 = [x * y / ((1 - x) * (1 - y)) for x, y in
         zip(sensitivity_3[n_components[0]:n_components[-1] + 1],
             specificity_3[n_components[0]:n_components[-1] + 1])]
dor_CV = [x * y / ((1 - x) * (1 - y)) for x, y in
          zip(avg_sensitivity_CV[n_components[0]:n_components[-1] + 1],
              avg_specificity_CV[n_components[0]:n_components[-1] + 1])]
dor_CV_3 = [x * y / ((1 - x) * (1 - y)) for x, y in
            zip(avg_sensitivity_CV_3[n_components[0]:n_components[-1] + 1],
                avg_specificity_CV_3[n_components[0]:n_components[-1] + 1])]


""" Plot Sensitivity/Specificity """
min_sensitivity_1 = [0.7] * len(n_components)
min_sensitivity_2 = [0.6] * len(n_components)

# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 1])

avg_dor = np.sum(dor) / len(dor)
avg_f1 = np.sum(f1) / len(dor)
print("NO\t%f\t%f\tNO" % (avg_dor, avg_f1))

plt.plot(n_components, sensitivity[n_components[0]:n_components[-1] + 1],
         label="Sensitivity", linewidth=2.0)
plt.plot(n_components, specificity[n_components[0]:n_components[-1] + 1],
         label="Specificity", linewidth=2.0)
plt.plot(n_components, f1, "o-", label="F1 Score", linewidth=2.0)
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

avg_dor = np.sum(dor_CV) / len(dor)
avg_f1 = np.sum(f1_CV) / len(dor)
print("YES\t%f\t%f\tNO" % (avg_dor, avg_f1))

plt.plot(n_components,
         avg_sensitivity_CV[n_components[0]:n_components[-1] + 1],
         label="Sensitivity", linewidth=2.0)
plt.plot(n_components,
         avg_specificity_CV[n_components[0]:n_components[-1] + 1],
         label="Specificity", linewidth=2.0)
plt.plot(n_components, f1_CV, "o-", label="F1 Score", linewidth=2.0)
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

avg_dor = np.sum(dor_3) / len(dor)
avg_f1 = np.sum(f1_3) / len(dor)
print("NO\t%f\t%f\tYES" % (avg_dor, avg_f1))

plt.plot(n_components, sensitivity_3[n_components[0]:n_components[-1] + 1],
         label="Sensitivity", linewidth=2.0)
plt.plot(n_components, specificity_3[n_components[0]:n_components[-1] + 1],
         label="Specificity", linewidth=2.0)
plt.plot(n_components, f1_3, "o-", label="F1 Score", linewidth=2.0)
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

avg_dor = np.sum(dor_CV_3) / len(dor)
avg_f1 = np.sum(f1_CV_3) / len(dor)
print("YES\t%f\t%f\tYES" % (avg_dor, avg_f1))

plt.plot(n_components,
         avg_sensitivity_CV_3[n_components[0]:n_components[-1] + 1],
         label="Sensitivity", linewidth=2.0)
plt.plot(n_components,
         avg_specificity_CV_3[n_components[0]:n_components[-1] + 1],
         label="Specificity", linewidth=2.0)
plt.plot(n_components, f1_CV_3, "o-", label="F1 Score", linewidth=2.0)
plt.plot(n_components, min_sensitivity_1, "--", linewidth=2.0)
plt.plot(n_components, min_sensitivity_2, "--", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 with Cross-Validation 3 Follow-Ups")


""" Plot Sensitivity/Specificity """
# no CV
plt.figure()
plt.hold(True)
plt.axis([0, n_components[-1] + 1, 0, 10])

plt.plot(n_components, dor, label="DOR", linewidth=2.0)
plt.plot(n_components, dor_3, label="DOR 3 FU", linewidth=2.0)
plt.plot(n_components, dor_CV, label="DOR with CV", linewidth=2.0)
plt.plot(n_components, dor_CV_3, label="DOR with CV 3 FU", linewidth=2.0)
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of States')
plt.title("Method 1 Diagnostic odds ratio")

plt.show()
