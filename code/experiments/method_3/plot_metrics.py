#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from scipy.spatial import distance

""" Check if we already have the metrics file """
if os.path.isfile("metrics.pickle"):
    [HMM_train, n_components, iterations, fpr, tpr, sensitivity,
                 specificity, sensitivity_3, specificity_3, fpr_3, tpr_3,
                 folds, SVM_folds] =\
        pickle.load(open("metrics.pickle", "rb"))
    # [n_components, folds, iterations, fpr, tpr,
    #      best_F1, F1_global, F1_weighted, sensitivity, specificity,
    #      F1_global_3, F1_weighted_3, sensitivity_3, specificity_3, fpr_3,
    #      tpr_3] =\
    #     pickle.load(open("metrics.pickle", "rb"))
    # HMM_train = "woMCI"
    # SVM_train = "all"
else:
    n_components = []
    iterations = []
    fpr = []
    tpr = []
    F1_global = []
    F1_weighted = []
    sensitivity = []
    specificity = []
    fpr_3 = []
    tpr_3 = []
    F1_global_3 = []
    F1_weighted_3 = []
    sensitivity_3 = []
    specificity_3 = []

min_sensitivity_1 = [0.7] * len(n_components)
min_sensitivity_2 = [0.65] * len(n_components)


def plot_all(n_components=n_components, iterations=iterations, fpr=fpr, tpr=tpr,
             sensitivity=sensitivity, specificity=specificity,
             sensitivity_3=sensitivity_3, specificity_3=specificity_3,
             fpr_3=fpr_3, tpr_3=tpr_3, folds=folds):

    folder = ('results' + HMM_train + '_iters_%d/') % iterations

    """  Plot ROC curves """
    mean_fpr = np.linspace(0, 1, 1000)
    roc = plt.figure()
    roc.hold(True)
    plt.axis([0, 1, 0, 1])
    plt.grid()
    x = [0, 1]
    y = [0, 1]
    plt.plot(x, y, '--', label="Random")

    """  Plot ROC curves 3 FU """
    roc_3 = plt.figure()
    roc_3.hold(True)
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.plot(x, y, '--', label="Random")

    for fold in SVM_folds:
        plt.figure()
        plt.hold(True)
        plt.axis([0, n_components[-1] + 1, 0, 1])
        title = "Iters: %d, Fold: %d" % (iterations, fold)
        plt.title(title)

        plt.plot(n_components, sensitivity[fold][n_components[0]:n_components[-1] + 1],
                 '*-', label="Sensitivity")
        plt.plot(n_components, specificity[fold][n_components[0]:n_components[-1] + 1],
                 '*-', label="Specificity")
        plt.plot(n_components, min_sensitivity_1, "--")
        plt.plot(n_components, min_sensitivity_2, "--")
        plt.legend(loc=0)
        plt.grid()
        plt.xlabel('Number of States')

        manager_per = plt.get_current_fig_manager()
        manager_per.resize(*manager_per.window.maxsize())

        to_save_per = plt.gcf()
        to_save_per.savefig(folder + title + '.png')

        """ Plot 3 FU """
        plt.figure()
        plt.hold(True)
        plt.axis([0, n_components[-1] + 1, 0, 1])
        title_3 = "Iters: %d, Fold: %d, Follow-Ups: 3" % (iterations, fold)
        plt.title(title_3)
        plt.plot(n_components, sensitivity_3[fold][n_components[0]:n_components[-1] + 1],
                 '*-', label="Sensitivity")
        plt.plot(n_components, specificity_3[fold][n_components[0]:n_components[-1] + 1],
                 '*-', label="Specificity")
        plt.plot(n_components, min_sensitivity_1, "--")
        plt.plot(n_components, min_sensitivity_2, "--")
        plt.legend(loc=0)
        plt.grid()
        plt.xlabel('Number of States')

        manager_per = plt.get_current_fig_manager()
        manager_per.resize(*manager_per.window.maxsize())

        to_save_per = plt.gcf()
        to_save_per.savefig(folder + title_3 + '.png')

        """ Plot ROC curves """
        fpr[fold].insert(0, 0.0)
        fpr[fold].append(1.0)
        tpr[fold].insert(0, 0.0)
        tpr[fold].append(1.0)
        mean_tpr = np.interp(mean_fpr, fpr[fold], tpr[fold])
        roc_auc = auc(mean_fpr, mean_tpr)
        """ Get best point """
        dist = 100
        best_x = best_y = 0
        for f, t in zip(mean_fpr, mean_tpr):
            d = distance.euclidean((0, 1), (f, t))
            if d < dist:
                dist = d
                best_x = f
                best_y = t
        plt.figure(roc.number)
        plt.plot(mean_fpr, mean_tpr, label="AUC -> %f" % (roc_auc))
        plt.plot(best_x, best_y, '*', label="Euclidean -> %f" %dist)

        """ Plot ROC curves 3 FU """
        fpr_3[fold].insert(0, 0.0)
        fpr_3[fold].append(1.0)
        tpr_3[fold].insert(0, 0.0)
        tpr_3[fold].append(1.0)
        mean_tpr = np.interp(mean_fpr, fpr_3[fold], tpr_3[fold])
        roc_auc = auc(mean_fpr, mean_tpr)
        """ Get best point """
        dist = 100
        best_x = best_y = 0
        for f, t in zip(mean_fpr, mean_tpr):
            d = distance.euclidean((0, 1), (f, t))
            if d < dist:
                dist = d
                best_x = f
                best_y = t
        plt.figure(roc_3.number)
        plt.plot(mean_fpr, mean_tpr, label="AUC -> %f" % (roc_auc))
        plt.plot(best_x, best_y, '*', label="Euclidean -> %f" %dist)

    plt.figure(roc.number)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves, Iters: %d' % iterations)
    plt.legend(loc=0)

    to_save_per = plt.gcf()
    to_save_per.savefig(folder + 'ROC_curves_1_fold.png')

    plt.figure(roc_3.number)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves, Iters: %d, Follow-Ups: 3' % iterations)
    plt.legend(loc=0)

    to_save_per = plt.gcf()
    to_save_per.savefig(folder + 'ROC_curves_3_FU_1_fold.png')

    # plt.show()


if __name__ == '__main__':
    print("plotting?")
    plot_all()