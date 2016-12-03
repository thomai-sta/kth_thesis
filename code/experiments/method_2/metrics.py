#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import itertools

def get_positives_negatives(true, predicted):
    """ Positives: 1 - AD
        Negatives: 0 - non-AD """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for tr, pr in zip(true, predicted):
        if tr == pr:
            " TRUE "
            if tr == 1:
                TP += 1
            elif tr == 0:
                TN += 1
        else:
            " FALSE "
            if pr == 0:
                " Predicted negative, but it is positive "
                FN += 1
            elif pr == 1:
                " Predicted positive, but it is negative "
                FP += 1
    return TP, FP, TN, FN


def get_ROC_AUC(tpr, fpr):
    return metrics.auc(fpr, tpr)


def get_sensitivity(true, predicted):
    """ Recall, True positive rate """
    TP, FP, TN, FN = get_positives_negatives(true, predicted)
    sensitivity = TP / float(TP + FN)
    return sensitivity


def get_specificity(true, predicted):
    """ True negative rate """
    TP, FP, TN, FN = get_positives_negatives(true, predicted)
    specificity = TN / float(TN + FP)
    return specificity


def get_F1_score(true, predicted):
    f1_global = metrics.f1_score(true, predicted, average='micro',
                                 pos_label=None)
    f1_weighted = metrics.f1_score(true, predicted, average='weighted',
                                   pos_label=None)
    return f1_global, f1_weighted


def get_confusion_matrix(test_labels, predicted_vals):
    CM = metrics.confusion_matrix(test_labels, predicted_vals)
    return CM


def plot_confusion_matrix(cm, classes, iters, results_dir, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    manager_per = plt.get_current_fig_manager()
    manager_per.resize(*manager_per.window.maxsize())

    to_save_per = plt.gcf()
    to_save_per.savefig(results_dir + title + '.png')

def get_precision(true, predicted):
    """ tp / (tp + fp) """
    TP, FP, TN, FN = get_positives_negatives(true, predicted)
    precision = TP / float(TP + FP)
    return precision


def get_average_precision(true, predicted, average=None):
    """ This score corresponds to the area under the precision-recall curve """
    AP = metrics.average_precision_score(true, predicted, average=average)
    return AP


def get_accuracy_score(true, predicted, normalize=True):
    accuracy = metrics.accuracy_score(true, predicted, normalize=normalize)
    return accuracy


def print_and_get_summary(true, predicted, target_names=None):
    summary = metrics.classification_report(true, predicted,
                                            target_names=target_names)
    print(summary)
    return summary