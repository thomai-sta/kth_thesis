#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn import cross_validation as CV
import numpy as np
import pickle
import os
import metrics

kernel = 'poly'
degree = 3
coef0 = 3
C = 63.26
gamma = 0.001


def svm_cross_validation(train_X, train_labels, test_X, folds):
    best_f1 = 0
    """ Prepare train data for CV """
    cv = CV.StratifiedKFold(train_labels, folds)

    """ Cross-validate SVM with H/AD training data """
    for train_index, test_index in cv:
        X_train = train_X[train_index, :]
        Y_train = train_labels[train_index]

        X_test = train_X[test_index, :]
        Y_test = train_labels[test_index]

        clf = svm.SVC(C=C, coef0=coef0, kernel=kernel, random_state=3,
                      max_iter=-1, probability=False, tol=0.00001,
                      verbose=False, class_weight='balanced',
                      gamma=gamma)
        clf.fit(X_train, Y_train)
        predicted = clf.predict(X_test)

        f1_global, _ = metrics.get_F1_score(Y_test, predicted)
        if f1_global > best_f1:
            best_f1 = f1_global
            best_SVM = clf

    predicted = best_SVM.predict(test_X)

    return predicted
