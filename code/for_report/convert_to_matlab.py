#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import scipy.io
import os

file_path = "../experiments_only_hmms/CMs_woMCI_iters_10/metrics.pickle"

if os.path.isfile(file_path):
    # [HMM_train, n_components, iterations, fpr, tpr, F1_global,
    #  F1_weighted, sensitivity, specificity, F1_global_3, F1_weighted_3,
    #  sensitivity_3, specificity_3, fpr_3, tpr_3, folds]
    [HMM_train, n_components, iterations, fpr, tpr, F1_global,
     F1_weighted, sensitivity, specificity, F1_global_3, F1_weighted_3,
     sensitivity_3, specificity_3, fpr_3, tpr_3] =\
        pickle.load(open(file_path, "rb"))

a_dict = {'HMM_train': HMM_train,
          'n_components': n_components,
          'iterations': iterations,
          'fpr': fpr,
          'tpr': tpr,
          'F1_global': F1_global,
          'F1_weighted': F1_weighted,
          'sensitivity': sensitivity,
          'specificity': specificity,
          'fpr_3': fpr_3,
          'tpr_3': tpr_3,
          'F1_global_3': F1_global_3,
          'F1_weighted_3': F1_weighted_3,
          'sensitivity_3': sensitivity_3,
          'specificity_3': specificity_3}


scipy.io.savemat('/home/thomai/Dropbox/thesis_KTH_KI/mat_results/method_1.mat',
                 mdict={'HMM_train': HMM_train,
          'n_components': n_components,
          'iterations': iterations,
          'fpr': fpr,
          'tpr': tpr,
          'F1_global': F1_global,
          'F1_weighted': F1_weighted,
          'sensitivity': sensitivity,
          'specificity': specificity,
          'fpr_3': fpr_3,
          'tpr_3': tpr_3,
          'F1_global_3': F1_global_3,
          'F1_weighted_3': F1_weighted_3,
          'sensitivity_3': sensitivity_3,
          'specificity_3': specificity_3})