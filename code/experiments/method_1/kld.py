#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats


def kld_continuous(m1, v1, m2, v2):
    d = len(m1) # Size of distributions

    dv1 = np.linalg.det(v1) # Determinants
    dv2 = np.linalg.det(v2)

    iv2 = np.linalg.inv(v2)

    diff = m2 - m1

    KL = 0.5 * (np.log(dv2 / dv1)
                - d
                + np.trace(iv2 * v1)
                + diff.T * iv2 * diff)
    return KL.item(0)


def kld_discrete(pk, qk):
    return stats.entropy(pk, qk)
