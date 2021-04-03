#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


import SOMGit as sm
import classifier as cl




outlier_common_misclf=set(sm.indices).intersection(cl.msClsListNB)

outlier_common_misclf=list(outlier_common_misclf)

print("Noise indices", outlier_common_misclf)

