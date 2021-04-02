#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:29:29 2021

@author: rashidahasan
"""

import SOMGit as sm
import classifier as cl




outlier_common_misclf=set(sm.indices).intersection(cl.msClsListNB)

outlier_common_misclf=list(outlier_common_misclf)

print("Noise indices", outlier_common_misclf)

