#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def MomentAvg(momentums,w_locals):
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        w_avg[k] *= momentums[0]
    
    for k in w_avg.keys():
        for i in range(1,len(w_locals)):
            w_avg[k] += w_locals[i][k]*momentums[i]
    return w_avg