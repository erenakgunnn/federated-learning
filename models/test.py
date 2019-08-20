#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    counter = 0
    for idx, (data, target) in enumerate(data_loader):
        positions = []
        for i in range(len(target)):
            if target[i]==0 or target[i]==8:
                        target[i]=0
                        positions.append(i)
            elif target[i]==1 or target[i]==9:
                        target[i]=1 
                        positions.append(i)
        data = np.asarray(data)
        data = torch.tensor(data[positions])
        target = np.asarray(target)
        target = torch.tensor(target[positions])
        counter += len(positions)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= counter
    accuracy = 100.00 * float(correct) / float(counter)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct,counter, accuracy))
    return accuracy, test_loss

