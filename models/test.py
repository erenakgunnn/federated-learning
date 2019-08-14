#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as sci


def test_img(net_g, datatest, args):
    net_g.eval()
    with torch.no_grad():
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=args.bs)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * float(correct) / float(len(data_loader.dataset))
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def LocalAcc(args,w_locals,valid_set, net):
    #calculate accuracy of each local trained net
    accuracy = []
    for i in range(len(w_locals)):
        net.load_state_dict(w_locals[i])
        acc , test_loss = test_img(net,valid_set,args)
        accuracy.append(acc)
    #find normalized accuracies
    if args.client_momentum:
        momentums = []
        total = 0
        """
        for i in accuracy:
            total+=i**2
        momentums = [i**2/total for i in accuracy]
        """
        
        for i in accuracy:
            total+=np.exp(i)    
        momentums = [np.exp(i)/total for i in accuracy]
        return accuracy, momentums

    elif args.client_prob:
        increments = sci.zscore(accuracy)
        increments = np.asarray(increments)*0.1/args.num_users
        return accuracy, list(increments)
    else:
        print("An error occured")
        return accuracy, momentums