#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as sci


def test_img(net_g, datatest, args,class_0,class_1):
    net_g.eval()
    with torch.no_grad():
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=args.bs)
        l = len(data_loader)
        counter=0
        for idx, (data, target) in enumerate(data_loader):
            positions = []
            for i in range(len(target)):
                for j in range(len(class_0)):
                    if target[i]==class_0[j]:
                        positions.append(i)
                        target[i]=0
                for k in range(len(class_1)):
                    if target[i]==class_1[j]:
                        positions.append(i)
                        target[i]=1
            if len(positions)==0:
                continue
            data = np.asarray(data)
            data = torch.tensor(data[positions])
            target = np.asarray(target)
            target = torch.tensor(target[positions]) 
            counter+=len(positions)          
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
                test_loss, correct, counter, accuracy))
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
        increments = np.asarray(increments)/4
        return accuracy, list(increments)
    else:
        print("An error occured")
        return accuracy, momentums