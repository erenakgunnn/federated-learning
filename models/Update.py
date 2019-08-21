#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                positions = []
                for i in range(len(labels)):
                    if labels[i]==0 or labels[i]==8 or labels[i]==1 or labels[i]==9:
                        labels[i]=0
                        positions.append(i)
                    elif labels[i]==2 or labels[i]==3 or labels[i]==4 or labels[i]==5 or labels[i]==6 or labels[i]==7: 
                        labels[i]=1
                        positions.append(i)
                    else:
                        print("Problem with data labels", labels[i], i)    
                if len(positions) == 0:
                    continue        
                images = np.asarray(images)
                labels = np.asarray(labels)
                images = torch.tensor(images[positions])
                labels = torch.tensor(labels[positions])
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            if len(batch_loss) != 0:    
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                return net.state_dict(), "NoRelatedData"    
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

