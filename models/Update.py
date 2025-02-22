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
                for i in range(len(labels)):
                    if labels[i]==0 or labels[i]==1 or labels[i]==8 or labels[i]==9:
                        labels[i]=0
                    else:
                        labels[i]=1
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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def lineer_prob(scores):
  return scores/sum(scores)

def sigmoid(x):
    a = []
    for i in x:
        a.append(1/(1+np.exp(-i)))
    return a

def prob_update(idxs_users,client_prob,client_rank,rank_increment,client_freq, args):
    all = set([i for i in range(args.num_users)])-set(idxs_users)
    for i in range(len(rank_increment)):
        client_rank[idxs_users[i]]+=(rank_increment[i] - 0.125)
        client_freq[idxs_users[i]]+=1
    for x in list(all):
        client_rank[x] += 0.125/9
    temp_list = sigmoid(client_rank)
    total = sum(temp_list)
    for i in range(len(client_prob)):
        client_prob[i] = temp_list[i]/total

         
