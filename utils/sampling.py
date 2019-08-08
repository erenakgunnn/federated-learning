#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from utils.options import args_parser


np.random.seed(42)  #to make results reproducible
args = args_parser()

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if (args.client_momentum or args.client_prob):
        items = len(dataset)*0.9
    else:
        items = len(dataset)    
    num_items = int(items/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, all_idxs


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    if (args.client_momentum or args.client_prob):
        bias = 30
    else:
        bias = 0   
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    # divide and assign
    valid_set=[]
    for i in range(num_shards):
        valid_set= valid_set + list(idxs[num_imgs*i:num_imgs*i+bias])
    np.random.shuffle(valid_set)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set: 
            dict_users[i] = np.concatenate((dict_users[i], idxs[(rand*num_imgs+bias):(rand+1)*num_imgs]), axis=0)
    return dict_users,valid_set


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if (args.client_momentum or args.client_prob):
        items = len(dataset)*0.9
    else:
        items = len(dataset)    
    num_items = int(items/num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, all_idxs

def cifar_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 250
    if (args.client_momentum or args.client_prob):
        bias = 25
    else:
        bias = 0

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.targets.numpy() it does not work for some reason
    labels = np.asarray(dataset.targets, dtype=np.int32)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0 , :]
    # divide and assign
    valid_set=[]
    for i in range(num_shards):
        valid_set= valid_set + list(idxs[num_imgs*i:num_imgs*i+bias])
    np.random.shuffle(valid_set)

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set: 
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs+bias:(rand+1)*num_imgs]), axis=0) 
    return dict_users,valid_set

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    print(d[100].shape)
