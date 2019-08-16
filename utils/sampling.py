#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from utils.options import args_parser


#np.random.seed(41)  #to make results reproducible
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
    if args.mixed==0.0:
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
    else:
        num_tot= 50000
        idxs = np.arange(num_tot)
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        labels = np.asarray(dataset.targets, dtype=np.int32)
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
        idxs = idxs_labels[0 , :]
        valid_set=[]
        non_iid_set=[]
        num_shards, num_imgs = 160 , 225
        idx_shard = [i for i in range(num_shards)]
        for i in range(10):
            valid_set = valid_set + list(idxs[(5000*i):5000*i+500])
            non_iid_set = non_iid_set + list(idxs[(5000*i)+1400:5000*(i+1)])         
            for j in range(int(args.mixed*num_users)):
            #    dict_users[j] = dict_users[j] + list(idxs[(5000)*i+500+(45*j):(5000)*i+500+(45*(j+1))])
                dict_users[j] = np.concatenate((dict_users[j], list(idxs[(5000)*i+500+(45*j):(5000)*i+500+(45*(j+1))])), axis=0)
        for z in range(int(num_users*args.mixed), num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set: 
                dict_users[z] = np.concatenate((dict_users[z], non_iid_set[rand*num_imgs:(rand+1)*num_imgs]), axis=0) 
        for a in range(int(args.mixed*num_users)):
            np.random.shuffle(dict_users[a])  
        np.random.shuffle(valid_set)          
        print("Labels for user 0:", labels[dict_users[0]])
        print("Labels for user 19:", labels[dict_users[19]])
        print("Labels for user 20:", labels[dict_users[20]])
        print("Labels for user 90:", labels[dict_users[90]])        
        return dict_users,valid_set

               
if __name__ == '__main__':
    '''
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
                                   
    num = 100
    d = mnist_noniid(dataset_train, num)
    print(d[100].shape)
    '''
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dict_users,valid_idxs = cifar_noniid(dataset_train, args.num_users)
    print(dict_users[19].shape)
