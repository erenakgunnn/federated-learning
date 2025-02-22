#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate,DatasetSplit,prob_update
from models.Nets import MLP, CNNMnist, CNNCifar,resnet56
from models.Fed import FedAvg, MomentAvg
from models.test import test_img, LocalAcc


if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = 'cpu'
    print(args.device)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users,valid_idxs = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users,valid_idxs = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users,valid_idxs = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users,valid_idxs = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet':
        net_glob = resnet56().to(args.device)    
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    client_prob = [0.01]*args.num_users
    client_freq = [0]*args.num_users
    client_rank = [0]*args.num_users
    
    #create validation set
    valid_set = DatasetSplit(dataset_train,valid_idxs)

    for iter in range(args.epochs):
        w_locals, loss_locals, client_acc, momentums = [], [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=client_prob)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        if args.client_momentum:
            #calculate accuracies
            client_acc, momentums = LocalAcc(args,w_locals, valid_set, net=copy.deepcopy(net_glob).to(args.device))
            print(client_acc,momentums)
            w_glob = MomentAvg(momentums,w_locals)
        elif args.client_prob:
            client_acc, rank_increment = LocalAcc(args,w_locals, valid_set, net=copy.deepcopy(net_glob).to(args.device))
            print("client acc: ",client_acc)
            print("rank_increments: ", rank_increment)
            prob_update(idxs_users,client_prob,client_rank,rank_increment,client_freq, args)
            w_glob = FedAvg(w_locals)

        else:
            # update global weights
            w_glob = FedAvg(w_locals)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        val_acc_list.append(acc_test)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if args.client_prob:
            print("user idxs: ",idxs_users)
            print("user probabilities: ",client_prob)
            print("user freqs: ", client_freq)
        print('Round {:3d}, Average loss {:.3f}, Accuracy {}'.format(iter, loss_avg, val_acc_list[iter]))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/Newfed_{}_{}_{}_C{}_Num_users{}_iid{}_locEp{}_ClMom{}_ClProb{}_mixed{}.png'.format(args.dataset, args.model, args.epochs, args.frac,args.num_users, args.iid, args.local_ep,args.client_momentum,args.client_prob,args.mixed))
    
    plt.figure()
    plt.plot(range(len(val_acc_list)), val_acc_list)
    plt.ylabel('test_accuracy')
    plt.xlabel("epoch")
    plt.savefig('./log/Newaccuracy_{}_{}_{}_C{}_Num_users{}_iid{}_locEp{}_ClMom{}_ClProb{}_mixed{}.png'.format(args.dataset, args.model, args.epochs, args.frac,args.num_users, args.iid, args.local_ep,args.client_momentum,args.client_prob,args.mixed))

    # testing
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Trainingg accuracy: {}".format(acc_train))
    print("Testing accuracy: {}".format(acc_test))
    print("All test accuracies: {}".format(val_acc_list))

    #writing to txt file
    text_logs = open("./log/text_log.txt","a")
    text_logs.write('--dataset:"{}"  model:"{}"  epochs:{}  local epochs:{}  fraciton:{}  number of user:{}  iid:{}  client momentum:{} client prob:{} mixed:{}\n'.format(args.dataset, args.model, args.epochs, args.local_ep, args.frac,args.num_users, args.iid, args.client_momentum,args.client_prob,args.mixed))
    text_logs.write('Accuracies during training:"{}" \n'.format(val_acc_list))
    if args.client_prob:
        text_logs.write("Frequencies of IID users: {} \n".format(client_freq[0:20]))
        text_logs.write("Probs of IID users: {} \n".format(client_prob[0:20]))
    text_logs.write('train accuracy: {} test accuracy: {} final train loss: {} final test loss: {}\n\n'.format(acc_train,acc_test,loss_train,loss_test))
