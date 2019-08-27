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
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,resnet56
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
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
    if args.load_model != "-":
        path = "./pretrained_models/"+args.load_model+".pth"
        net_glob.load_state_dict(torch.load(path))


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
    w_acc = []
    updated_epoch=0
    class_accuracies = []
    # virtual classes
    classes = args.classes.split('-')
    class0 = list(map(int,list(classes[0])))
    class1 = list(map(int,list(classes[1])))


    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        if args.groupdata:
            if args.classes=="08-19":
                inputs=range(40)
            elif args.classes == "247-356":
                inputs=range(40,100)
        else:
            inputs = range(100)        
        idxs_users = np.random.choice(inputs, m, replace=False)
        for idx in idxs_users:
#            print(idx,": ",dict_users[idx])
            local = LocalUpdate(class0, class1, args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if loss != "NoRelatedData":
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            else:
                print("prints user idx and NoRelatedData condition",idx, loss)    
        # update global weights

        if args.accumulate:
#            print("len w locals: ",len(w_locals))
            for i in w_locals:
                w_acc.append(i)
                if len(w_acc) == 10:
                    w_glob = FedAvg(w_acc)
                    w_acc = []
                    updated_epoch+=1
                    print("updated epoch number:",updated_epoch)
#            print("accumulated weight number: ",len(w_acc))        
        else:
            w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_test, loss_test = test_img(class_accuracies,class0, class1, net_glob, dataset_test, args)
        val_acc_list.append(acc_test)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Accuracy {}'.format(iter, loss_avg, val_acc_list[iter]))
        loss_train.append(loss_avg)

    # plot loss curve

    if args.save_model :
        path = "./pretrained_models/"+"{}_{}.pth".format(args.classes,args.epochs)
        torch.save(net_glob.state_dict(),path)
        

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel("epoch")
    plt.title("Splitting: {}  Pretrained: {}".format(args.classes,args.load_model))
    plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}_locEp{}_groupdata{}_split"{}".png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep,args.groupdata,args.classes))


    plt.figure()
    class_accuracies = np.asarray(class_accuracies)
    class_accuracies = class_accuracies/10
    plt.ylabel('class accuracies')
    plt.xlabel("epoch")
    plt.title("Splitting: {}  Pretrained: {}".format(args.classes,args.load_model))
    for i in range(len(class0)):
        plt.plot(range(len(loss_train)), class_accuracies[:,class0[i]], label="class: {}".format(class0[i]) )
    plt.legend()
    plt.savefig('./log/class0_accuracies_{}_{}_{}_C{}_iid{}_locEp{}_groupdata{}_split"{}".png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep,args.groupdata,args.classes))

    plt.figure()
    plt.ylabel('class accuracies')
    plt.xlabel("epoch")
    plt.title("Splitting: {}  Pretrained: {}".format(args.classes,args.load_model))
    for i in range(len(class1)):
        plt.plot(range(len(loss_train)), class_accuracies[:,class1[i]], label="class: {}".format(class1[i]) )
    plt.legend()
    plt.savefig('./log/class1_accuracies_{}_{}_{}_C{}_iid{}_locEp{}_groupdata{}_split"{}".png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep,args.groupdata,args.classes))


    plt.figure()
    plt.plot(range(len(val_acc_list)), val_acc_list)
    plt.title("Splitting: {}  Pretrained: {}".format(args.classes,args.load_model))
    plt.ylabel('test_accuracy')
    plt.xlabel("epoch")
    plt.savefig('./log/accuracy_{}_{}_{}_C{}_iid{}_locEp{}_groupdata{}_split"{}".png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep,args.groupdata,args.classes))



    # testing
    class_accuracies2 = []
    acc_train, loss_train = test_img(class_accuracies2,class0, class1, net_glob, dataset_train, args)
    acc_test, loss_test = test_img(class_accuracies2,class0, class1, net_glob, dataset_test, args)
    print("Training accuracy: {}".format(acc_train))
    print("Testing accuracy: {}".format(acc_test))
    #writing to txt file
    text_logs = open("./log/text_log.txt","a")
    text_logs.write('--dataset:"{}"  model:"{}"  epochs:{}  local epochs:{}  fraciton:{}  number of user:{}  iid:{}  groupdata:{} \n'.format(args.dataset, args.model, args.epochs, args.local_ep, args.frac,args.num_users, args.iid, args.groupdata))
    text_logs.write('Accuracies during training::"{}" \n'.format(val_acc_list))
    text_logs.write('train accuracy: {} test accuracy: {} final train loss: {} final test loss: {}\n\n'.format(acc_train,acc_test,loss_train,loss_test))
    text_logs.write('Classes splitted::"{}" \n'.format(args.classes))

