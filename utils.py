#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    user_groups_test = None
    if args.dataset == 'cifar':
        data_dir = '{}/cifar10/'.format(args.data_root)  #  ../data/cifar/'
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=val_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups, user_groups_test = cifar_iid(train_dataset, test_dataset, args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, user_groups_test = cifar_noniid(train_dataset, test_dataset, args)
    elif args.dataset == 'cifar100':
        data_dir = '{}/cifar100/'.format(args.data_root)  #  ../data/cifar100/'
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )])
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )])
        
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=val_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups, user_groups_test = cifar_iid(train_dataset, test_dataset, args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, user_groups_test = cifar_noniid(train_dataset, test_dataset, args)
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, user_groups_test = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, user_groups_test


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    # 预训练
    print(f'    Pretrained Epochs : {args.pretrained_epochs}')
    print(f'    Pretrained Batch Size : {args.pretrained_bs}')
    print(f'    Pretrained Learning Rate : {args.pretrained_lr}')
    print(f'    Pretrained Optimizer : {args.pretrained_optimizer}')


    # 蒸馏
    print(f'    Model     : {args.model}')
    print('    Optimizer : SGD ')
    print(f'    Global Rounds   : {args.comm_rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    return
