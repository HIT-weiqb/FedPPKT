#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from stringprep import in_table_c4
import numpy as np
from torchvision import datasets, transforms

def iid(dataset, test_dataset, args):
    """
    Sample I.I.D. client data from CIFAR10/CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/args.num_users)  # 5000
    num_items_test = int(len(test_dataset) / args.num_users)  # 1000
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_test, all_idxs_test = {}, [i for i in range(len(test_dataset))]
    for i in range(args.num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        dict_users_test[i] = set(np.random.choice(all_idxs_test, num_items_test,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
    return dict_users, dict_users_test


def noniid(dataset, test_dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10/CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 狄利克雷分布来划分数据集
    y_train = np.array(dataset.targets)
    y_test = np.array(test_dataset.targets)
    label_distribution = np.random.dirichlet([args.beta]*args.num_users, args.num_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少
    class_idcs = [np.argwhere(y_train==y).flatten()    for y in range(args.num_classes)]
    class_idcs_test = [np.argwhere(y_test==y).flatten()    for y in range(args.num_classes)]
    # 记录每个K个类别对应的样本下标
    dict_users = {i: np.array([], dtype=int) for i in range(args.num_users)}
    dict_users_test = {i: np.array([], dtype=int) for i in range(args.num_users)}
    client_idcs = [[] for _ in range(args.num_users)]
    client_idcs_test = [[] for _ in range(args.num_users)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
        
    for c, fracs in zip(class_idcs_test, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs_test[i] += [idcs]
        

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    client_idcs_test = [np.concatenate(idcs) for idcs in client_idcs_test]

    # 统计各client上每个类别的数据的数量        
    for i in range(args.num_users):
        dict_users[i] = client_idcs[i]
        dict_users_test[i] = client_idcs_test[i]
        num_per_classes = [0 for k in range(args.num_classes)]
        num_per_classes_test = [0 for k in range(args.num_classes)]
        num_total = 0
        num_total_test = 0
        print(f'client[{i}]\'s num_per_classes:')
        for j in range(len(dict_users[i])):
            label = dataset.targets[dict_users[i][j]]
            num_per_classes[label] += 1
            num_total += 1
        print(num_per_classes)
        print('total = {}'.format(num_total))
        print(f'client[{i}]\'s num_per_classes_test:')
        for j in range(len(dict_users_test[i])):
            label = test_dataset.targets[dict_users_test[i][j]]
            num_per_classes_test[label] += 1
            num_total_test += 1
        print(num_per_classes_test)
        print('total = {}'.format(num_total_test))
    
    return dict_users, dict_users_test


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
