#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from stringprep import in_table_c4
import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, test_dataset, args):
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


def cifar_noniid(dataset, test_dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10/CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards = args.num_shards * args.num_users
    # num_imgs = int(len(dataset) / num_shards)
    # num_imgs_test = int(len(test_dataset) / num_shards)  # 一共10个client 每个client 分5个shard 
    # idx_shard = [i for i in range(num_shards)]
    # # idx_shard_test = [i for i in range(num_shards)]

    # dict_users = {i: np.array([], dtype=int) for i in range(args.num_users)}
    # dict_users_test = {i: np.array([], dtype=int) for i in range(args.num_users)}
    # idxs = np.arange(len(dataset))
    # idxs_test = np.arange(len(test_dataset))

    # # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.targets)
    # labels_test = np.array(test_dataset.targets)

    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]

    # idxs_labels_test = np.vstack((idxs_test, labels_test))
    # idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    # idxs_test = idxs_labels_test[0, :]
    # # divide and assign
    # for i in range(args.num_users):
    #     rand_set = set(np.random.choice(idx_shard, args.num_shards, replace=False))  # 从0,1,2.....19 shard的标号中随机抽取两个
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    #         dict_users_test[i] = np.concatenate(  # 按照同样的方式分配test data
    #             (dict_users_test[i], idxs_test[rand*num_imgs_test:(rand+1)*num_imgs_test]), axis=0)


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
