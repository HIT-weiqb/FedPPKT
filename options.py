#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # pretrained stage parameters
    parser.add_argument('--pretrained', type=int, default=1, help="whether pretrained or not")
    parser.add_argument('--pretrained_epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--pretrained_lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--pretrained_bs', type=int, default=32, help="batch size of pretraining")
    parser.add_argument('--pretrained_momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--pretrained_optimizer', type=str, default='sgd', help="type \
                    of optimizer")
    parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

    # model arguments
    parser.add_argument('--model', type=str, default='VGG8', help='model name')
    # parser.add_argument('--kernel_num', type=int, default=9,
    #                     help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to \
    #                     use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    # parser.add_argument('--num_filters', type=int, default=32,
    #                     help="number of filters for conv nets -- 32 for \
    #                     mini-imagenet, 64 for omiglot.")
    # parser.add_argument('--max_pool', type=str, default='True',
    #                     help="Whether use max pooling rather than \
    #                     strided convolutions")
    

    # Basic
    parser.add_argument('--gpu_id', type=int, default=7, help="GPU id,-1 for CPU")
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--log_tag', default='cifar10')
    parser.add_argument('--lr', default=0.2, type=float,
                    help='initial learning rate for KD')
    parser.add_argument('--T', default=20, type=float)
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")

    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--num_shards', type=int, default=2, help="number of each clients' shard")
    parser.add_argument('--gpu', default=7, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--iid', type=int, default=1,  
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    # Fast Data Free 
    parser.add_argument('--epoch', type=int, default=10, help="number of epoch")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--comm_rounds', type=int, default=50,  #   200   50    1
                        help="number of rounds of training")
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--adv', default=1.1, type=float, help='scaling factor for adversarial distillation')
    parser.add_argument('--bn', default=10.0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0.4, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
    parser.add_argument('--save_dir', default='run/cifar10', type=str)
    parser.add_argument('--lr_g', default=5e-3, type=float, help='initial learning rate for generator')
    parser.add_argument('--lr_z', default=0.015, type=float, help='initial learning rate for latent code')
    parser.add_argument('--g_steps', default=2, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--nz', default=256, type=int, help='nz')
    parser.add_argument('--reset_l0', default=1, type=int,
                        help='reset l0 in the generator during training')
    parser.add_argument('--reset_bn', default=0, type=int,
                        help='reset bn layers during training')
    parser.add_argument('--bn_mmt', default=0.9, type=float,
                        help='momentum when fitting batchnorm statistics')
    parser.add_argument('--is_maml', default=1, type=int,
                        help='meta gradient: is maml or reptile')
    parser.add_argument('--kd_steps', default=200, type=int, metavar='N',
                    help='number of iterations for KD after generation')
    parser.add_argument('--ep_steps', default=200, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--warmup', default=2, type=int, metavar='N',
                        help='which epoch to start kd')
    parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args
