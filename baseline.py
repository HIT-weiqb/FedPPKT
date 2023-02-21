#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import sys
import time
import numpy as np
import random
from tqdm import tqdm

import torch
import datafree
from models.generator import Generator
from datafree.utils._utils import dummy_ctx
from torch.backends import cudnn
from options import args_parser
from train.update import PreTrained, FastDateFree, test_inference
from models.VGG import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
# from resnet_8x import ResNet34_8x
from utils import get_dataset, average_weights, exp_details
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

CANDIDATE_MODELS = {"VGG8": vgg8_bn,   ## 一共十个模型架构
                    "VGG11": vgg11_bn,
                    "VGG13": vgg13_bn,
                    "VGG16": vgg16_bn,
                    "VGG19": vgg19_bn
                    } 
MODEL_NAMES = ["VGG8", "VGG8", "VGG11", "VGG11", "VGG13", "VGG13", "VGG16", "VGG16", "VGG19", "VGG19"]
if __name__ == '__main__':



    start_time = time.time()

    # set args
    args = args_parser()
    exp_details(args)

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # set device
    if args.gpu_id>=0:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu is not None else 'cpu'
    args.autocast = dummy_ctx

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name =  '%s-iid[%d]-%s'%(args.dataset, args.iid, args.model)
    if args.pretrained == 0:
        logger = datafree.utils.logger.get_logger(log_name, output='pretraineds/log-%s-iid[%d]-%s.txt'%(args.dataset, args.iid, args.model))
    else:
        logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/log-%s-iid[%d]-%s.txt'%(args.dataset, args.iid, args.model))


    # load dataset and user groups     数据集的获取以及划分？
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(args)

    # build teacher model
    Pretrained_Models = {}
    for i in range(args.num_users):
        Pretrained_Models[i] = CANDIDATE_MODELS[MODEL_NAMES[i]](args)

    # Set the model to train and send it to device.
    for i in range(args.num_users):
        Pretrained_Models[i].to(device)
        Pretrained_Models[i].train()


    # pretrained
    if args.pretrained == 1:  # 预训练好了，加载数据集的划分、准确率、模型参数
        pretraineds = torch.load('pretraineds/VGG_%s_%s_iid[%d].pth.tar'%(args.dataset, args.pretrained_epochs, args.iid))
        user_groups = pretraineds['user_groups']
        user_groups_test = pretraineds['user_groups_test']
        list_test_acc = pretraineds['test_accuracy'] 
        list_test_loss = pretraineds['test_loss']
        
        print('Pre-trained Model Loaded.\n')
        for idx in range(args.num_users):
            Pretrained_Models[idx].load_state_dict(pretraineds[idx])
            Pretrained_Models[idx].to(device)
            Pretrained_Models[idx].train()
            print('| Client Idx : {} | Model Architecture : {:.12s} | Test Acc : {}  Test loss : {}'.format(idx, MODEL_NAMES[idx], list_test_acc[idx], list_test_loss[idx])) 
    else:
        # Pretraining
        train_loss, train_accuracy = [], []


        for idx in range(args.num_users):
        # 对每个局部模型做预训练
            print('#################################################################################')
            print('Preraining Clinet Idx: {} , Model Architecutre : {:.12s}'.format(idx, MODEL_NAMES[idx]))
            model = Pretrained_Models[idx]

            # Set optimizer for the local updates
            if args.pretrained_optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrained_lr,
                                            momentum=args.pretrained_momentum, weight_decay=1e-4)
            elif args.pretrained_optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrained_lr,
                                            weight_decay=1e-4)

            milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
            scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.2)

            pretrained = PreTrained(args=args, dataset=train_dataset,
                                idxs=user_groups[idx],optimizer=optimizer, scheduler=scheduler, logger=logger)  # , logger=logger
            for epoch in tqdm(range(args.pretrained_epochs)):
                w, loss = pretrained.update_weights(model=model, global_round=epoch+1, idx=idx)
                model.load_state_dict(w)  # 将训练的模型参数保存下来

                if (epoch+1) % 10 == 0:
                    # eval
                    list_acc, list_loss = [], []
                    acc, loss = pretrained.inference(model=model)            
                    
                    logger.info('[Eval] Client Idx={idx} Architecture={arch:.12s} Round={round} Acc={acc} Loss={loss:.4f}'.format(
                        idx=idx, arch=MODEL_NAMES[idx], round=epoch+1, acc=acc, loss=loss))

        # Test inference 
        list_test_acc, list_test_loss = [], []
        for idx in range(args.num_users):
            test_acc, test_loss = test_inference(args, Pretrained_Models[idx], test_dataset, user_groups_test[idx])
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)

        logger.info('Final Test Loss : {loss}'.format(loss=list_test_loss))
        logger.info('Final Test Accuracy: {acc}'.format(acc=list_test_acc))

        # Saving the objects train_loss and train_accuracy:
        pretraineds = {}  # 存储已预训练好的模型
        pretraineds['test_accuracy'] = list_test_acc
        pretraineds['test_loss'] = list_test_loss
        pretraineds['user_groups'] = user_groups
        pretraineds['user_groups_test'] = user_groups_test
        for idx in range(args.num_users):
            pretraineds[idx] = Pretrained_Models[idx].state_dict()
        model_path = os.path.join('pretraineds/VGG_%s_%s_iid[%d].pth.tar'%(args.dataset, args.pretrained_epochs, args.iid))
        torch.save(pretraineds, model_path)
        sys.exit(0)



    # federated learning & data free distillation
    # global model & generator
    global_model = CANDIDATE_MODELS[args.model](args)  
    global_model.to(device)
    global_model.train()
    print(global_model)
    # initialize normalizer
    normalizer = datafree.utils.Normalizer(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    
    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    generator = {}
    synthesizer = {}
    for i in range(args.num_users):
        generator[i] = Generator(nz=args.nz, ngf=64, img_size=32, nc=3)
        generator[i].to(device)
        synthesizer[i] = datafree.synthesis.FastMetaSynthesizer(Pretrained_Models[i], copy.deepcopy(global_model), generator[i],args=args, 
                    img_size=(3, 32, 32), transform=train_dataset.transform, normalizer=normalizer, idx=i)
    # copy weights 存储全局模型的参数
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    local_val_acc, local_val_loss = [0. for i in range(args.num_users)], [0. for i in range(args.num_users)] # 测试的local model在本地数据集上的acc,loss
    max_acc_global = 0.
    print('#########################################################################################')
    logger.info('Start Fast Data Free Distillation.')
    for epoch in tqdm(range(args.comm_rounds)):  # 通讯轮数
        local_weights = []
        print(f'\n | Global Communication Round : {epoch+1} |\n')

        global_model.train()

        for idx in range(args.num_users):  # 用deepcopy来实现，global model初始化local model
            local_model = FastDateFree(args=args, dataset=test_dataset,
                                      idxs=user_groups_test[idx])  # , logger=logger
            # update student model's weight first
            w = local_model.distillation(  # 在这里实现蒸馏
                student=copy.deepcopy(global_model), synthesizer=synthesizer[idx], teacher=Pretrained_Models[idx], global_round=epoch, client=idx, logger=logger)
        
            # record local model
            local_weights.append(copy.deepcopy(w))  # 统计各client的local model参数

            # 测试local model在local dataset上的准确率和loss
            # if (epoch+1) % 1 == 0:
            tmp_weights = global_model.state_dict()  # 暂存global model的权重
            global_model.load_state_dict(w)  # 加载local model
            test_acc, test_loss = local_model.inference(global_model)   
            local_val_acc[idx] = test_acc
            local_val_loss[idx] = test_loss
            global_model.load_state_dict(tmp_weights)
        
            logger.info('| Communication Round : {} | Client Idx : {} | Distillation Test Acc : {}   Test Loss : {}'.format(
                epoch+1, idx, test_acc, test_loss))

                
        # update global weights
        global_weights = average_weights(local_weights)  # 模型聚合
        global_model.load_state_dict(global_weights)

    
    checkpoint ={
                    'local_val_acc': local_val_acc,  
                    'local_val_loss':local_val_loss,
                    'global_model': global_model.state_dict()
                }
    model_path = os.path.join('results/Baseline_%s_%s_iid[%d].pth.tar'%(args.dataset, args.epoch, args.iid))
    torch.save(checkpoint, model_path)
    print(local_val_acc)
    print(local_val_loss)
        # if (epoch+1) % 1 == 0:  # 每过1个epoch, 测试global model在整个dataset上的性能
        #     acc_global, loss_global = test_inference(args, global_model, test_dataset, idxs=[i for i in range(len(test_dataset))])
        #     print('##############################################################################################')
        #     print('| Testing Global Model Stage | Communication Round : {} | Client Idx : {} | Global Test Acc : {}   Test Loss : {}'.format(
        #             epoch+1, idx, acc_global, loss_global))
        #     print('##############################################################################################')

        #     checkpoint = {
        #             'epoch': epoch+1,
        #             'global_acc': acc_global,
        #             'global_loss': loss_global,
        #             'local_val_acc': local_val_acc,  
        #             'local_val_loss':local_val_loss,
        #             'global_model': global_model.state_dict()
        #         }
        #     result_path2 = os.path.join('{}/distillation_checkpoints'.format(path_project),'VGGtraining_checkpoint_{}_{}_iid[{}]_{}.pth.tar'.format(args.dataset, args.comm_rounds, args.iid, epoch+1))
        #     torch.save(checkpoint, result_path2)
        #     if acc_global >= max_acc_global:
        #         max_acc_global = copy.deepcopy(acc_global)
        #         result_path3 = os.path.join('{}/best_results'.format(path_project),'VGGbest_results_checkpoint_{}_{}_iid[{}]_{}.pth.tar'.format(args.dataset, args.comm_rounds, args.iid, epoch+1))
        #         torch.save(checkpoint, result_path3)
                
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
