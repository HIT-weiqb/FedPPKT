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

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
    if args.gpu>=0:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu is not None else 'cpu'
    args.autocast = dummy_ctx

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name =  '%s-iid[%d]-%s'%(args.dataset, args.iid, args.model)
    if args.pretrained == 0:
        logger = datafree.utils.logger.get_logger(log_name, output='pretraineds/log-%s-iid[%d]-%s-user[%d].txt'%(args.dataset, args.iid, args.model, args.num_users))
    else:
        logger = datafree.utils.logger.get_logger(log_name, output='results/log-%s-iid[%d]-%s-user[%d].txt'%(args.dataset, args.iid, args.model, args.num_users))


    # load dataset and user groups     数据集的获取以及划分
    # 这里要改一下，如果是预训练好的，就要加载一下之前保存的user_groups
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
        pretraineds = torch.load('pretraineds/VGG_%s_%s_iid[%d]_user[%d].pth.tar'%(args.dataset, args.pretrained_epochs, args.iid, args.num_users), map_location='cuda')
        user_groups = pretraineds['user_groups']
        user_groups_test = pretraineds['user_groups_test']
        list_test_acc = pretraineds['test_accuracy_global'] 
        list_test_loss = pretraineds['test_loss_global']
        list_vaild_acc = pretraineds['valid_accuracy_local']

        # 统计各client上每个类别的数据的数量        
        # for i in range(args.num_users):
        #     num_per_classes = [0 for k in range(args.num_classes)]
        #     num_per_classes_test = [0 for k in range(args.num_classes)]
        #     num_total = 0
        #     num_total_test = 0
        #     print(f'client[{i}]\'s num_per_classes:')
        #     for j in range(len(user_groups[i])):
        #         label = train_dataset.targets[user_groups[i][j]]
        #         num_per_classes[label] += 1
        #         num_total += 1
        #     print(num_per_classes)
        #     print('total = {}'.format(num_total))
        #     print(f'client[{i}]\'s num_per_classes_test:')
        #     for j in range(len(user_groups_test[i])):
        #         label = test_dataset.targets[user_groups_test[i][j]]
        #         num_per_classes_test[label] += 1
        #         num_total_test += 1
        #     print(num_per_classes_test)
        #     print('total = {}'.format(num_total_test))

        
        print('Pre-trained Model Loaded.\n')
        for idx in range(args.num_users):
            Pretrained_Models[idx].load_state_dict(pretraineds[idx])
            Pretrained_Models[idx].to(device)
            Pretrained_Models[idx].train()
            print('| Client Idx : {} | Model Architecture : {:.12s} | Test Global Acc : {:.2f}  Test Global loss : {:.2f} Vaild Local acc: {}'.format(idx, MODEL_NAMES[idx], list_test_acc[idx], list_test_loss[idx], list_vaild_acc[idx])) 
    else:
        # Pretraining
        pretraineds = {}  # 存储已预训练好的模型
        max_eval_acc = [0. for i in range(args.num_users)]
        for idx in range(args.num_users):
        # 对每个局部模型做预训练
            print('#################################################################################')
            print('Pretraining Clinet Idx: {} , Model Architecutre : {:.12s}'.format(idx, MODEL_NAMES[idx]))
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

            pretrained = PreTrained(args=args, dataset=train_dataset,test_dataset=test_dataset,
                                idxs=user_groups[idx],test_idx=user_groups_test[idx],optimizer=optimizer, scheduler=scheduler, logger=logger)  # , logger=logger
            for epoch in tqdm(range(args.pretrained_epochs)):
                w, _ = pretrained.update_weights(model=model, global_round=epoch+1, idx=idx)
                model.load_state_dict(w)  # 将训练的模型参数保存下来
                Pretrained_Models[idx].load_state_dict(w)
                if (epoch+1) % 20 == 0:
                    # evalW
                    eval_acc, eval_loss = pretrained.inference(model=model)            
                    eval_acc = float(eval_acc)
                    eval_loss = float(eval_loss)
                    # eval acc 就是在本地数据集上的准确率；
                    logger.info('[Eval] Client Idx={idx} Architecture={arch:.12s} Round={round} Acc={acc:.2f} Loss={loss:.2f}'.format(
                        idx=idx, arch=MODEL_NAMES[idx], round=epoch+1, acc=eval_acc, loss=eval_loss))
                    
                    if eval_acc >= max_eval_acc[idx]:
                        max_eval_acc[idx] = copy.deepcopy(eval_acc)
                        pretraineds[idx] = Pretrained_Models[idx].state_dict()
                    
                    # test_acc, test_loss = test_inference(args, Pretrained_Models[idx], test_dataset, idxs=[i for i in range(len(test_dataset))])
                    # logger.info('[Test] Client Idx={idx} Architecture={arch:.12s} Round={round} Acc={acc:.2f} Loss={loss:.2f}'.format(
                    #     idx=idx, arch=MODEL_NAMES[idx], round=epoch+1, acc=test_acc, loss=test_loss))
                

        # Test inference 
        list_test_acc, list_test_loss = [], []
        for idx in range(args.num_users):
            # 加载在本地数据集上准确度最高的模型（因为保存的模型是它
            Pretrained_Models[idx].load_state_dict(pretraineds[idx])
            Pretrained_Models[idx].to(device)
            test_acc, test_loss = test_inference(args, Pretrained_Models[idx], test_dataset, idxs=[i for i in range(len(test_dataset))])
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)
        logger.info('Final Vaild Accuracy on Local Dataset: {acc}'.format(acc=max_eval_acc))
        logger.info('Final Test Loss on Global Dataset : {loss}'.format(loss=list_test_loss))
        logger.info('Final Test Accuracy on Global Dataset: {acc}'.format(acc=list_test_acc))

        # Saving the objects train_loss and train_accuracy:
        pretraineds['test_accuracy_global'] = list_test_acc
        pretraineds['test_loss_global'] = list_test_loss
        pretraineds['valid_accuracy_local'] = max_eval_acc  # 在本地数据集上测试的结果
        pretraineds['user_groups'] = user_groups
        pretraineds['user_groups_test'] = user_groups_test
            
        model_path = os.path.join('pretraineds/VGG_%s_%s_iid[%d]_user[%d].pth.tar'%(args.dataset, args.pretrained_epochs, args.iid, args.num_users))
        torch.save(pretraineds, model_path)
        # sys.exit(0)



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
    max_acc_global = 0.
    # 新增
    global_generator_weights = generator[0].state_dict()
    print('#########################################################################################')
    logger.info('Start Fast Data Free Distillation.')
    global_test_acc, global_test_loss = [], []
    max_acc_global_sum = []
    max_loss_global_sum = []
    local_valid_acc = {}
    for i in range(args.num_users):
        local_valid_acc[i] = []
    for epoch in tqdm(range(args.comm_rounds)):  # 通讯轮数 50
        local_weights = []
        local_generator_weights = []
        global_model.train()
        # 每个local epoch会测试local model的准确率
        for idx in range(args.num_users):  # 用deepcopy来实现，global model初始化local model
            # 加载local model
            local_model = FastDateFree(args=args, dataset=test_dataset,
                                      idxs=user_groups_test[idx])  # , logger=logger
            # update student model's weight first
            w, local_acc = local_model.distillation(  # 在这里实现蒸馏
                student=copy.deepcopy(global_model), synthesizer=synthesizer[idx], teacher=Pretrained_Models[idx], global_round=epoch, client=idx, logger=logger)
            local_generator_weights.append(synthesizer[idx].generator_state_dict())
            local_valid_acc[idx].append(local_acc)
            # record local model
            local_weights.append(copy.deepcopy(w))  # 统计各client的local model参数
        
        # update global weights 
        global_weights = average_weights(local_weights)  # 模型聚合
        global_model.load_state_dict(global_weights)

        # 新增 update global generator weights
        global_generator_weights = average_weights(local_generator_weights)

        # 在这加server端，对合成器的蒸馏
        if epoch >= args.warmup:
            logger.info('Start server training global Generator with global model')
            Sgenerator = Generator(nz=args.nz, ngf=64, img_size=32, nc=3)
            Ssynthesizer = datafree.synthesis.FastMetaSynthesizer(copy.deepcopy(global_model), None, Sgenerator,args=args,   # 全局模型作为teacher model
                        img_size=(3, 32, 32), transform=train_dataset.transform, normalizer=normalizer, idx=-1)
            Ssynthesizer.update_generator(global_generator_weights)  # 训练全局合成器
            genWeight = None
            for i in range(args.Sepoch):
                genWeight, LOSS = Ssynthesizer.refineMetaGenerator(local_weights)
                if((i+1)%10 == 0):
                    logger.info('| Server Training Generaotr Stage | Communication Round : {} | Server Training Round : {} |  Training Loss : {} |'.format(
                            epoch+1, i+1, LOSS))

            # 下发给各client局部的元合成器
            for i in range(args.num_users):
                synthesizer[i].update_generator(genWeight)
                # synthesizer[i].update_generator(global_generator_weights)



        # 每个comm后会测试global model的acc
        if epoch >= args.warmup:  # 每过1个epoch, 测试global model在整个dataset上的性能
            acc_global, loss_global = 0., 0.
            acc_global, loss_global = test_inference(args, global_model, test_dataset, idxs=[i for i in range(len(test_dataset))])
    
            global_test_acc.append(acc_global)
            global_test_loss.append(loss_global)
            logger.info('| Global Model Testing Stage | Communication Round : {} | Global Test Acc : {}   Test Loss : {}'.format(
                    epoch+1, acc_global, loss_global))

            checkpoint = {
                    'epoch': epoch+1,
                    'global_acc': acc_global,
                    'global_loss': loss_global,
                    'global_model': global_model.state_dict()
                }
            model_path = os.path.join('results/Global_[%s]_VGG_%s_%s_iid[%d]_epoch[%d].pth.tar'%(args.alg, args.dataset, args.pretrained_epochs, args.iid, args.Sepoch))    
            torch.save(checkpoint, model_path)
            if acc_global >= max_acc_global:
                max_acc_global = copy.deepcopy(acc_global)
                result_path3 = os.path.join('results/Best_Global_[%s]_VGG_%s_[%s]_iid[%d]_epoch[%d].pth.tar'%(args.alg, args.dataset, args.pretrained_epochs, args.iid, args.Sepoch)) 
                torch.save(checkpoint, result_path3)
            # draw curve
            if (epoch+1) % 5 == 0: # 每5个comm round 重新画一遍
                # Plot Global Test Acc Curve
                plt.figure()
                plt.title('Global Model Test Acc vs Communication rounds')
                plt.plot(range(len(global_test_acc)), global_test_acc, color='r')
                plt.ylabel('Global Model Test Acc')
                plt.xlabel('Communication Rounds')
                plt.savefig(os.path.join('figure/Global_[%s]_TestAcc_%s_iid[%d]_epoch[%d].png'%(args.alg, args.dataset, args.iid, args.Sepoch)) )
                # Plot Global Test Loss Curve
                plt.figure()
                plt.title('Global Model Test Loss vs Communication rounds')
                plt.plot(range(len(global_test_loss)), global_test_loss, color='b')
                plt.ylabel('Global Model Test Loss')
                plt.xlabel('Communication Rounds')
                plt.savefig(os.path.join('figure/Global_[%s]_TestLoss_%s_iid[%d]_epoch[%d].png'%(args.alg, args.dataset, args.iid, args.Sepoch)))

                # # Plot Local Vaild Acc Curve
                # for idx in range(args.num_users):
                #     plt.figure()
                #     plt.title('Local Model Valid Acc vs Communication rounds')
                #     plt.plot(range(len(local_valid_acc[idx])), local_valid_acc[idx], color='b')
                #     plt.ylabel('Local Model[%d] Valid Acc'%(idx))
                #     plt.xlabel('Communication Rounds')
                #     plt.savefig(os.path.join('figure/Local_VaildAcc_%s_iid[%d]_idx[%d].png'%(args.dataset, args.iid, idx)))
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
