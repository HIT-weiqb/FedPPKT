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
from torch.backends import cudnn
from options import args_parser
from train.update import LocalUpdate, test_inference
from models.VGG import vgg8_bn
from utils import get_dataset, average_weights, exp_details
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

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

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name =  'Baseline_%s-iid[%d]-%s'%(args.dataset, args.iid, args.model)
    logger = datafree.utils.logger.get_logger(log_name, output='results/Baseline_log-%s-iid[%d]-%s.txt'%(args.dataset, args.iid, args.model))

    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(args)

    # 加载预先划分好的数据分布
    pretraineds = torch.load('pretraineds/VGG_%s_%s_iid[%d].pth.tar'%(args.dataset, args.pretrained_epochs, args.iid))
    user_groups = pretraineds['user_groups']
    user_groups_test = pretraineds['user_groups_test']
    # list_test_acc = pretraineds['test_accuracy_global'] 
    # list_test_loss = pretraineds['test_loss_global']
    # list_vaild_acc = pretraineds['valid_accuracy_local']
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

    # build model
    global_model = vgg8_bn(args)
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    test_acc_list = []
    for epoch in tqdm(range(args.comm_rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            w, _ = local_model.update_weights(copy.deepcopy(global_model), global_round=epoch+1, client=idx, round=epoch)
            local_weights.append(copy.deepcopy(w))
            
        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # 在整个测试数据集上测试
        test_acc, test_loss = test_inference(args, global_model, test_dataset, idxs=[i for i in range(len(test_dataset))])
        logger.info('[Test] FedAvg  Round={round} Acc={acc:.2f} Loss={loss:.2f}'.format(
            round=epoch+1, acc=test_acc, loss=test_loss))
        test_acc_list.append(test_acc)

    plt.figure()
    plt.title('[Baseline] Global Model Test Acc vs Communication rounds')
    plt.plot(range(len(test_acc_list)), test_acc_list, color='r')
    plt.ylabel('Global Model Test Acc')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join('figure/[Baseline]Global_[%s]_TestAcc_%s_iid[%d].png'%(args.alg, args.dataset, args.iid)) )

    print(test_acc_list)
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))