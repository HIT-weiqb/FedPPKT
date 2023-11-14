#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import copy
import random
import datafree

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device) 

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        random.shuffle(idxs)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss
    
    def update_weights(self, model, global_round, client, round):
        # Set mode to train model
        model.train()
        epoch_loss = [] 
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.baseline_lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.baseline_lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 50 == 0):
                    print('| Global Round : {} | Client Idx : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, client, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 本地测试下
        acc, loss = self.inference(model=model)
        self.logger.info('[Eval] Round={round} Client Idx={idx} Acc={acc:.2f} Loss={loss:.2f}'.format(
                        round=round+1, idx=client,  acc=acc, loss=loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

class PreTrained(object):
    def __init__(self, args, dataset, test_dataset, idxs, test_idx, optimizer, scheduler, logger):  # , logger
        self.args = args
        self.logger = logger
        self.test_dataset = test_dataset
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs), test_dataset, list(test_idx))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to CrossEntropy Loss Function
        self.criterion = nn.CrossEntropyLoss().to(self.device) 
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_val_test(self, dataset, idxs, test_dataset, test_idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train and test (80, 20)
        random.shuffle(idxs)

        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.pretrained_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                batch_size=int(len(test_idxs)/10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round, idx):
        # Set mode to train model
        model.train()
        model.to(self.device)
        epoch_loss = 0.

    
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()

            # if self.args.verbose and ((batch_idx+1) % 25 == 0):  # training loss
            #     print('| Client Idx : {} | Pretraining Round : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         idx, global_round, (batch_idx+1) * len(images), len(self.trainloader.dataset),
            #         100. * batch_idx / len(self.trainloader), loss.item()))

            # self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())  # 记录每个batch的loss
        
        self.scheduler.step()
        epoch_loss = sum(batch_loss)/len(batch_loss)  # 求每个batch的平均

        return model.state_dict(), epoch_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = format((correct/total) *100, '.2f')
        return accuracy, loss

class FastDateFree(object):
    def __init__(self, args, dataset, idxs):  # dataset=test dataset ; idx也是测试集的下标
        self.args = args
        self.testloader = DataLoader(DatasetSplit(dataset, idxs),
                                batch_size=int(len(idxs)/10), shuffle=False)
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        self.test_criterion = nn.CrossEntropyLoss().to(self.device)
        self.distill_criterion = datafree.criterions.KLDiv(T=args.T)


    def distillation(self, student, synthesizer, teacher, global_round, client, logger):
        # path_project = '/home/aiia611/wqb/data'  #   /data_b/wqb/src/data
        MODEL_NAMES = ["VGG8", "VGG8", "VGG11", "VGG11", "VGG13", "VGG13", "VGG16", "VGG16", "VGG19", "VGG19"]
        
        # distill optimizer
        optimizer = torch.optim.SGD(student.parameters(), self.args.lr, weight_decay=self.args.weight_decay,
                                momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        
        list_test_acc = []
        list_test_loss = []
        acc = 0.
        for epoch in range(self.args.epoch):  #  epoch 10
            for bt in range( self.args.ep_steps//self.args.kd_steps ): # total kd_steps < ep_steps  1000/200 = 5 ,  这个循环似乎不该写，因为这样跑多了 去了试试*
                # update student model's weight
                synthesizer.update_student(copy.deepcopy(student.state_dict()))
                # g_steps, generate fake data
                vis_results, _ = synthesizer.synthesize() # g_steps
                
                if global_round >= self.args.warmup:
                    # kd_steps
                    student.train()
                    teacher.eval()

                    batch_loss = []
                    for i in range(self.args.kd_steps):  # distillation rounds 200 batch
                        images = synthesizer.sample()  #随机取一个batch的数据
                        if self.args.gpu is not None:
                            images = images.cuda(self.args.gpu, non_blocking=True)
                        with self.args.autocast():
                            with torch.no_grad():
                                t_out = teacher(images)
                            s_out = student(images.detach())
                            loss_s = self.distill_criterion(s_out, t_out.detach())  # KL divergence
                        batch_loss.append(loss_s)
                        optimizer.zero_grad()
                        loss_s.backward()
                        optimizer.step()

                        if i%50==0:
                            # 这里的i 和 args.kd_steps也得改 改成跟上一个循环有关的，以及ep_steps
                            print('| Communication Round : {} |  Client Idx : {} | Local Epoch : {} | Local Batch : {}/{} ({:.0f}%)]\t| Loss: {:.6f} '.format(
                                global_round, client, epoch, i+bt*self.args.kd_steps, self.args.ep_steps,
                                100. * (i+bt*self.args.kd_steps) / self.args.ep_steps, sum(batch_loss)/len(batch_loss)))
                    scheduler.step()
                # every new data
                for vis_name, vis_image in vis_results.items():
                    datafree.utils.save_image_batch( vis_image, 'checkpoints/datafree-%d/%s%s.png'%(client, vis_name, self.args.log_tag) )
            
            if  global_round >= self.args.warmup and ((epoch+1) % 5 ==0 or epoch== 0):  # 最开始也测一下，看性能如何，如果性能很差说明正常，后续再多跑下看看
                # evaluate after every epoch
                student.eval()
                test_acc, test_loss = self.inference(student)
                list_test_acc.append(round(test_acc, 2))
                list_test_loss.append(round(test_loss, 2))
                acc = copy.deepcopy(test_acc)
                logger.info('| Local Model Testing Stage | Communication Round : {} |  Client Idx : {} | Local Epoch : {} | Student Model Test Acc : {:.4f}   Test Loss : {:.4f}'.format(
                        global_round, client, epoch+1, test_acc, test_loss))  # 每个local epoch后的local acc

                if epoch%20==0:
                    checkpoint = {
                        'epoch': epoch+1,
                        'test_acc': list_test_acc,
                        'test_loss': list_test_loss,
                        'student_model': student.state_dict()
                    }
                    result_path = os.path.join('checkpoints/VGG_%s_iid[%d]_idx[%d]_%s'%(
                        self.args.dataset, self.args.iid, client, MODEL_NAMES[client]))
                    torch.save(checkpoint, result_path)

        
        return student.state_dict(), acc

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.test_criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = (correct/total) *100
        return accuracy, loss
        



def test_inference(args, model, test_dataset, idxs):  # 这个应该可以复用，测local model还是global model都行
    """ Returns the test accuracy and loss.
    """

    

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' 
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, idxs),
                                 batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = (correct/total) * 100
    return accuracy, loss
