import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time
import os

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class FastMetaSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, args, img_size,
                  transform=None, normalizer=None, idx=None):
        super(FastMetaSynthesizer, self).__init__(teacher, student)
        self.idx = idx
        self.save_dir = os.path.join(args.save_dir, 'client%d'%(self.idx))
        self.img_size = img_size 
        self.iterations = args.g_steps
        self.lr_g = args.lr_g
        self.lr_z = args.lr_z
        self.nz = args.nz
        self.adv = args.adv
        self.bn = args.bn
        self.oh = args.oh
        self.bn_mmt = args.bn_mmt
        self.ismaml = args.is_maml
        self.args = args
        self.num_classes = args.num_classes
        self.synthesis_batch_size = args.synthesis_batch_size
        self.sample_batch_size = args.batch_size
        self.autocast = args.autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.device = args.gpu
        self.generator = generator.to(self.device).train()
        self.hooks = []

        self.ep = 0
        self.ep_start = args.warmup
        self.reset_l0 = args.reset_l0
        self.reset_bn = args.reset_bn
        self.prev_z = None

        if self.ismaml:  # meta_optimizer: 用于更新元生成器的optimizer
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )
        self.aug = transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
    def update_student(self, w):
        self.student.load_state_dict(w)

    def update_generator(self, w):
        self.generator.load_state_dict(w)

    def generator_state_dict(self):
        return self.generator.state_dict()
    
    def synthesize(self, targets=None):

        start = time.time()

        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120+self.ep_start) and self.reset_l0:
            reset_l0(self.generator)
        
        best_inputs = None
        # randomly create z  (batchsize, 256) 即每个样本的特征是256维
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        if targets is None:  # randomly create groung truth
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        fast_generator = self.generator.clone()  # self.generator 就是 元生成器， fast_generator 是 内部循环时用到的临时生成器

        optimizer = torch.optim.Adam([  # 同时对z和fast generator的参数进行优化
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):  # synthesize data
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            if it == 0:
                originalMeta = inputs  # record initial z
            
            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(inputs_aug)
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0 and (self.ep >= self.ep_start):  # adverisal loss
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv  # total loss

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()     
                    best_inputs = inputs.data  # only reserve the best input

            optimizer.zero_grad()
            loss.backward()

            if self.ismaml:  # meta-learn the generator
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations-1): self.meta_optimizer.step()

            optimizer.step()  # 更新z以及fast optimizer

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add( best_inputs )
        dst = self.data_pool.get_dataset(transform=self.transform)
 
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}, end - start
        # return {"synthetic": best_inputs}, {"meta": originalMeta}  # get original meta points for tSNE vis
        
    def sample(self):
        return self.data_iter.next()
