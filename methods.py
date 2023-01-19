import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import pdb, os, copy
import torchattacks
from pathlib import Path

def robust_finetune_2(model, epochs, train_loader, config):
        # Todo
        origin_model = copy.deepcopy(model)
        origin_model.eval()
        model.train()
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr = lr)
        pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=config.robust_steps, random_start=True)
        criterion_kl = nn.KLDivLoss(size_average=False)
        XENT_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for step, (indexs, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                N = x.shape[0]
                x_adv = pgd_attack(x,y)
                #output = self.model(images)
                adv_out = model(x_adv)

                feat_clean = origin_model.feature_extractor(x)
                feat_adv = origin_model.feature_extractor(x_adv)

                score = torch.exp(-config.beta * ((feat_clean - feat_adv)**2)) 
                score = score.reshape(N,-1)
                teacher_out = origin_model.forward_with_score(feat_clean, score)
     
                # kl_loss = criterion_kl(F.log_softmax(adv_out, dim=1), F.softmax(teacher_out.detach(), dim=1))
                kl_loss = ((adv_out - teacher_out)**2).mean()
                loss = XENT_loss(adv_out,y) + config.alpha * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
            adjust_learning_rate(lr, optimizer, epoch, epochs)
        return

def robust_finetune(model, epochs, train_loader, config):
        # Todo
        origin_model = copy.deepcopy(model)
        origin_model.eval()
        model.train()
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr = lr)
        pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=config.robust_steps, random_start=True)
        criterion_kl = nn.KLDivLoss(size_average=False)
        XENT_loss = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for step, (indexs, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                N = x.shape[0]
                x_adv = pgd_attack(x,y)
                #output = self.model(images)
                adv_out = model(x_adv)

                feat_clean = origin_model.feature_extractor(x)
                feat_adv = origin_model.feature_extractor(x_adv)

                score = torch.exp(-config.beta * ((feat_clean - feat_adv)**2)) 
                score = score.reshape(N,-1)
                teacher_out = origin_model.forward_with_score(feat_clean, score)
     
                kl_loss = criterion_kl(F.log_softmax(adv_out, dim=1), F.softmax(teacher_out.detach(), dim=1))
                loss = XENT_loss(adv_out,y) + config.alpha * (1.0 / N) *  kl_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
            adjust_learning_rate(lr, optimizer, epoch, epochs)
        return


def adjust_learning_rate(lr, optimizer, epoch, epochs):
    criteria = epochs // 2
    max_lr = lr
    if epoch + 1 <= criteria :
        lr = max_lr/criteria * (epoch + 1)
    else  :
        # lr = max_lr/criteria * (10 - epoch)
        lr = (1/(2**(epoch + 1 - criteria)))*max_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr