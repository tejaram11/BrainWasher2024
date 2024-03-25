# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:44:58 2024

@author: TEJA
"""

import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train import train_valid
from data_loader import get_dataloader
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from loss import TripletLoss
import time
from tqdm import tqdm


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
print(DEVICE)



class BrainWasher:
    def __init__(self,USE_MOCK=False):
        self.USE_MOCK=USE_MOCK
        
    def evaluation(self,net, dataloader, criterion, device = 'cuda:0'): ##evaluation function
        net.eval()
        total_samp = 0
        total_acc = 0
        total_loss = 0.0
        for sample in dataloader:
            images, labels = sample['image'].to(device), sample['label'].to(device)
            _pred = net.forward_classifier(images)
                       
            total_samp+=len(labels)
            #print(f'total_samp={total_samp}')
            loss = criterion(_pred, labels)
            _, _pred = torch.max(_pred,1)
            total_loss += loss.item()
            total_acc+=(_pred == labels).float().sum().item()
            #print(f'total_acc={total_acc}')
            #print(f'total_sample={total_samp}')
            mean_loss = total_loss / len(dataloader)
            mean_acc = total_acc/total_samp
            print(f'loss={mean_loss}')
            print(f'acc={mean_acc}')
            return 
    
    def kl_loss_sym(self,x,y):
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        return kl_loss(nn.LogSoftmax(dim=-1)(x),y)
    def unlearning(self,
            net,
            retain_loader,
            forget_loader,
            validation_loader
            ):
        """Simple unlearning by finetuning."""
        print('-----------------------------------')
        for param in net.parameters():
            param.requires_grad = True
        epochs = 8
        retain_bs = 64
        criterion = nn.CrossEntropyLoss()
        margin= 0.7
        optimizer = optim.SGD(net.parameters(), lr=0.005,
                              momentum=0.9, weight_decay=0)
        optimizer_retain = optim.SGD(net.parameters(), lr=0.001*retain_bs/64, momentum=0.9, weight_decay=1e-2)
        ##the learning rate is associated with the batchsize we used
        optimizer_forget = optim.SGD(net.parameters(), lr=3e-4, momentum=0.9, weight_decay=0)
        total_step = int(len(forget_loader)*epochs)
        retain_ld = DataLoader(retain_loader.dataset, batch_size=retain_bs, shuffle=True)
        retain_ld4fgt = DataLoader(retain_loader.dataset, batch_size=retain_bs, shuffle=True)
        scheduler = CosineAnnealingLR(optimizer_forget, T_max=total_step, eta_min=1e-6)
        scheduler_finetune= StepLR(optimizer_retain,step_size=20, gamma=0.1)
        triplet_loss=TripletLoss(margin).to(DEVICE)
        net.to(DEVICE)
        if self.USE_MOCK: ##Use some Local Metric as reference
            net.eval()
            print('Forget')
            self.evaluation(net, forget_loader, criterion)
            print('Valid')
            self.evaluation(net, validation_loader, criterion)
        net.train()
        time0 = time.time()
        print("First stage")
        for sample in tqdm(forget_loader): ##First Stage 
            
            inputs = sample["image"]
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            outputs = net.forward_classifier(inputs)
            uniform_label = torch.ones_like(outputs).to(DEVICE) / outputs.shape[1] ##uniform pseudo label
            loss = self.kl_loss_sym(outputs, uniform_label) ##optimize the distance between logits and pseudo labels
            loss.backward()
            optimizer.step()
        print(f'  Execution time                 = {time.time() - time0}')
        if self.USE_MOCK:
            print('Forget')
            self.evaluation(net,forget_loader,criterion)
            print('Valid')
            self.evaluation(net, validation_loader,criterion)
            print(f'epoch={epochs} and retain batch_sz={retain_bs}') 
        net.train()
        print(f'  Execution time = {time.time() - time0}')
        print("second stage")
        for ep in range(epochs): ##Second Stage
            time0=time.time()
            net.train()
            for sample_id,(sample_forget, sample_retain) in tqdm(enumerate(zip(forget_loader, retain_ld4fgt))):##Forget Round
                #if sample_id%25==0:
                #    print(f'{sample_id}/{len(forget_loader)}')
                t = 1.15 ##temperature coefficient
                inputs_forget,inputs_retain = sample_forget["image"],sample_retain['image']
                inputs_forget, inputs_retain = inputs_forget.to(DEVICE), inputs_retain.to(DEVICE)
                optimizer_forget.zero_grad()
                outputs_forget,outputs_retain = net(inputs_forget),net(inputs_retain).detach()
                loss = (-1 * nn.LogSoftmax(dim=-1)(outputs_forget @ outputs_retain.T/t)).mean() ##Contrastive Learning loss
                loss.backward()
                optimizer_forget.step()
                scheduler.step()
            print(f'  Execution time = {time.time() - time0}')
             ##Retain Round
            #triplet_loader = { x: torch.utils.data.DataLoader(x.dataset, batch_size=16, shuffle=False, num_workers=1) for x in [retain_loader, validation_loader]}
            #triplet_data_size = {x: len(x.dataset) for x in [retain_loader, validation_loader]}
            triplet_loader, triplet_data_size = get_dataloader("/kaggle/input/casia-webface/MS1M_112x112",
                                                     "/kaggle/input/cplfw/aligned",
                                                     "files/casia_retain_set.csv",
                                                     "files/lfwd.csv",
                                                     30000, 512,
                                                     64,1,ep)
            train_valid(net,optimizer_retain,triplet_loss,scheduler_finetune,ep,triplet_loader,triplet_data_size)
                
            '''
                inputs, labels = sample["image"],sample["age_group"]
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer_retain.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_retain.step()
            '''
            print(f'epoch: {ep}')
            print(f'  Execution time = {time.time() - time0}')    
            if self.USE_MOCK: 
                print(f'epoch {ep}:')
                print('Retain')
                self.evaluation(net, retain_ld, criterion)
                print('Forget')
                self.evaluation(net, forget_loader, criterion)
                print('Valid')
                self.evaluation(net, validation_loader, criterion)
        print('-----------------------------------')
        return net