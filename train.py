# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:55:08 2024

@author: TEJA
"""


import datetime
import time
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler
import torch_xla.core.xla_model as xm
#from pretrainedmodels import inceptionresnetv2


from utils import ModelSaver, init_log_just_created
from utils_inceptionresnetv2 import InceptionResNetV2
from loss import TripletLoss
from data_loader import get_dataloader
from eval_metrics import evaluate, plot_roc
from write_csv_for_making_dataset import write_csv




learning_rate=0.01
step_size=50
num_epochs=50
margin = 0 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
modelsaver = ModelSaver()

train_root_dir="./datasets/faces_webface_112x112/MS1M_112x112"
valid_root_dir=""
train_csv_name= "./files/casia_details_final.csv"
valid_csv_name= ""
num_train_triplets= 10000
num_valid_triplets= 10000
batch_size=16
num_workers=1
num_classes=10572
unfreeze=[]

os.environ['PJRT_DEVICE']='TPU'

device=xm.xla_device()




def main():
    init_log_just_created("log/valid.csv")
    init_log_just_created("log/train.csv")
    
    valid = pd.read_csv('log/valid.csv')
    max_acc = valid['acc'].max()
    

    #pretrain = args.pretrain
    #fc_only = args.fc_only
    #except_fc = args.except_fc
    #train_all = args.train_all
    #unfreeze = args.unfreeze.split(',')
    #freeze = args.freeze.split(',')
    #start_epoch = 0
    #print(f"Transfer learning: {pretrain}")
    #print("Train fc only:", fc_only)
    #print("Train except fc:", except_fc)
    #print("Train all layers:", train_all)
    #print("Unfreeze only:", ', '.join(unfreeze))
    #print("Freeze only:", ', '.join(freeze))
    #print(f"Max acc: {max_acc:.4f}")
    #print(f"Learning rate will decayed every {args.step_size}th epoch")
    model = InceptionResNetV2(num_classes)
    model.to(device)
    triplet_loss = TripletLoss(margin).to(device)

    

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False,
        weight_decay=1e-5
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    

    for epoch in range(0, num_epochs):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, num_epochs))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(train_root_dir, valid_root_dir,
                                                 train_csv_name, valid_csv_name,
                                                 num_train_triplets, num_valid_triplets,
                                                 batch_size, num_workers)

        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')


def save_last_checkpoint(state):
    xm.save(state, 'log/last_checkpoint.pth')

def save_if_best(state, acc):
    modelsaver.save_if_best(acc, state)
    
def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
     for phase in ['train', 'valid']:
    #for phase in ['train']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            # pos_cls = batch_sample['pos_class'].to(device)
            # neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                # pos_hard_cls = pos_cls[hard_triplets]
                # neg_hard_cls = neg_cls[hard_triplets]

                model.module.forward_classifier(anc_hard_img)
                model.module.forward_classifier(pos_hard_img)
                model.module.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    #optimizer.step()
                    xm.optimizer_step(optimizer)
                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_lr()))
        layers = '+'.join(unfreeze.split(','))
        write_csv(f'log/{phase}.csv', [time, epoch, np.mean(accuracy), avg_triplet_loss, layers, batch_size, lr])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  })
            save_if_best({'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, np.mean(accuracy))
        else:
            plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    main()
