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
import torch_xla.debug.metrics as met
#from pretrainedmodels import inceptionresnetv2


from utils import ModelSaver, init_log_just_created
from utils_inceptionresnetv2 import InceptionResNetV2
from loss import TripletLoss
from data_loader import get_dataloader
from eval_metrics import evaluate, plot_roc
from write_csv_for_making_dataset import write_csv
from train import train_valid

import os
os.environ['PJRT_DEVICE']='TPU'

import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

learning_rate=0.01*xm.xrt_world_size()
step_size=50
num_epochs=50
margin = 0.1 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
modelsaver = ModelSaver()

train_root_dir="/home/TEJA/datasets/MS1M_112x112"
valid_root_dir="/home/TEJA/datasets/aligned"
train_csv_name= "files/casia_full.csv"
valid_csv_name= "files/lfwd.csv"
num_train_triplets= 4096
num_valid_triplets= 4096
batch_size=32
num_workers=4
num_classes=10572
unfreeze=[]

device=xm.xla_device()

os.environ['PJRT_DEVICE']='TPU'



def save_last_checkpoint(state):
    xm.save(state, 'log/last_checkpoint.pth')

def save_if_best(state, acc):
    modelsaver.save_if_best(acc, state)

def _mp_fn(index):
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
    print(device)
    triplet_loss = TripletLoss(margin).to(device)

    

    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,eps=1e-08,weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    

    for epoch in range(0, num_epochs):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, num_epochs))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(train_root_dir, valid_root_dir,
                                                 train_csv_name, valid_csv_name,
                                                 num_train_triplets, num_valid_triplets,
                                                 batch_size, num_workers)
        data_loaders= pl.MpDeviceLoader(data_loaders, device)
        print("data loaded")
        print(f'  Execution time                 = {time.time() - time0}')
        

        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')

if __name__== '__main__':
    
    xmp.spawn(_mp_fn,args=())