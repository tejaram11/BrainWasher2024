# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:55:08 2024

@author: TEJA
"""

import signal,sys
import datetime
import time


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler
#import torch_xla.core.xla_model as xm
#import torch_xla.debug.metrics as met
#from pretrainedmodels import inceptionresnetv2


from utils import ModelSaver, init_log_just_created
from utils_inceptionresnetv2 import InceptionResNetV2
from models import FaceNetModel
from loss import TripletLoss
from data_loader import get_dataloader
from eval_metrics import evaluate, plot_roc
from write_csv_for_making_dataset import write_csv



learning_rate=0.075
step_size=25
num_epochs=100

margin = 0.2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
modelsaver = ModelSaver()
torch.cuda.empty_cache()




kaggle_dir= "/kaggle/working/BrainWasher2024/"
train_root_dir="/kaggle/input/pins-aligned/105_classes_pins_dataset"
valid_root_dir="/kaggle/input/cplfw-mtcnn/cplfw_aligned"
train_csv_name= "files/pins.csv"
valid_csv_name= "files/lfw.csv"
num_train_triplets= 20000
num_valid_triplets= 512
batch_size=128
num_workers=1
load_best=False
load_last=False
continue_step=False

num_classes=10572
unfreeze=[]


#os.environ['PJRT_DEVICE']='TPU'

#device=xm.xla_device()
def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
     #time0 = time.time()
     for phase in ['train', 'valid']:
     #for phase in ['valid']:
        

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            
            model.train()
        else:
            model.eval()
        
        print(phase)

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):
            if batch_idx % 100 == 0:  # Print every 100 batches
                #xm.master_print(met.metrics_report())
                print(f"Batch [{batch_idx}/{len(dataloaders[phase])}]")

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)
            

            #print("forward pass")
            #print(f'  Execution time                 = {time.time() - time0}')

            # pos_cls = batch_sample['pos_class'].to(device)
            # neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)
                
                #print("gradient calc")
                #print(f'  Execution time                 = {time.time() - time0}')
                

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                neg_dist = neg_dist.to(device)
                pos_dist = pos_dist.to(device)

                
                # Calculate condition and move result to host CPU as NumPy array
                # Assuming margin is a constant value
                first_condition = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                second_condition = (pos_dist < neg_dist).cpu().numpy().flatten()
                all = (np.logical_and(first_condition, second_condition))
                all = torch.tensor(all)
                #all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = torch.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = torch.where(all >= 0)

                anc_embed = anc_embed[hard_triplets]
                pos_embed = pos_embed[hard_triplets]
                neg_embed = neg_embed[hard_triplets]
                
                '''
                anc_img = anc_img[hard_triplets]
                model.modules.forward_classifier(anc_img.to(device))
                anc_img = pos_img[hard_triplets]
                model.modules.forward_classifier(anc_img.to(device))
                anc_img = neg_img[hard_triplets]
                model.modules.forward_classifier(anc_img.to(device))
                '''
                # pos_hard_cls = pos_cls[hard_triplets]
                # neg_hard_cls = neg_cls[hard_triplets]
                
                #print("gradients")
                #print(f'  Execution time                 = {time.time() - time0}')
                #print("loss calc")
                triplet_loss = triploss.forward(anc_embed, pos_embed, neg_embed)
                #print(f'  Execution time                 = {time.time() - time0}')

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()
                    #xm.optimizer_step(optimizer)
                    #print("backprop")
                #print(f'  Execution time                 = {time.time() - time0}')
            
                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()
        
        scheduler.step()
        if scheduler.last_epoch % scheduler.step_size == 0:
            print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
        
        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        time_ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_lr()))
        layers = 'all'
        write_csv(f'log/{phase}.csv', [time_, epoch, np.mean(accuracy), avg_triplet_loss, layers, batch_size, lr])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  })
            save_if_best({'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, np.mean(accuracy))
        else:
            plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))

def save_last_checkpoint(state):
    torch.save(state, "log/last_checkpoint.pth")

def save_if_best(state, acc):
    modelsaver.save_if_best(acc, state)

def compute_l2_distance(x1, x2):
  # Move tensors to TPU device
  x1 = x1.to(device)
  x2 = x2.to(device)

  # Calculate squared Euclidean distance (L2 norm)
  diff = x1 - x2
  distances = torch.sum(diff * diff, dim=1)  # Sum squares along feature dimension
  return distances




def main():
    
    
    init_log_just_created("log/valid.csv")
    init_log_just_created("log/train.csv")
    
    valid = pd.read_csv('log/valid.csv')
    max_acc = valid['acc'].max()
    start_epoch=0


    model = FaceNetModel()
    model.unfreeze_all()
    model.to(device)
    print(device)
    triplet_loss = TripletLoss(margin).to(device)
    #optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    #optimizer = optim.Adagrad(params=model.parameters(), lr=learning_rate, lr_decay=0, initial_accumulator_value=0.1, eps=1e-10, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    def handle_interrupt(signal, frame):
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), "models/interrupted_model.pt")
        torch.save(optimizer.state_dict(),"models/optimizer_state.pt")
        print(epoch)
        sys.exit(0)

    # Register the Ctrl+C signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    if load_best or load_last:
        checkpoint = 'log/best_state.pth' if load_best else 'log/last_checkpoint.pth'
        print('loading', checkpoint)
        checkpoint = torch.load(checkpoint)
        modelsaver.current_acc = max_acc
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        print("Stepping scheduler")
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        except ValueError as e:
            print("Can't load last optimizer")
            print(e)
        
        if continue_step:
            scheduler.step(checkpoint['epoch'])
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}\n"
              f"Loaded checkpoint accuracy: {checkpoint['accuracy']}\n"
              f"Loaded checkpoint loss: {checkpoint['loss']}")

    
    #try:
     #model = torch.nn.DataParallel(model)
    for epoch in range(start_epoch, num_epochs):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, num_epochs))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(train_root_dir, valid_root_dir,
                                                 train_csv_name, valid_csv_name,
                                                 num_train_triplets, num_valid_triplets,
                                                 batch_size, num_workers,epoch)
        #print("data loaded")
        #print(f'  Execution time                 = {time.time() - time0}')
        
        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')
    
    '''
    #except:
        print("Training excepted. Saving model...")
        torch.save(model.state_dict(), "models/interrupted_model_except.pt")
        torch.save(optimizer.state_dict(),"models/optimizer_state.pt")
    '''
        



    

if __name__ == '__main__':
    
    

    main()
