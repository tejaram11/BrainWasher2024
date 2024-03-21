# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:37:06 2024

@author: TEJA
"""


import torch 
from torch.utils.data import DataLoader, Dataset
from BrainWasher_algorithm import BrainWasher
from unlearner_data_loader import get_dataset
from utils_inceptionresnetv2 import InceptionResNetV2
from facenet_pytorch import InceptionResnetV1


trained_model_path='/kaggle/input/facenet-models/best_state_87.pth'
        

BrainWasher_Inception=BrainWasher(USE_MOCK=True)
model=InceptionResnetV1(pretrained=None)

model.load_state_dict(trained_model_path['state_dict'])

retain_loader,forget_loader,validation_loader= get_dataset(64)
model_forget=BrainWasher_Inception.unlearning(model, retain_loader, forget_loader, validation_loader)
forget_state=model_forget.state_dict()
torch.save(forget_state,'/kaggle/working/models/unlearned_model.pth')


