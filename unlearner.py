# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:37:06 2024

@author: TEJA
"""

import torch 
from BrainWasher_algorithm import BrainWasher
from utils_inceptionresnetv2 import InceptionResNetV2

trained_model_path=''

def prepare_dataset(dataset_path,forget_class):
    forget_class=forget_class
    

BrainWasher_Inception=BrainWasher()
model=InceptionResNetV2()

model.load_state_dict(torch.load(trained_model_path))

retain_loader,forget_loader,validation_loader= prepare_dataset()
model_forget=BrainWasher_Inception.unlearning(model, retain_loader, forget_loader, validation_loader)
forget_state=model_forget.state_dict()
torch.save(forget_state,'unlearned_model.pth')


