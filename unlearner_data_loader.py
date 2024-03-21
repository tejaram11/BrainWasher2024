# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:27:35 2024

@author: TEJA
"""
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import Subset
from PIL import Image
import torch
import numpy as np




class casia_dataset(Dataset):
    def __init__(self,root_dir, csv_file ,phase='train', transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file,dtype={'id': object, 'name': object, 'class': object})
        self.root_dir = root_dir
        self.transform = transform
        if phase=='valid':
            self.class_to_int_map = {cls: i for i, cls in enumerate(self.data['class'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'name']
        label = self.data.loc[idx, 'class']
        img_path=os.path.join(self.root_dir,str(label),str(img_name)+'.jpg')
        image = Image.open(img_path).convert('RGB')  # Load image as RGB
        label = self.data.loc[idx, 'class']
        label = self.class_to_int_map[self.label]
        label= torch.from_numpy(np.array([self.label]).astype('long'))

        if self.transform:
            image = self.transform(image)  # Apply transformations if any
        
        sample={
            'image': image,
            'label': label}
        return sample

        
def get_dataset(batch_size) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        # utils
        normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        kaggle_dir='/kaggle/working/BrainWasher2024/'
        dataset_dir="/kaggle/input/casia-webface/casia-webface"
        valid_dataset_dir='/kaggle/input/cplfw/aligned'
        
        retain_csv='files/casia_retain_set.csv'
        forget_csv='files/casia_forget_set.csv'
        valid_csv='files/lfwd.csv'
        

        # create dataset
         
        retain_ds = casia_dataset(root_dir=dataset_dir,csv_file=retain_csv,transform=normalize)
        forget_ds = casia_dataset(root_dir=dataset_dir,csv_file=forget_csv,transform=normalize)
        val_ds = casia_dataset(root_dir=valid_dataset_dir,csv_file=valid_csv,phase='valid',transform=normalize)
        
        
        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        return retain_loader, forget_loader, validation_loader