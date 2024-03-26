# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:36:46 2024

@author: TEJA
"""
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage import io
import numpy as np
import torch

import torch.optim as optim
from models import FaceNetModel

from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class casia_dataset(Dataset):
    def __init__(self,root_dir, csv_file ,phase='train', transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file,dtype={'id': object, 'name': object, 'class': object})
        self.root_dir = root_dir
        self.transform = transform
        self.phase=phase
        self.class_to_int_map = {cls: i for i, cls in enumerate(self.data['class'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'name']
        label = self.data.loc[idx, 'class']
        img_path=os.path.join(self.root_dir,str(label),str(img_name)+'.jpg')
        image = io.imread(img_path)  # Load image as RGB
        #label = self.data.loc[idx, 'class']
        label = self.class_to_int_map[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)  # Apply transformations if any
        
        sample={
            'image': image,
            'label': label}
        return sample
    
normalize = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomRotation(15),
    transforms.RandomResizedCrop(199),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():
        param.requires_grad=True

# Step 4: Define loss and optimizer
model = FaceNetModel()
trained_model_path='/kaggle/working/models/pins_unlearned_model.pth'
trained_model=torch.load(trained_model_path)
model.load_state_dict(trained_model)

freeze_layers(model)
criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.075,momentum=0.9)

train_ds=casia_dataset(root_dir="/kaggle/input/pins-aligned/105_classes_pins_dataset",
                       csv_file='files/pins.csv',
                       transform=normalize)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

valid_ds=casia_dataset(root_dir="/kaggle/input/cplfw/aligned",
                       csv_file='files/lfwd.csv',
                       transform=normalize)
valid_loader = DataLoader(valid_ds, batch_size=1024, shuffle=True)

# Step 5: Training loop
num_epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_acc= 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for sample in tqdm(train_loader):
        
        images, labels = sample['image'],sample['label']
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item() * images.size(0)
    
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100*(correct / total)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    if epoch_accuracy > best_acc:
        best_acc=epoch_accuracy
        torch.save(model.state_dict(), f"log/fc_finetune.pth")
        

# Step 6: Evaluation (optional)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sample in valid_loader:
        images, labels = sample['image'], sample['label']
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward_classifier(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")