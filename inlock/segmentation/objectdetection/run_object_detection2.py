import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from utils import *
from model import *
import os
import wandb


import torch
import torchvision
from torchvision import ops
from torchvision import ops, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# Initialize WandB
wandb.init(project="floor_plan_training")

# Ensure device is set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

class ObjectDetectionDataset(Dataset):
    def __init__(self, annotation_path, img_dir, img_size, name2idx, augment=False):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()

    def __getitem__(self, idx):
        img = self.img_data_all[idx]
        if self.augment:
            img = self.transform(img)
        return img, self.gt_bboxes_all[idx], self.gt_classes_all[idx]

    def __len__(self):
        return self.img_data_all.size(dim=0)

    
    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
        for i, img_path in enumerate(img_paths):
            if (not img_path) or (not os.path.exists(img_path)):
                continue
            img = io.imread(img_path)
            img = resize(img, self.img_size)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])
            img_data_all.append(img_tensor)
            gt_idxs_all.append(gt_idx)
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        img_data_stacked = torch.stack(img_data_all, dim=0)
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad


img_width =2100 # Change the width as needed
img_height = 1500  # Change the height as needed
annotation_path = "data/annotations12.xml"
image_dir = os.path.join("data", "images")
name2idx = {'pad': -1, 'doors': 0, 'stairs': 1}
img_size = (img_height, img_width)
od_dataset = ObjectDetectionDataset(annotation_path, image_dir, img_size, name2idx)
od_dataloader = DataLoader(od_dataset, batch_size=4)


for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
    img_data_all = img_batch.to(device)
    gt_bboxes_all = gt_bboxes_batch.to(device)
    gt_classes_all = gt_classes_batch.to(device)
    break

img_data_all = img_data_all[:2]
gt_bboxes_all = gt_bboxes_all[:2]
gt_classes_all = gt_classes_all[:2]

model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers).to(device)

for param in backbone.named_parameters():
    param[1].requires_grad = True

out = backbone(img_data_all)
out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h
height_scale_factor, width_scale_factor
print( out_c, out_h, out_w)

def training_loop(model, learning_rate, train_dataloader, n_epochs, model_save_path):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    epoch = 0 
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            
            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        
        # Log the loss to WandB
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss}")
        
        loss_list.append(total_loss)
        epoch +=1
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f"{model_save_path}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        

    # Save the model at the end of training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return loss_list

learning_rate = 1e-3
n_epochs = 1000  # Set the number of epochs
model_save_path = "object_detection5_model.pt"  # Path to save the model

# Ensure model is moved to device
model.to(device)
img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes = len(name2idx) - 1 # exclude pad idx
roi_size = (2, 2)

detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)

# Ensure model is moved to device
model.to(device)

training_loop(detector, learning_rate, od_dataloader, n_epochs, model_save_path)