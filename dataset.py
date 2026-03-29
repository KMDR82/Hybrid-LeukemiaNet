import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class StandardDataset(Dataset):
    """Dataset class for standard training and evaluation."""
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self): 
        return len(self.paths)
        
    def __getitem__(self, idx): 
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

class TTADataset(Dataset):
    """Dataset class for Test-Time Augmentation evaluation."""
    def __init__(self, paths, labels, transforms_list):
        self.paths = paths
        self.labels = labels
        self.transforms_list = transforms_list
        
    def __len__(self): 
        return len(self.paths)
        
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # Apply all transforms and stack them into a single tensor
        augmented_images = torch.stack([t(img) for t in self.transforms_list])
        return augmented_images, self.labels[idx]

def get_transforms():
    """Returns the transformation pipelines used in the study."""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ft_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(), # TTA'da olduğu için eğitimde de olması mantıklı
        transforms.RandomRotation(15),     # Notebook'tan eklendi
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Değerler notebook ile eşitlendi
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tta_transforms = [
        base_transform,
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ]
    
    return base_transform, ft_transform, tta_transforms