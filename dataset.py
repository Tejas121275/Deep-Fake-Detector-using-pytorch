import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import numpy as np
from sklearn.model_selection import KFold

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, use_mtcnn=True):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.use_mtcnn = use_mtcnn
        self.mtcnn = MTCNN(image_size=224, margin=20, device='cuda' if torch.cuda.is_available() else 'cpu') if use_mtcnn else None
        self.real_images = [(os.path.join(real_dir, img), 0) for img in os.listdir(real_dir) if img.endswith(('.jpg', '.png'))]
        self.fake_images = [(os.path.join(fake_dir, img), 1) for img in os.listdir(fake_dir) if img.endswith(('.jpg', '.png'))]
        # Balance dataset
        min_size = min(len(self.real_images), len(self.fake_images))
        self.real_images = self.real_images[:min_size]
        self.fake_images = self.fake_images[:min_size]
        self.all_images = self.real_images + self.fake_images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.use_mtcnn and self.mtcnn:
                img_array = np.array(image)
                boxes, _ = self.mtcnn.detect(img_array)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    image = image.crop((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(real_dir, fake_dir, batch_size=64, use_mtcnn=True):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = DeepfakeDataset(real_dir, fake_dir, use_mtcnn=use_mtcnn)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def get_kfold_data_loaders(real_dir, fake_dir, batch_size=64, k_folds=5, use_mtcnn=True):
    full_dataset = DeepfakeDataset(real_dir, fake_dir, use_mtcnn=use_mtcnn)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    data_loaders = []
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for train_idx, val_idx in kf.split(range(len(full_dataset))):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataset = DeepfakeDataset(real_dir, fake_dir, transform=train_transform, use_mtcnn=use_mtcnn)
        val_dataset = DeepfakeDataset(real_dir, fake_dir, transform=val_transform, use_mtcnn=use_mtcnn)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4, pin_memory=True
        )
        data_loaders.append((train_loader, val_loader))
    return data_loaders