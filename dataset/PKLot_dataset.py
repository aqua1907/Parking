import torch
import numpy as np
import os
import torchvision.transforms as T
import random

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import CFG


class PKLot_dataset(Dataset):
    def __init__(self, images_list, class_names, transform=None):
        self.paths = images_list
        self.transform = transform
        self.class_names = class_names

    def __getitem__(self, idx):

        image = Image.open(self.paths[idx])
        label_str = self.paths[idx].split("\\")[-2]
        target = self.class_names.index(label_str)

        image = self.transform(image)
        target = torch.as_tensor([target], dtype=torch.float32)

        return image, target

    def __len__(self):
        return len(self.paths)


def get_transforms(train, prob=None):
    transforms = [T.Resize([120, 120])]
    if train:
        if random.random() < prob:
            transforms.append(T.ColorJitter(brightness=0.25, contrast=0.25,
                                            saturation=0.25, hue=0.25))
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def train_val_loaders(root_dir):
    images_list = []
    root_dir = Path(root_dir)
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            images_list.append(os.path.join(root, name))

    np.random.shuffle(images_list)
    num_train = len(images_list)
    split = int(np.floor(CFG["valid_size"] * num_train))
    train_images, valid_images = images_list[split:], images_list[:split]

    transform = get_transforms(train=True, prob=0.5)
    train_data = PKLot_dataset(train_images, CFG["CLASSES"], transform=transform)

    transform = get_transforms(train=False, prob=0.5)
    val_data = PKLot_dataset(valid_images, CFG["CLASSES"], transform=transform)

    train_loader = DataLoader(train_data, batch_size=CFG["batch_size"], num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=CFG["batch_size"], num_workers=0, drop_last=True)

    return train_loader, val_loader


