import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets
from torchvision import models

from train import train, compute_class_weights


random.seed(0)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_dataset = datasets.ImageFolder("./data/yoga16-dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("./data/yoga16-dataset/val", transform=test_transform)
test_dataset = datasets.ImageFolder('./data/yoga16-dataset/test', transform=test_transform)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")

    return obj


model = models.mobilenet_v2()

model.classifier[1] = nn.Linear(model.last_channel, 16)
model = to_device(model)

adamW_params = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-8
}

class_weights = compute_class_weights("./data/dataset-16/train")

train(model, 15, train_loader, val_loader, test_loader, loss_func=nn.CrossEntropyLoss(weight=class_weights) ,optimizer_params=adamW_params)

