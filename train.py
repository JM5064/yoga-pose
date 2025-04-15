import random
import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torchvision
from torchvision import transforms, datasets

from convnext import ConvNeXt


random.seed(0)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('data/dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('data/dataset/val', transform=transform)
test_dataset = datasets.ImageFolder('data/dataset/test', transform=transform)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False ,num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False ,num_workers=2)


def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")

    return obj


def log_results(file, acc, precision, recall, f1, average_train_loss, average_val_loss):
    file.write(f'Train Loss: {average_train_loss}\tValidation Loss: {average_val_loss}\t')
    file.write(f'Accuracy: {acc}\tPrecision: {precision}\tRecall: {recall}\tF1-score: {f1}')
    file.write('\n')


def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = to_device(inputs)
            labels = to_device(labels)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_val_loss = running_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return acc, precision, recall, f1, average_val_loss


def train(model, num_epochs, train_loader, val_loader, optimizer=optim.AdamW, optimizer_params=None, log_dir="logs"):
    logfile = open(log_dir + "/" + datetime.now(), "a")

    loss_func = nn.CrossEntropyLoss()
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs = to_device(inputs)
            labels = to_device(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)
        acc, precision, recall, f1, average_val_loss = validate(model, val_loader, loss_func)
        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss}\tValidation Loss: {average_val_loss}')
        print(f'Accuracy: {acc}\tPrecision: {precision}\tRecall: {recall}\tF1-score: {f1}')

        log_results(logfile, acc, precision, recall, f1, average_train_loss, average_val_loss)


model = ConvNeXt(layer_distribution=[3,3,9,3], num_classes=81)
model = to_device(model)
adamW_params = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-8
}

train(model, num_epochs=1, train_loader=train_loader, val_loader=val_loader, optimizer_params=adamW_params)
torch.save(model.state_dict(), 'model.pth')