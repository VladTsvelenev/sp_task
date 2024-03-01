## -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
import os
import matplotlib.image as mpimg
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score 
import csv
import warnings 
from torch.utils.tensorboard import SummaryWriter
from cData import ImageDataset
from ConvNetF import ConvNet


os.chdir(os.getcwd())
warnings.filterwarnings('ignore')

transform1 = transforms.Compose([         
            v2.ToTensor(),
            v2.Resize([150, 150])])

train_data = ImageDataset(annotations_file = 'table.csv', img_dir = 'test_data', transform=transform1)
train_size = int(len(train_data) * 0.70)
val_size = int(len(train_data) * 0.20)
test_size = len(train_data) - train_size - val_size


train_data, val_data, test_data = torch.utils.data.random_split(train_data, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
val_loder = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False)


train_dataD = ImageDataset(annotations_file = 'other3.csv', img_dir = 'test_data', transform=transform1)
train_sizeD = int(len(train_dataD) * 0.70)
val_sizeD = int(len(train_dataD) * 0.20)
test_sizeD = len(train_dataD) - train_sizeD - val_sizeD

train_dataD, val_dataD, test_dataD = torch.utils.data.random_split(train_dataD, [train_sizeD, val_sizeD, test_sizeD])

train_loaderD = torch.utils.data.DataLoader(train_dataD, batch_size=20, shuffle=True)
val_loderD = torch.utils.data.DataLoader(val_dataD, batch_size=20, shuffle=True)
test_loaderD = torch.utils.data.DataLoader(test_dataD, batch_size=20, shuffle=False)

def evaluate(model, dataloader, loss_fn):
    
    losses = []

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):
        
        X_batch, y_batch = batch
        num_elements += y_batch.size(0)
        with torch.no_grad():
            logits = model(X_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))
            losses.append(loss.item())
            
            y_pred = torch.argmax(logits, dim=1).cpu()
            
            num_correct += torch.sum(y_pred == y_batch)
    accuracy = num_correct / num_elements
            
    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epoch, load, val):
    num_iter = 0
    minloss = 0
    for epoch in range(n_epoch):

        model.train(True)

        running_losses = []
        running_accuracies = []
        for i, batch in enumerate(load):
            X_batch, y_batch = batch
            logits = model(X_batch.to(device))
            
            loss = loss_fn(logits, y_batch.to(device))
            running_losses.append(loss.item())

            loss.backward()
            optimizer.step() 
            optimizer.zero_grad() 
            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_batch == model_answers.cpu()) / len(y_batch)
            running_accuracies.append(train_accuracy)
            num_iter += 1

        val_accuracy, val_loss = evaluate(model, val, loss_fn=loss_fn)
            
        model.train(False)
        accuracy = (sum(running_accuracies) * 100 / len(running_accuracies)).numpy()
        losst = sum(running_losses) * 100 / len(running_losses)
        print("Epoch:", str(epoch+1))
        print("Train | accuracy:", accuracy, ", loss:", losst)
        print("Validation | accuracy:", val_accuracy.numpy() * 100, ", loss:", val_loss)
        print("__________________________________________________________")
        if val_loss <= minloss:
            torch.save(model.state_dict(), './model.pt')
            minloss = val_loss
        
    return model

def create_model(model, num_freeze_layers, num_out_classes):
    model.fc3 = nn.Linear(512, num_out_classes)
    model.load_state_dict(torch.load('model.pt'))
    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model

model = ConvNet(4)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train(model, loss_fn, optimizer, 10, train_loader, val_loder)

model = create_model(model, 3, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = train(model, loss_fn, optimizer, 20, train_loaderD, val_loderD)
model.load_state_dict(torch.load('model.pt'))


train_accuracy, _ = evaluate(model, train_loaderD, loss_fn)
print('Train accuracy:', train_accuracy.numpy())

test_accuracy, _ = evaluate(model, test_loaderD, loss_fn)
print('Test accuracy:', test_accuracy.numpy())