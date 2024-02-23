## -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import datasets, transforms, models
import os
import matplotlib.image as mpimg
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score 
import csv
import warnings 

os.chdir(os.getcwd())
warnings.filterwarnings('ignore')
from cData import ImageDataset

transform1 = transforms.Compose([          
            transforms.ToTensor(),
            transforms.Resize([150, 150])])

train_data = ImageDataset(annotations_file = 'tab3.csv', img_dir = 'test_data', transform=transform1)
train_size = int(len(train_data) * 0.8)
test_size = len(train_data) - train_size

train_data, test_data = torch.utils.data.random_split(train_data, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #150*150*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3,3)) 
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(3,3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*17*17, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)

    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        
        return x
    




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


def train(model, loss_fn, optimizer, n_epoch=5):

    # цикл обучения сети
    for epoch in range(n_epoch):

        model.train(True)

        running_losses = []
        running_accuracies = []
        for i, batch in enumerate(train_loader):
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
        model.train(False)
        print("Epoch:", str(epoch+1) + ", accuracy:", (sum(running_accuracies) * 100 / len(running_accuracies)).numpy())
        
    return model

def create_model(model, num_freeze_layers, num_out_classes):
    model.fc = nn.Linear(512, num_out_classes)

    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model

model = ConvNet()
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train(model, loss_fn, optimizer, n_epoch=10)

train_accuracy, _ = evaluate(model, train_loader, loss_fn)
print('Train accuracy:', train_accuracy.numpy())

test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Test accuracy:', test_accuracy.numpy())
