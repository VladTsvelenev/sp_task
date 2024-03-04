# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import warnings
from cData import ImageDataset
from ConvNetF import ConvNet

os.chdir(os.getcwd())
warnings.filterwarnings('ignore')


def evaluate(model, dataloader, loss_fn):

    losses = []

    num_correct = 0
    num_elements = 0

    for _, batch in enumerate(dataloader):

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
            train_accuracy = torch.sum(
                y_batch == model_answers.cpu()) / len(y_batch)
            running_accuracies.append(train_accuracy)
            num_iter += 1

        val_accuracy, val_loss = evaluate(model, val, loss_fn=loss_fn)

        model.train(False)
        accuracy = (sum(running_accuracies) * 100 /
                    len(running_accuracies)).numpy()
        losst = sum(running_losses) * 100 / len(running_losses)
        print("Epoch:", str(epoch+1))
        print("Train | accuracy:", accuracy, ", loss:", losst)
        print("Validation | accuracy:", val_accuracy.numpy()
              * 100, ", loss:", val_loss)
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


def createdata(annotation, dirname, isval):
    global train_data, val_data, test_data
    train_data_temp = ImageDataset(
        annotations_file=annotation, img_dir=dirname, transform=transform1)
    if isval == True:
        train_size_temp = int(len(train_data_temp) * 0.70)
        val_size_temp = int(len(train_data_temp) * 0.20)
        test_size_temp = len(train_data_temp) - train_size_temp - val_size_temp
        train_data_temp, val_data_temp, test_data_temp = torch.utils.data.random_split(
            train_data_temp, [train_size_temp, val_size_temp, test_size_temp])
    else:
        train_size_temp = int(len(train_data_temp) * 0.80)
        test_size_temp = len(train_data_temp) - train_size_temp
        train_data_temp, test_data_temp = torch.utils.data.random_split(
            train_data_temp, [train_size_temp, test_size_temp])
    if train_data == 0:
        train_data = train_data_temp
        test_data = test_data_temp
        if isval:
            val_data = val_data_temp
    else:
        train_data += train_data_temp
        test_data += test_data_temp
        if isval:
            if val_data == 0:
                val_data = val_data_temp
            else:
                val_data += val_data_temp


transform1 = transforms.Compose([
    v2.ToTensor(),
    v2.Resize([150, 150])])

train_data = 0
val_data = 0
test_data = 0
createdata('dataset/jeans.csv', 'dataset/dataset_equal/jeans', False)
createdata('dataset/tshirt.csv', 'dataset/dataset_equal/tshirt', False)
createdata('dataset/sofa.csv', 'dataset/dataset_equal/sofa', False)
createdata('dataset/tv.csv', 'dataset/dataset_equal/tv', False)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=False)

train_data = 0
val_data = 0
test_data = 0
createdata('dataset/jeans.csv', 'dataset/dataset_equal/jeans', True)
createdata('dataset/tshirt.csv', 'dataset/dataset_equal/tshirt', True)
createdata('dataset/sofa.csv', 'dataset/dataset_equal/sofa', True)
createdata('dataset/tv.csv', 'dataset/dataset_equal/tv', True)
createdata('dataset/other.csv', 'dataset/dataset_equal/other', True)
train_loaderD = torch.utils.data.DataLoader(
    train_data, batch_size=20, shuffle=True)
val_loaderD = torch.utils.data.DataLoader(
    val_data, batch_size=20, shuffle=True)
test_loaderD = torch.utils.data.DataLoader(
    test_data, batch_size=20, shuffle=False)

model = ConvNet(4)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train(model, loss_fn, optimizer, 10, train_loader, test_loader)

model = create_model(model, 3, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = train(model, loss_fn, optimizer, 20, train_loaderD, test_loader)
model.load_state_dict(torch.load('model.pt'))

train_accuracy, _ = evaluate(model, train_loaderD, loss_fn)
print('Train accuracy:', train_accuracy.numpy())

test_accuracy, _ = evaluate(model, test_loaderD, loss_fn)
print('Test accuracy:', test_accuracy.numpy())
