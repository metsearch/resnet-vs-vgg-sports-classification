import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, vgg16

import matplotlib.pyplot as plt

from rich.progress import track

from utilities.utils import *
    
def learn_with_resnet18(train_loader: DataLoader, num_classes, device, num_epochs, path2models, path2metrics):
    logger.info('Learning with resnet18...')
    nb_data = len(train_loader)
    
    pretrained_resnet18 = resnet18(weights='DEFAULT')
    
    for param in pretrained_resnet18.parameters():
        param.requires_grad = False
    
    pretrained_resnet18.fc = nn.Linear(pretrained_resnet18.fc.in_features, num_classes)
    pretrained_resnet18 = pretrained_resnet18.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_resnet18.fc.parameters(), lr=0.001)
    
    losses = []
    nb_data = len(train_loader)
    for epoch in range(num_epochs):
        counter = 0
        epoch_loss = 0.0
        pretrained_resnet18.train()
        for inputs, labels in track(train_loader, description='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = pretrained_resnet18(inputs)
            E: torch.Tensor = criterion(outputs, labels)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(inputs)
        
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{num_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')

    torch.save(pretrained_resnet18.cpu(), os.path.join(path2models, 'resnet18.pth'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, num_epochs + 1), losses, label='ResNet18 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet18 Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'resnet_training_loss.png'))
    plt.show()

def learn_with_vgg16(train_loader: DataLoader, num_classes, device, num_epochs, path2models, path2metrics):
    logger.info('Learning with VGG16...')
    nb_data = len(train_loader)
    
    pretrained_vgg16 = vgg16(weights='DEFAULT')
    
    for param in pretrained_vgg16.parameters():
        param.requires_grad = False
    
    pretrained_vgg16.classifier[6] = nn.Linear(pretrained_vgg16.classifier[6].in_features, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_vgg16.parameters(), lr=0.001)
    
    losses = []
    nb_data = len(train_loader)
    for epoch in range(num_epochs):
        counter = 0
        epoch_loss = 0.0
        pretrained_vgg16.train()
        for inputs, labels in track(train_loader, description='Training'):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = pretrained_vgg16(inputs)
            E: torch.Tensor = criterion(outputs, labels)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(inputs)
        
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{num_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')
        
    torch.save(pretrained_vgg16.cpu(), os.path.join(path2models, 'vgg16.pth'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, num_epochs + 1), losses, label='Vgg16 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Vgg16 Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'vgg_training_loss.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('... [ Learning ] ...')