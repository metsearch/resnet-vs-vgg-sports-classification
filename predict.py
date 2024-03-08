import torch
from torch.autograd import Variable

from rich.progress import track

from utilities.utils import *

def resnet_inference(test_loader, model, device, class_names):
    predictions = []
    images = []
    with torch.no_grad():
        for X in track(test_loader, description='Inference...'):
            X = Variable(X)
            X = X.to(device)
            P = model(X)
            predictions.extend(torch.argmax(P, dim=1).cpu().numpy())
            images.extend(X.cpu().numpy())

    model.train()
    predictions = [class_names[pred] for pred in predictions]
    return images, predictions

def vgg_inference(test_loader, model, device, class_names):
    predictions = []
    images = []
    
    with torch.no_grad():
        for X in track(test_loader, description='Inference...'):
            X = Variable(X)
            X = X.to(device)
            P = model(X)
            print(torch.argmax(P, dim=1).cpu().numpy())
            predictions.extend(class_names[torch.argmax(P, dim=1).cpu().numpy()])
            images.extend(X.cpu().numpy())

    model.train()
    predictions = [class_names[pred] for pred in predictions]
    return images, predictions

if __name__ == '__main__':
    logger.info('Inference test...')