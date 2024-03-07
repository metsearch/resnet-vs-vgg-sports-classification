import os
import pickle
import click
import numpy as np

from torchvision import transforms

from dataset import CustomImageDataset
from learn import *
from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='Enable debug mode', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug
    invoked_subcommand = ctx.invoked_subcommand
    if invoked_subcommand is None:
        logger.info('No subcommand was specified')
    else:
        logger.info(f'Invoked subcommand: {invoked_subcommand}')

@router_cmd.command()
@click.option('--path2source', help='Path to source data', required=True)
@click.option('--path2destination', help='Path to data', default='data/')
def grabber(path2source, path2destination):
    if not os.path.exists(path2destination):
        os.makedirs(path2destination)
    
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values, std=std_values),
    ])
    
    def create_and_save_dataset(annotations_file, img_dir, destination_file, show_labels):
        dataset = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform, show_labels=show_labels)
        with open(os.path.join(path2destination, destination_file), 'wb') as f:
            pickle.dump(dataset, f)
    
    create_and_save_dataset(os.path.join(path2source, 'train.csv'), os.path.join(path2source, 'train'), 'train_dataset.pkl', True)
    create_and_save_dataset(os.path.join(path2source, 'test.csv'), os.path.join(path2source, 'test'), 'test_dataset.pkl', False)
    logger.info('Train and Test datasets saved!')
    
@router_cmd.command()
@click.option('--path2data', help='Path to data', default='data/')
@click.option('--path2models', help='Path to models', default='models/')
@click.option('--path2metrics', help='Path to metrics', default='metrics/')
@click.option('--model_name', help='Model name', type=click.Choice(['resnet', 'vgg']), default='resnet')
@click.option('--bt_size', help='Batch size', default=32)
@click.option('--num_epochs', help='Number of epochs', default=10)
def learn(path2data, path2models, model_name, bt_size, num_epochs, path2metrics):
    if not os.path.exists(path2models):
        os.makedirs(path2models)

    with open(os.path.join(path2data, 'train_dataset.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    
    train_loader = DataLoader(train_dataset, batch_size=bt_size, shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
    
    num_classes = np.unique(train_dataset.img_labels.iloc[:, 1]).shape[0]
    logger.info(f'Number of classes: {num_classes}') 
    
    if model_name == 'resnet':
        learn_with_resnet18(train_loader, num_classes, device, num_epochs, path2models, path2metrics)
    elif model_name == 'vgg':
        learn_with_vgg16(train_loader, num_classes, device, num_epochs, path2models, path2metrics)
    
@router_cmd.command()
def predict():
    logger.info('Testing...')

if __name__ == '__main__':
    logger.info('...')
    router_cmd(obj={})