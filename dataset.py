import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from utilities.utils import *

class CustomImageDataset(Dataset):
    def __init__(self, data_source, data_type, transform=None, target_transform=None):
        self.data_source = data_source
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = {}
        df = pd.read_csv(os.path.join(data_source, f'{data_type}.csv'))
        if data_type == 'train':
            self.image_ids, self.labels = df['image_ID'], df['label']
            classes = [label for label in np.unique(self.labels)]
            self.class_names = {label: idx for idx, label in enumerate(classes)}
        else:
            self.image_ids = df['image_ID']

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_source, self.data_type, self.image_ids[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label if self.data_type == 'train' else image
        

if __name__ == "__main__":
    logger.info('... [ Testing dataset ] ...')
    dataset = CustomImageDataset(annotations_file='/home/muhammet/Challenges/Datasets/sports/train.csv', img_dir='/home/muhammet/Challenges/Datasets/sports/train')
    print(len(dataset))