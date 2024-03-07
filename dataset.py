import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utilities.utils import *

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, show_labels=False):
        self.img_labels = pd.read_csv(annotations_file)
        if show_labels:
            print(self.img_labels['label'].value_counts())
            class_names = list(self.img_labels['label'].value_counts().keys())
            self.class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
            self.index_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        label = self.class_to_index[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    logger.info('... [ Testing dataset ] ...')
    dataset = CustomImageDataset(annotations_file='/home/muhammet/Challenges/Datasets/sports/train.csv', img_dir='/home/muhammet/Challenges/Datasets/sports/train')
    print(len(dataset))