import os
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from .log import logger

def tensor2img(t, t_type='rgb'):
    if t_type == 'gray':
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.ToPILImage(),
        ])
    elif t_type == 'rgb':
        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage(),
        ])
    else:
        raise ValueError('Unsupported image type. Use "gray" or "rgb".')

    return np.array(transform(t).convert('RGB'))

def display_images_with_predictions(path2metrics, arch, images, predictions, num_images=20, num_images_per_row=4):    
    plt.figure(figsize = (20, 10))
    images_to_display_idxes = [random.randint(0, len(images) - 1) for _ in range(num_images)]
    for i, idx in enumerate(images_to_display_idxes):
        img = tensor2img(images[idx].squeeze())
        plt.subplot(num_images_per_row, num_images // num_images_per_row, i + 1)
        plt.imshow(img, cmap = "gray"); plt.axis("off")
        plt.title(f'{predictions[idx]}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path2metrics, f'{arch}_inference_examples.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('Testing utils...')