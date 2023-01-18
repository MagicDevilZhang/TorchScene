import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(input_size, data_path, batch_size, num_workers) -> Dict[DataLoader, DataLoader]:
    """transform data and load data into dataloader. Images should be arranged in this way by default: ::

        root/my_dataset/train/dog/xxx.png
        root/my_dataset/train/dog/xxy.png
        root/my_dataset/train/dog/[...]/xxz.png

        root/my_dataset/val/cat/123.png
        root/my_dataset/val/cat/nsdf3.png
        root/my_dataset/val/cat/[...]/asd932_.png


    notice that the directory of your training data must be names as 'train', and
    the directory name of your validation data must be named as 'val', and they should
    under the same directory.

    Args:
        input_size (int): transformed image resolution, such as 224.
        data_path (string): eg. xx/my_dataset/
        batch_size (int): batch size
        num_workers (int): number of pytorch DataLoader worker subprocess
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in
                      ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ) for x in ['train', 'val']
    }
    return dataloaders_dict
