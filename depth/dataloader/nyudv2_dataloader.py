import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from depth.dataloader.nyu_transform import *
import os


class NYUDV2Dataset(Dataset):
    """NYUV2D dataset."""

    def __init__(self, csv_file, root_path, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform 
        self.root_path = root_path

    def __getitem__(self, idx):
        image_name = self.frame.loc[idx, 0]
        depth_name = self.frame.loc[idx, 1]
        root_path = self.root_path

        # image = Image.open(root_path+image_name)
        image = Image.open(os.path.join(root_path, image_name))
        # depth = Image.open(root_path+depth_name)
        depth = Image.open(os.path.join(root_path, depth_name))

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)

class SingleDataset(Dataset):
    def __init__(self, image_name, root_path, transform=None):
        self.image_name = image_name
        self.root_path = root_path
        self.transform = transform

    def __getitem__(self, idx):
        root_path = self.root_path
        image_name = self.image_name

        # image = Image.open(root_path+image_name)
        image = Image.open(os.path.join(root_path, image_name))

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return 1


def getTrainingData_NYUDV2(batch_size, trainlist_path, root_path):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = NYUDV2Dataset(csv_file=trainlist_path,
                                        root_path = root_path,
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training


def getTestingData_NYUDV2(batch_size, testlist_path, root_path):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = NYUDV2Dataset(csv_file=testlist_path,
                                        root_path=root_path,
                                        transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_testing

def getTestingData_IKEA(batch_size, testlist_path, root_path):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = NYUDV2Dataset(csv_file=testlist_path,
                                        root_path=root_path,
                                        transform=transforms.Compose([
                                           # Scale(240),  #将图片短边缩放至x，长宽比保持不变
                                           # CenterCrop([304, 228], [304, 228]),
                                            Scale(228),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_testing

def getSingleData_IKEA(batch_size, image_name, root_path):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = SingleDataset(image_name=image_name,
                                        root_path=root_path,
                                        transform=transforms.Compose([
                                            # Scale(240),  #将图片短边缩放至x，长宽比保持不变
                                            # CenterCrop([304, 228], [304, 228]),
                                            ScaleImage(228),
                                            ToTensorImage(is_test=True),
                                            NormalizeImage(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_testing
