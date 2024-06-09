# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_data_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from datasets.seq_imagenet import store_masked_loaders_imagenet
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


class MyImageNet100(ImageNet):
    """
    Overrides the Imagenet dataset to change the getitem function.
    """
    def __init__(self, root, split='train', transform=None,
                 target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
        super(MyImageNet100, self).__init__(root, split=split,
                                         transform=transform,
                                         target_transform=target_transform)

        df = pd.read_csv(cur_dir + '/utils/ImageNet_train_100.csv')
        df.iloc[:, 0] = root + '/' + df.iloc[:, 0].astype(str)
        self.imgs = list(df.to_records(index=False))
        self.samples = self.imgs
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        not_aug_img = self.not_aug_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, torch.as_tensor(int(target)), not_aug_img


class MyTestImageNet100(ImageNet):
    """
    Overrides the Imagenet dataset to change the getitem function.
    """
    def __init__(self, root, split='val', transform=None,
                 target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
        super(MyTestImageNet100, self).__init__(root, split=split,
                                         transform=transform,
                                         target_transform=target_transform)
        df = pd.read_csv(cur_dir + '/utils/ImageNet_val_100.csv')
        df.iloc[:, 0] = root + '/' + df.iloc[:, 0].astype(str)
        self.imgs = list(df.to_records(index=False))
        self.samples = self.imgs
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, torch.as_tensor(int(target))


class SequentialImageNet100(ContinualDataset):

    NAME = 'seq-imagenet100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

    def get_data_loaders(self, task_id=None):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyImageNet100(base_data_path() + 'ImageNet_2012', split='train', transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = MyTestImageNet100(base_data_path() + 'ImageNet_2012', split='val', transform=test_transform)

        train, test = store_masked_loaders_imagenet(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyImageNet100(base_data_path() + 'ImageNet_2012', split='train', transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImageNet100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialImageNet100.N_CLASSES_PER_TASK
                        * SequentialImageNet100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))
        return transform
