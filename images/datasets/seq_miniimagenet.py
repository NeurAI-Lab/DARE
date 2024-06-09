# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_data_path
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

H_resize, H_crop = 84, 84


class MiniImagenet(torchvision.datasets.ImageFolder):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize((H_resize,H_resize)), transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        super(MiniImagenet, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.data = np.array([x[0] for x in self.samples])

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]
        img = self.loader(path)

        original_img = img.copy()

        # for our method to be compatible with the framework,
        # we need to resize each (original) image for to a fixed shp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target

    def __len__(self):
        return self.data.shape[0]


class MyMiniImagenet(MiniImagenet):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyMiniImagenet, self).__init__(
            root, train, transform, target_transform, download)

        self.backup_crop = transforms.CenterCrop(H_crop)

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]
        img = self.loader(path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                img = self.backup_crop(img)
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img


class SequentialMiniImagenet(ContinualDataset):

    NAME = 'seq-miniimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.228, 0.224, 0.225]
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop((H_crop, H_crop), padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self, task_id=None):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((H_resize, H_resize)),
             transforms.CenterCrop(H_crop),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        train_dataset = MyMiniImagenet(base_data_path() + 'MiniImageNet/train',
                                 train=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = MiniImagenet(base_data_path() + 'MiniImageNet/test',
                        train=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((H_resize, H_resize)),
            transforms.CenterCrop(H_crop),
            transforms.ToTensor(), self.get_denormalization_transform()])

        train_dataset = MyMiniImagenet(base_data_path() + 'MiniImageNet/train',
                            train=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_backbone():
        return resnet18(SequentialMiniImagenet.N_CLASSES_PER_TASK
                        * SequentialMiniImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialMiniImagenet.MEAN,
                                         SequentialMiniImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialMiniImagenet.MEAN,
                                SequentialMiniImagenet.STD)
        return transform