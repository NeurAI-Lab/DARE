# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from backbone.ResNet18_mod import resnet18_mod
import torch.nn.functional as F
from utils.conf import base_data_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_domc100_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from argparse import Namespace
import numpy as np
import math
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class DomainCIFAR100(ContinualDataset):

    NAME = 'domain-cifar100'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 100
    # number of tasks is read from the list of corruptions
    N_TASKS = 0

    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             # get_color_distortion(s=1),
             # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(DomainCIFAR100, self).__init__(args)

        DomainCIFAR100.N_TASKS = math.ceil(len(args.corruptions)/2) if args.combine_category else len(args.corruptions)

        self.corruptions = args.corruptions
        print("Corruptions sampled: ", ", ".join(self.corruptions))

    def get_data_loaders(self, task_id=None):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train, test = store_domc100_loaders(train_transform=transform, test_transform=test_transform,
                                               setting=self, task_id=task_id,
                                               combine_category=self.args.combine_category)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_data_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), DomainCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(num_classifier=2, norm_feature=False, diff_classifier=False):
        if num_classifier == 3:
            return resnet18_mod(DomainCIFAR100.N_CLASSES_PER_TASK)
        else:
            return resnet18(DomainCIFAR100.N_CLASSES_PER_TASK)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform
