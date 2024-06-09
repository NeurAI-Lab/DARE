# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import math
from utils.conf import base_data_path

from utils.custom_dataset import CustomDataset


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self, task_id=None) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_domc100_loaders(train_transform, test_transform, setting, task_id=0, combine_category=False):
    '''
    Args:
        combine_category: Set it to true if noises from the same category should be combined into one task
    '''

    # FIXME: Make this generic for different severity levels
    cur_corruption = setting.corruptions[task_id]
    X = np.load('{}CIFAR-100-C/Train/sev5/{}.npy'.format(base_data_path(), cur_corruption))
    y = np.load('{}CIFAR-100-C/Train/labels.npy'.format(base_data_path()))

    train_dataset = CustomDataset(X, y, train_transform, train=True)

    test_X = np.load('{}CIFAR-100-C/Test/sev5/%s.npy'.format(base_data_path(), cur_corruption))
    test_y = np.load('{}input/CIFAR-100-C/Test/sev5/labels.npy'.format(base_data_path()))

    test_dataset = CustomDataset(test_X, test_y, test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=setting.args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=setting.args.num_workers,
                             pin_memory=True)

    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def store_supc100_loaders(train_transform, test_transform, setting):
    '''
    Args:
        combine_category: Set it to true if noises from the same category should be combined into one task
    '''

    # FIXME: Make this generic for different severity levels
    X = np.load('{}CIFAR-100-D/Train/Domain{}.npy'.format(base_data_path(), setting.i+1))
    y = np.load('{}CIFAR-100-D/Train/labels{}.npy'.format(base_data_path(), setting.i+1))

    train_dataset = CustomDataset(X, y, train_transform, train=True)

    test_X = np.load('{}CIFAR-100-D/Test/Domain{}.npy'.format(base_data_path(), setting.i+1))
    test_y = np.load('{}CIFAR-100-D/Test/labels{}.npy'.format(base_data_path(), setting.i+1))

    test_dataset = CustomDataset(test_X, test_y, test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=setting.args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=setting.args.num_workers,
                             pin_memory=True)

    setting.i += 1
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def store_domc10_loaders(train_transform, test_transform, setting, task_id=0, combine_category=False):
    '''
    Args:
        combine_category: Set it to true if noises from the same category should be combined into one task
    '''

    # set indices in the setting
    y = np.load(r'/input/CIFAR-10-C-Train/labels.npy')
    # sampling non-overlapping indices
    if task_id == 0:
        for i in range(math.ceil(len(setting.args.corruptions) / 2)):
            index_mask_half = np.hstack([np.random.choice(np.where(y == l)[0], 2500,
                                                          replace=False) for l in np.unique(y)])
            index_mask_full = np.hstack([np.random.choice(np.where(y == l)[0], 5000,
                                                          replace=False) for l in np.unique(y)])
            setting.TASK_INDICES.append(index_mask_half)
            setting.TASK_INDICES.append(np.array([i for i in index_mask_full if i not in index_mask_half]))

    # FIXME: Make this generic for different severity levels
    # load train Set
    if combine_category:
        cur_corruption = setting.corruptions[task_id*2]
        X1 = np.load(r'/input/CIFAR-10-C-Train/sev5/%s.npy' % cur_corruption)

        X1 = X1[setting.TASK_INDICES[task_id*2]]
        y1 = y[setting.TASK_INDICES[task_id*2]]

        cur_corruption = setting.corruptions[task_id*2+1]
        X2 = np.load(r'/input/CIFAR-10-C-Train/sev5/%s.npy' % cur_corruption)

        X2 = X2[setting.TASK_INDICES[task_id*2+1]]
        y2 = y[setting.TASK_INDICES[task_id*2+1]]

        train_X = np.vstack((X1, X2))
        train_y = np.hstack((y1, y2))
    else:
        cur_corruption = setting.corruptions[task_id]
        X = np.load(r'/input/CIFAR-10-C-Train/sev5/%s.npy' % cur_corruption)

        train_X = X[setting.TASK_INDICES[task_id]]
        train_y = y[setting.TASK_INDICES[task_id]]

    train_dataset = CustomDataset(train_X, train_y, train_transform, train=True)

    # load test set
    if combine_category:
        cur_corruption = setting.corruptions[task_id*2]
        test_X1 = np.load(r'/input/CIFAR-10-C/%s.npy' % cur_corruption)[-10000:]
        test_y1 = np.load(r'/input/CIFAR-10-C/labels.npy')[-10000:]

        cur_corruption = setting.corruptions[task_id*2+1]
        test_X2 = np.load(r'/input/CIFAR-10-C/%s.npy' % cur_corruption)[-10000:]
        test_y2 = np.load(r'/input/CIFAR-10-C/labels.npy')[-10000:]

        test_X = np.vstack((test_X1, test_X2))
        test_y = np.hstack((test_y1, test_y2))
    else:
        if cur_corruption == 'clean':
            test_X = np.load(r'/input/CIFAR-10-C/%s.npy' % cur_corruption)
            test_y = np.load(r'/input/CIFAR-10-C/clean_labels.npy')
        else:
            test_X = np.load(r'/input/CIFAR-10-C/%s.npy' % cur_corruption)[40000:]
            test_y = np.load(r'/input/CIFAR-10-C/labels.npy')[40000:]

    test_dataset = CustomDataset(test_X, test_y, test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=setting.args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=setting.args.num_workers,
                             pin_memory=True)

    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=setting.args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=setting.args.num_workers)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def store_domain_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True,
                              num_workers=setting.args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False,
                             num_workers=setting.args.num_workers, pin_memory=True)

    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader

def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
