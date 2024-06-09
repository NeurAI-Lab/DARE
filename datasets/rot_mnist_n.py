# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from datasets.transforms.rotation import Rotation
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from datasets.perm_mnist import store_mnist_loaders
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
import numpy as np


class RotatedMNISTN(ContinualDataset):
    NAME = 'rot-mnist-n'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    # read number of tasks from arguments now
    N_TASKS = 0

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(RotatedMNISTN, self).__init__(args)
        np.random.seed(args.mnist_seed)
        RotatedMNISTN.N_TASKS = args.num_tasks

        self.rotations = [Rotation() for _ in range(RotatedMNISTN.N_TASKS)]
        print("Rotations sampled: ", [self.rotations[i].degrees for i in range(args.num_tasks)])
        # degree_min = 0
        # degree_max = 180
        # num_tasks = 20
        # degree_inc = (degree_max - degree_min) / (args.num_tasks - 1)

        # lst_degrees = [args.deg_inc * i for i in range(args.num_tasks)]
        # lst_degrees = np.random.permutation(lst_degrees)
        # self.rotations = [GivenRotation(deg) for deg in lst_degrees]

    def get_data_loaders(self, task_id=None):
        # transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        transform = transforms.Compose((self.rotations[task_id], transforms.ToTensor()))
        train, test = store_mnist_loaders(transform, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        return DataLoader(self.train_loader.dataset,
                          batch_size=batch_size, shuffle=True)

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNISTN.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None
