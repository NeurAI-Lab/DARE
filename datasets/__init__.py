# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.perm_mnist import PermutedMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_stl10 import SequentialSTL10
from datasets.seq_core50 import SequentialCore50j
from datasets.seq_miniimagenet import SequentialMiniImagenet
from datasets.seq_imagenet import SequentialImageNet
from datasets.seq_imagenet100 import SequentialImageNet100
from datasets.rot_mnist import RotatedMNIST
from datasets.rot_mnist_n import RotatedMNISTN
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.mnist_360 import MNIST360
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.cifar10_noisy import CIFAR10Noisy
from datasets.domain_cifar10 import DomainCIFAR10
from datasets.domain_cifar100 import DomainCIFAR100
from datasets.domain_net import DomainNet
from datasets.super_cifar100 import SuperCIFAR100

NAMES = {
    PermutedMNIST.NAME: PermutedMNIST,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialSTL10.NAME: SequentialSTL10,
    SequentialCore50j.NAME: SequentialCore50j,
    SequentialMiniImagenet.NAME: SequentialMiniImagenet,
    SequentialImageNet.NAME: SequentialImageNet,
    SequentialImageNet100.NAME: SequentialImageNet100,
    RotatedMNIST.NAME: RotatedMNIST,
    RotatedMNISTN.NAME: RotatedMNISTN,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MNIST360.NAME: MNIST360,
    DomainCIFAR10.NAME: DomainCIFAR10,
    DomainCIFAR100.NAME: DomainCIFAR100,
    DomainNet.NAME: DomainNet,
    SuperCIFAR100.NAME: SuperCIFAR100,
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
