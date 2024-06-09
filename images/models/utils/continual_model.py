# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from torchvision import transforms
from self_supervised.augmentations.helper import norm_mean_std, get_color_distortion, GaussianBlur, Solarization
from self_supervised.augmentations import RotationTransform, SimCLRTransform, SimSiamTransform, RandAugment, AutoAugment


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        if self.args.adam_lr == 0:
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            self.opt2 = None
        else:
            self.opt = Adam(self.net.parameters(), lr=self.args.adam_lr)
            self.opt2 = None
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
        self.device = get_device()
        if self.args.num_rotations>0:
            self.rotation_transform = RotationTransform(num_rotations=self.args.num_rotations)
        self.buffer_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=self.args.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToPILImage(),
                    # AutoAugment(),
                    # RandAugment(num_ops=5),
                    # get_color_distortion(s=0.5),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.ToPILImage(),
                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    # transforms.RandomApply([transforms.GaussianBlur(self.args.img_size-1)], p=0.2),
                    # transforms.RandomSolarize(threshold=200),
                    transforms.ToTensor(),
                    norm_mean_std(self.args.img_size),
                    # transforms.RandomErasing(p=0.5),
                ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def rotate(self, inputs: torch.Tensor):
        inputs_rot, labels_rot = [], []
        for input in inputs:
            x, y = self.rotation_transform(input)
            inputs_rot.append(x)
            labels_rot.append(y)
        labels_rot = torch.stack(labels_rot).to(inputs.device)
        inputs_rot = torch.stack(inputs_rot, dim=0).to(inputs.device)
        return inputs_rot, labels_rot
