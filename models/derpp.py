# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from self_supervised.criterion import DINOLoss, NTXent, VICRegLoss
import sys
from torchvision import transforms
import math

import numpy as np
from scipy.stats import skewnorm

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if self.args.pretext_task == 'dino':
            self.dino = DINOLoss(args.n_classes)
        if self.args.pretext_task == 'simclr':
            self.simclr = NTXent()
        if self.args.pretext_task == 'mae':
            self.l1_loss = torch.nn.L1Loss()
        self.calculate_drift = False
        self.drift = []

        # create normal distribution for intermediate logit storing
        self.intermediate_sampling = self.args.intermediate_sampling
        if self.intermediate_sampling:
            if self.args.skewness > 0:
                self.sampling_probs = skewnorm.pdf(np.linspace(0, self.args.n_epochs),
                                                   np.ones(self.args.n_epochs) * -(self.args.skewness),
                                                   loc=self.args.n_epochs * 0.7, scale=self.args.skewness)
            else:
                self.sampling_probs = (1.0 / (np.sqrt(2 * np.pi) * self.args.std)) * np.exp(
                    -0.5 * ((np.arange(self.args.n_epochs) - (self.args.n_epochs // 2)) / self.args.std) ** 2)
            self.sampling_probs /= np.max(self.sampling_probs)

    def observe(self, inputs, labels, not_aug_inputs, task_ids=None, epoch=0):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        # CE for current task samples
        loss = self.loss(outputs['logits1'], labels)
        loss_1, loss_2 = torch.tensor(0), torch.tensor(0)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            # Pretext task
            loss_1 = self.args.alpha * F.mse_loss(buf_outputs['logits1'], buf_logits)
            loss += loss_1

            # CE for buffered images
            buf_inputs_2, buf_labels_2, _, _ = self.buffer.get_data(
                self.args.batch_size, transform=self.transform)
            buf_outputs_2 = self.net(buf_inputs_2)
            loss_2 = self.args.beta * self.loss(buf_outputs_2['logits1'], buf_labels_2)
            loss += loss_2

        # calculating feature drift
        initial_rep = []
        latter_rep = []
        if self.calculate_drift:
            if not self.buffer.is_empty():
                for fi in range(int(math.ceil(self.args.buffer_size / 25))):
                    buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                        25, transform=self.transform, finetuning=True, index_start=fi)
                    buf_outputs = self.net(buf_inputs, return_rep=True)
                    initial_rep.append(buf_outputs['features'])

        loss.backward()
        self.opt.step()

        if self.calculate_drift:
            if not self.buffer.is_empty():
                for fi in range(int(math.ceil(self.args.buffer_size / 25))):
                    buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                        25, transform=self.transform, finetuning=True, index_start=fi)
                    buf_outputs = self.net(buf_inputs, return_rep=True)
                    latter_rep.append(buf_outputs['features'])

                # calculate drift in terms of mse
                mse_dist = F.mse_loss(torch.cat(initial_rep), torch.cat(latter_rep))
                self.drift.append(torch.round(mse_dist, decimals=4))
                initial_rep, latter_rep = None, None

        # populate buffer
        if self.intermediate_sampling:
            if torch.rand(1) < self.sampling_probs[epoch]:
                self.buffer.add_data(examples=not_aug_inputs,
                                     labels=labels,
                                     logits=outputs['logits1'].data,
                                     logits2=outputs['logits2'].data)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs['logits1'].data,
                                 logits2=outputs['logits2'].data)

        return loss.item(), loss_2.item(), loss_1.item()
