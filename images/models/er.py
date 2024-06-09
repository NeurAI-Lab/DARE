# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import math
from torch.nn import functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.calculate_drift = False
        self.drift = []

    def observe(self, inputs, labels, not_aug_inputs, task_ids=None, epoch=0):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs = buf_inputs.to(self.device)
            buf_labels = buf_labels.to(self.device)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            # buf_outputs = self.net(buf_inputs)
            # loss += self.args.er_weight * self.loss(buf_outputs['logits1'], buf_labels)

        outputs = self.net(inputs)
        loss += self.loss(outputs['logits1'], labels)

        # calculating feature drift
        initial_rep = []
        latter_rep = []
        if self.calculate_drift:
            if not self.buffer.is_empty():
                for fi in range(int(math.ceil(self.args.buffer_size / 25))):
                    buf_inputs, buf_labels = self.buffer.get_data(
                        25, transform=self.transform, finetuning=True, index_start=fi)
                    buf_outputs = self.net(buf_inputs, return_rep=True)
                    initial_rep.append(buf_outputs['features'])

        loss.backward()
        self.opt.step()

        if self.calculate_drift:
            if not self.buffer.is_empty():
                for fi in range(int(math.ceil(self.args.buffer_size / 25))):
                    buf_inputs, buf_labels = self.buffer.get_data(
                        25, transform=self.transform, finetuning=True, index_start=fi)
                    buf_outputs = self.net(buf_inputs, return_rep=True)
                    latter_rep.append(buf_outputs['features'])

                # calculate drift in terms of mse
                mse_dist = F.mse_loss(torch.cat(initial_rep), torch.cat(latter_rep))
                self.drift.append(torch.round(mse_dist, decimals=4))
                initial_rep, latter_rep = None, None

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        return loss.item(), torch.tensor(0), torch.tensor(0)




