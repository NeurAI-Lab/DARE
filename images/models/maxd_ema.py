# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from self_supervised.criterion import DINOLoss, NTXent, VICRegLoss, SupConLoss
import sys
import torch.nn as nn
import math
import numpy as np
from torch.optim import SGD, Adam
from torchvision import transforms
from copy import deepcopy

from scipy.stats import skewnorm


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pairwise_dist(x):
    x_square = x.pow(2).sum(dim=1)
    prod = x @ x.t()
    pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) - 2 * prod).clamp(1e-12)
    pdist[range(len(x)), range(len(x))] = 0.
    return pdist


def pairwise_prob(pdist):
    return torch.exp(-pdist)


def hcr_loss(h, g, eps):
    q1, q2 = pairwise_prob(pairwise_dist(h)), pairwise_prob(pairwise_dist(g))
    return -1 * (q1 * torch.log(q2 + eps)).mean() + -1 * ((1 - q1) * torch.log((1 - q2) + eps)).mean()


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


class MaxdEma(ContinualModel):
    NAME = 'maxd_ema'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(MaxdEma, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if self.args.pretext_task == 'dino':
            self.dino = DINOLoss(args.n_classes)
        if self.args.pretext_task == 'simclr':
            self.simclr = NTXent()
        if self.args.pretext_task == 'mae':
            self.l1_loss = torch.nn.L1Loss()
        if self.args.supcon_weight > 0:
            self.supcon = SupConLoss(temperature=self.args.supcon_temp)

        self.intermediate_sampling = self.args.intermediate_sampling
        self.calculate_drift = False
        self.drift = []

        # ema model params
        self.global_step = 0
        self.ema_update_freq = args.ema_update_freq
        self.ema_alpha = args.linear_alpha
        self.ema_model = deepcopy(self.net).to(self.device)

        # create normal distribution for intermediate logit storing
        if self.intermediate_sampling:
            if self.args.skewness > 0:
                self.sampling_probs = skewnorm.pdf(np.linspace(0, self.args.n_epochs),
                                                   np.ones(self.args.n_epochs)*-(self.args.skewness),
                                                   loc=self.args.n_epochs*0.7, scale=self.args.skewness)
            else:
                self.sampling_probs = (1.0 / (np.sqrt(2*np.pi)*self.args.std)) * np.exp(
                -0.5 * ((np.arange(self.args.n_epochs) - (self.args.n_epochs // 2)) / self.args.std) ** 2)
            self.sampling_probs /= np.max(self.sampling_probs)

    def compute_pretext_task_loss(self, buf_outputs, buf_logits, weight=0.):

        if self.args.pretext_task == 'l1':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=1).mean()

        elif self.args.pretext_task == 'l2':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=2).mean()

        elif self.args.pretext_task == 'linf':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=float('inf')).mean()

        elif self.args.pretext_task == 'kl':
            sim_logits = F.softmax(buf_logits)
            loss = self.args.alpha * F.kl_div(F.log_softmax(buf_logits), sim_logits)

        else:
            loss = weight * F.mse_loss(buf_outputs, buf_logits)

        return loss

    def compute_buffer_loss(self, exclude_logit_loss=False, weight=0., task_aware=False, cur_task=0, cross_distill=False):
        loss_aux_ce, loss_aux_logit = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits1, buf_logits2, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, task_aware=task_aware, cur_task=cur_task)
            buf_outputs = self.net(buf_inputs)
            if self.args.supcon_weight > 0:
                loss_aux_ce = self.args.beta * self.loss(buf_outputs['logits1'], buf_labels) + \
                              self.args.supcon_weight * self.loss(buf_outputs['logits2'], buf_labels)
            else:
                loss_aux_ce = self.args.beta * (self.loss(buf_outputs['logits1'], buf_labels) +
                                            self.loss(buf_outputs['logits2'], buf_labels))

            if not exclude_logit_loss:
                buf_inputs_, buf_labels_, buf_logits1_, buf_logits2_, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, task_aware=task_aware, cur_task=cur_task)
                buf_outputs_ = self.net(buf_inputs_)
                ema_outputs = self.ema_model(buf_inputs_)
                if cross_distill:
                    loss_aux_logit = self.compute_pretext_task_loss(buf_outputs_['logits1'], buf_logits1_, weight) + \
                                     self.compute_pretext_task_loss(buf_outputs_['logits2'], buf_logits1_, weight)
                else:
                    loss_aux_logit = self.compute_pretext_task_loss(buf_outputs_['logits1'], buf_logits1_, weight) + \
                                     self.compute_pretext_task_loss(buf_outputs_['logits2'], buf_logits2_, weight)

        return loss_aux_ce, loss_aux_logit

    def stepb(self, inputs, labels, not_aug_inputs, task_id=None):
        # Step B: Maximize the discrepancy
        softmax = torch.nn.Softmax(dim=-1)
        self.opt.zero_grad()
        if self.opt2 is not None:
            self.opt2.zero_grad()

        if self.args.frozen_supcon:
            self.net.freeze(name='classifier_2')

        loss = 0

        outputs = self.net(inputs)

        if self.args.dt_stepb:
            # CE for current task samples
            loss_ce = self.loss(outputs['logits1'], labels) + self.loss(outputs['logits2'], labels)
            loss += loss_ce

        if self.args.maximize_task == 'l1':
            d_loss = -(torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']), p=1).mean())

        elif self.args.maximize_task == 'hcr':
            d_loss = -hcr_loss(F.normalize(outputs['logits1'], dim=1), F.normalize(outputs['logits2'], dim=1), eps=1e-12)


        loss_b = self.args.maxd_weight * d_loss
        loss += loss_b

        # Buffered samples CE
        loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(self.args.exclude_logit_loss_in_b_and_c,
                                                               weight=self.args.logitb_weight,
                                                               task_aware=self.args.task_buffer,
                                                               cur_task=task_id[0].item())
        loss += self.args.weight_l3 * loss_aux_ce
        loss += self.args.weight_l3 * loss_aux_logit

        loss.backward()
        self.opt.step()
        if self.opt2 is not None:
            self.opt2.step()

        # populate buffer if step a is not present
        if self.args.iterative_buffer and self.args.no_stepa:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs['logits1'].data,
                                 logits2=outputs['logits2'].data,
                                 task_labels=task_id)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss_b.item(), loss_aux_ce.item(), loss_aux_logit.item()

    def stepc(self, inputs, labels, t):
        # Step C: Minimize the discrepancy
        softmax = torch.nn.Softmax(dim=-1)
        loss = 0
        self.opt.zero_grad()
        if self.opt2 is not None:
            self.opt2.zero_grad()

        outputs = self.net(inputs)

        if self.args.dt_stepc:
            # CE for current task samples
            if self.args.buffer_only:
                loss_a = self.loss(outputs['logits2'], labels)
            elif self.args.supcon_weight > 0:
                loss_a = self.loss(outputs['logits1'], labels) + \
                         self.args.supcon_weight * self.supcon(outputs['logits2'], labels)
            else:
                loss_a = self.loss(outputs['logits1'], labels) + self.loss(outputs['logits2'], labels)
            loss += loss_a

        if self.args.maximize_task == 'l1':
            d_loss = torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']), p=1).mean()

        elif self.args.maximize_task == 'hcr':
            d_loss = hcr_loss(F.normalize(outputs['logits1'], dim=1), F.normalize(outputs['logits2'], dim=1), eps=1e-12)

        loss_c = self.args.mind_weight * d_loss
        loss += loss_c

        # Buffered samples CE
        loss_aux_ce, loss_aux_logit = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        if not self.args.no_stepc_buffer:
            loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(self.args.exclude_logit_loss_in_b_and_c,
                                                                   weight=self.args.logitc_weight,
                                                                   task_aware=self.args.task_buffer,
                                                                   cur_task=t)
        loss += self.args.weight_l3 * loss_aux_ce
        loss += self.args.weight_l3 * loss_aux_logit

        loss.backward()
        self.opt.step()
        if self.opt2 is not None:
            self.opt2.step()

        self.net.unfreeze()

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss_c.item(), loss_aux_ce.item(), loss_aux_logit.item()

    def observe(self, inputs, labels, not_aug_inputs, task_id=None, epoch=0):
        self.opt.zero_grad()
        if self.opt2 is not None:
            self.opt2.zero_grad()

        loss = 0
        outputs = self.net(inputs)
        # CE for current task samples
        if self.args.buffer_only:
            loss_a = self.loss(outputs['logits2'], labels)
        elif self.args.supcon_weight > 0:
            loss_a = self.loss(outputs['logits1'], labels) + \
                     self.args.supcon_weight * self.supcon(outputs['logits2'], labels)
        else:
            loss_a = self.loss(outputs['logits1'], labels) + self.loss(outputs['logits2'], labels)

        if task_id[0].item() > 0:
            loss += self.args.weight_l1 * loss_a
        else:
            loss += loss_a

        # Buffered samples
        loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(weight=self.args.alpha,
                                                               cross_distill=self.args.cross_distill)
        loss += self.args.weight_l3 * loss_aux_ce
        loss += self.args.weight_l3 * loss_aux_logit


        loss.backward()
        self.opt.step()
        if self.opt2 is not None:
            self.opt2.step()

        # populate buffer
        if self.args.iterative_buffer and not self.intermediate_sampling:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs['logits1'].data,
                                 logits2=outputs['logits2'].data,
                                 task_labels=task_id)

        if self.args.iterative_buffer and self.intermediate_sampling:
            if torch.rand(1) < self.sampling_probs[epoch]:
                self.buffer.add_data(examples=not_aug_inputs,
                                     labels=labels,
                                     logits=outputs['logits1'].data,
                                     logits2=outputs['logits2'].data,
                                     task_labels=task_id)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss_a.item(), loss_aux_ce.item(), loss_aux_logit.item()

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=(1-alpha))

    def end_task(self, dataset):
        # store samples in the buffer at the end of the task
        if not self.args.iterative_buffer:
            for _, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(inputs)
                self.buffer.add_data(examples=not_aug_inputs,
                                     labels=labels,
                                     logits=outputs['logits1'].data,
                                     logits2=outputs['logits2'].data)

            # unfreeze the whole architecture
            self.net.unfreeze()
