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
from torch.optim import SGD, Adam
from torchvision import transforms
from copy import deepcopy


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


class MaxER(ContinualModel):
    NAME = 'maxer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(MaxER, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.ema_model = deepcopy(self.net).to(self.device)
        if self.args.supcon_weight > 0:
            self.supcon = SupConLoss(temperature=self.args.supcon_temp)
        self.global_step = 0
        self.ema_update_freq = args.ema_update_freq
        self.ema_alpha = args.linear_alpha

    def compute_pretext_task_loss(self, buf_outputs, buf_logits, weight=0.):

        # Alignment and uniform loss
        if self.args.pretext_task == 'align_uni':
            loss = self.args.align_weight * (buf_outputs - buf_logits.detach()).norm(p=2, dim=1).pow(2).mean() / 10 \
                    + self.args.uni_weight * (
                        torch.pdist(F.normalize(buf_outputs), p=2).pow(2).mul(-2).exp().mean().log())

        # Barlow twins
        elif self.args.pretext_task == 'barlow_twins':
            # buf_outputs_norm = (buf_outputs[0] - buf_outputs[0].mean(0)) / buf_outputs[0].std(0)
            # buf_logits_norm = (buf_logits - buf_logits.mean(0)) / buf_logits.std(0)
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            c = torch.mm(buf_outputs_norm.T, buf_logits_norm)
            c.div_(self.args.minibatch_size)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = self.args.barlow_on_weight * on_diag + self.args.barlow_off_weight * off_diag

        # SimSiam loss
        elif self.args.pretext_task == 'simsiam':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            loss = -(buf_logits_norm * buf_outputs_norm).sum(dim=1).mean()

        # BYOL
        elif self.args.pretext_task == 'byol':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            loss = self.args.byol_weight * (2 - 2 * (buf_outputs_norm * buf_logits_norm).sum(dim=-1).mean())

        # vicreg
        elif self.args.pretext_task == 'vicreg':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            loss = VICRegLoss(buf_outputs_norm, buf_logits_norm)

        # Dino
        elif self.args.pretext_task == 'dino':
            loss = self.args.dino_weight * self.dino(F.normalize(buf_outputs), F.normalize(buf_logits))

        elif self.args.pretext_task == 'simclr':
            loss = self.args.simclr_weight * self.simclr(F.normalize(buf_outputs), F.normalize(buf_logits))

        elif self.args.pretext_task == 'mi':
            EPS = sys.float_info.epsilon
            z = F.softmax(buf_outputs, dim=-1)
            zt = F.softmax(buf_logits, dim=-1)
            _, C = z.size()
            P_temp = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
            P = ((P_temp + P_temp.t()) / 2) / P_temp.sum()
            P[(P < EPS).data] = EPS
            Pi = P.sum(dim=1).view(C, 1).expand(C, C).clone()
            Pj = P.sum(dim=0).view(1, C).expand(C, C).clone()
            Pi[(Pi < EPS).data] = EPS
            Pj[(Pj < EPS).data] = EPS
            loss = self.args.mi_weight * (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()

        elif self.args.pretext_task == 'l1':
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
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, task_aware=task_aware, cur_task=cur_task)

            buf_outputs = self.net(buf_inputs)
            if self.args.supcon_weight > 0:
                loss_aux_ce = self.args.beta * self.loss(buf_outputs['logits1'], buf_labels) + \
                              self.args.supcon_weight * self.supcon(buf_outputs['logits2'], buf_labels)
            else:
                loss_aux_ce = self.args.beta * (self.loss(buf_outputs['logits1'], buf_labels) +
                                            self.loss(buf_outputs['logits2'], buf_labels))

            if not exclude_logit_loss:
                buf_inputs_, buf_labels_ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, task_aware=task_aware, cur_task=cur_task)
                buf_outputs_ = self.net(buf_inputs_)
                ema_outputs = self.ema_model(buf_inputs_)
                loss_aux_logit = self.compute_pretext_task_loss(buf_outputs_['logits1'], ema_outputs['logits1'].detach(), weight) + \
                                 self.compute_pretext_task_loss(buf_outputs_['logits2'], ema_outputs['logits2'].detach(), weight)

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

        if self.args.maximize_task == 'l1':
            d_loss = -(torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']), p=1).mean())

        elif self.args.maximize_task == 'l1abs':
            d_loss = -torch.mean(torch.abs(F.softmax(outputs['logits1'], dim=-1) - F.softmax(outputs['logits2'], dim=-1)))

        elif self.args.maximize_task == 'l2':
            d_loss = -(torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']), p=2).mean())

        elif self.args.maximize_task == 'linf':
            d_loss = -(torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']),
                                               p=float('inf')).mean())

        elif self.args.maximize_task == 'kl':
            d_loss = -F.kl_div(softmax(outputs['logits1']), softmax(outputs['logits2']), reduction='batchmean')

        elif self.args.maximize_task == 'mse':
            d_loss = -F.mse_loss(outputs['logits1'], outputs['logits2'])

        elif self.args.maximize_task == 'cosine':
            cos_criterion = nn.CosineSimilarity(dim=1)
            d_loss = torch.mean(cos_criterion(outputs['logits1'], outputs['logits2']))

        elif self.args.maximize_task == 'hcr':
            d_loss = -hcr_loss(F.normalize(outputs['logits1'], dim=1), F.normalize(outputs['logits2'], dim=1), eps=1e-12)


        loss_b = self.args.maxd_weight * d_loss
        loss += loss_b

        # Buffered samples
        loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(self.args.exclude_logit_loss_in_b_and_c,
                                                               weight=self.args.alpha,
                                                               task_aware=self.args.task_buffer,
                                                               cur_task=task_id[0].item())
        loss += loss_aux_ce
        loss += loss_aux_logit

        loss.backward()
        self.opt.step()
        if self.opt2 is not None:
            self.opt2.step()

        # populate buffer here if step a is not present
        if self.args.iterative_buffer and self.args.no_stepa:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels)

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

        elif self.args.maximize_task == 'l1abs':
            d_loss = torch.mean(torch.abs(F.softmax(outputs['logits1'], dim=-1) - F.softmax(outputs['logits2'], dim=-1)))

        elif self.args.maximize_task == 'l2':
            d_loss = torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']), p=2).mean()

        elif self.args.maximize_task == 'linf':
            d_loss = torch.pairwise_distance(softmax(outputs['logits1']), softmax(outputs['logits2']),
                                               p=float('inf')).mean()

        elif self.args.maximize_task == 'kl':
            d_loss = F.kl_div(softmax(outputs['logits1']), softmax(outputs['logits2']), reduction='batchmean')

        elif self.args.maximize_task == 'mse':
            d_loss = F.mse_loss(outputs['logits1'], outputs['logits2'])

        elif self.args.maximize_task == 'cosine':
            cos_criterion = nn.CosineSimilarity(dim=1)
            d_loss = -torch.mean(cos_criterion(outputs['logits1'], outputs['logits2']))

        elif self.args.maximize_task == 'hcr':
            d_loss = hcr_loss(F.normalize(outputs['logits1'], dim=1), F.normalize(outputs['logits2'], dim=1), eps=1e-12)

        loss_c = self.args.mind_weight * d_loss
        loss += loss_c

        # Buffered samples
        loss_aux_ce, loss_aux_logit = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
        if not self.args.no_stepc_buffer:
            loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(self.args.exclude_logit_loss_in_b_and_c,
                                                                   weight=self.args.alpha,
                                                                   task_aware=self.args.task_buffer,
                                                                   cur_task=t)
        loss += loss_aux_ce
        loss += loss_aux_logit

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
        loss += loss_a

        # Buffered samples
        loss_aux_ce, loss_aux_logit = self.compute_buffer_loss(weight=self.args.alpha,
                                                               cross_distill=self.args.cross_distill)
        loss += loss_aux_ce
        loss += loss_aux_logit

        loss.backward()
        self.opt.step()
        if self.opt2 is not None:
            self.opt2.step()

        # populate buffer
        if self.args.iterative_buffer:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels)

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
        # train_loader, test_loader = dataset.get_data_loaders(task_id=task_id)
        if not self.args.iterative_buffer:
            for _, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(inputs)
                self.buffer.add_data(examples=not_aug_inputs,
                                     labels=labels)

        if self.args.finetune_classifiers:
            if not self.buffer.is_empty():
                f_opt = SGD(self.net.parameters(), lr=self.args.finetune_lr)
                if self.args.frozen_finetune:
                    self.net.freeze('backbone')
                self.buffer.permute_indices()
                for _ in range(self.args.finetuning_epochs):
                    for fi in range(int(math.ceil(self.args.buffer_size / self.args.minibatch_size))):
                        buf_inputs, buf_labels = self.buffer.get_data(
                            self.args.minibatch_size, transform=self.transform, finetuning=True, index_start=fi)
                        f_opt.zero_grad()
                        buf_outputs = self.net(buf_inputs)
                        f_loss = self.args.beta * (self.loss(buf_outputs['logits1'], buf_labels) +
                                                        self.loss(buf_outputs['logits2'], buf_labels))
                        f_loss.backward()
                        f_opt.step()

            # unfreeze the whole architecture
            self.net.unfreeze()
