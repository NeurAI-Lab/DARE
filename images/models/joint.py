# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar

import os
import torch
import numpy as np
import math
from torchvision import transforms

from datasets.domain_net import ImageFilelist

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'domain-2il', 'domain-supcif']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il' and ('domain' not in dataset.SETTING):
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            if dataset.NAME == 'seq-stl10':
                all_data = np.moveaxis(all_data, 1, -1)
            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    if isinstance(outputs, dict):
                        loss = self.loss(outputs['logits1'], labels.long())
                    else:
                        loss = self.loss(outputs[0], labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())

        elif dataset.SETTING == 'domain-supcif':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    if isinstance(outputs, dict):
                        loss = self.loss(outputs['logits1'], labels.long())
                    else:
                        loss = self.loss(outputs[0], labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())

        elif dataset.SETTING == 'domain-2il':
            self.old_data.append(dataset.train_loader.dataset)
            # self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            # self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            dnet_dataset = ImageFilelist(
                root=dataset.data_path,
                flist=[os.path.join(dataset.annot_path, d + "_train.txt") for d in
                       ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']],
                transform=transforms.Compose(dataset.TRANSFORM),
                not_aug_transform=transforms.Compose(dataset.NOT_AUG_TRANSFORM),
            )

            # prepare dataloader
            loader = torch.utils.data.DataLoader(dnet_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels, _ = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    if isinstance(outputs, dict):
                        loss = self.loss(outputs['logits1'], labels.long())
                    else:
                        loss = self.loss(outputs[0], labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())

        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            # loader_caches = [[] for _ in range(len(self.old_data))]
            # sources = torch.randint(5, (128,))
            all_inputs = []
            all_labels = []
            all_tasks = []
            for si, source in enumerate(self.old_data):
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
                    all_tasks.append(torch.ones(l.shape) * si)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            all_tasks = torch.cat(all_tasks)
            bs = self.args.batch_size
            bst = bs // len(self.old_data)
            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    # torch.cat([all_inputs[order][torch.where(all_tasks == 0)][0:12],
                    #            all_inputs[order][torch.where(all_tasks == 1)][0:12]])
                    inputs = torch.cat(
                        [all_inputs[order][torch.where(all_tasks == task)][i*bst:(i+1)*bst] for task in
                         range(len(self.old_data))]
                    )
                    labels = torch.cat(
                        [all_labels[order][torch.where(all_tasks == task)][i*bst:(i+1)*bst] for task in
                         range(len(self.old_data))]
                    )
                    # inputs = all_inputs[order][i * bs: (i+1) * bs]
                    # labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    if isinstance(outputs, dict):
                        loss = self.loss(outputs['logits1'], labels.long())
                    else:
                        loss = self.loss(outputs[0], labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

    def observe(self, inputs, labels, not_aug_inputs, task_id=None):
        return 0, 0, 0
