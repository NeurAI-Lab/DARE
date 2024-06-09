# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.conf import base_path
import os
from argparse import Namespace
from typing import Dict, Any
import numpy as np


class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str,
                 stash: Dict[Any, str]=None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')
        self.loggers = {}
        self.name = stash['model_name']
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(args.output_folder, 'tensorboard_runs', a_setting, self.name),
                purge_step=stash['task_idx'] * args.n_epochs + stash['epoch_idx']+1)
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_')])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)

    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def get_log_dir(self):
        if 'class-il' in self.loggers.keys():
            return self.loggers['class-il'].log_dir
        elif 'domain-il' in self.loggers.keys():
            return self.loggers['domain-il'].log_dir
        elif 'domain-2il' in self.loggers.keys():
            return self.loggers['domain-2il'].log_dir
        elif 'domain-supcif' in self.loggers.keys():
            return self.loggers['domain-supcif'].log_dir
        else:
            return self.loggers['general-continual'].log_dir

    def log_all_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int, identifier: str = '') -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """

        for setting, a_logger in self.loggers.items():
            for i, scalar_name in zip([0, 2, 3, 4, 5, 6], ["acc_mean", "acc_avg", "acc_voting", "acc_confident",
                                                        "acc_second", "acc_third"]):
                mean_acc = all_mean_accs[i]
                accs = [all_accs[i][kk] for kk in range(len(all_accs[0]))]
                for kk, acc in enumerate(accs):
                    a_logger.add_scalar('%sperformance/acc_task%02d' % (identifier, kk + 1), acc,
                                        task_number * args.n_epochs)
                a_logger.add_scalar(identifier+scalar_name, mean_acc, task_number * args.n_epochs)

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int, identifier: str = '') -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        :param identifier: postfix to add after the task_performance and mean acc
        """
        mean_acc_common, mean_acc_task_il, _, _ = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il\
                if setting == 'task-il' else mean_acc_common
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('%sperformance/acc_task%02d' % (identifier, kk + 1), acc,
                                    task_number * args.n_epochs)
            a_logger.add_scalar('%sacc_mean' % identifier, mean_acc, task_number * args.n_epochs)

    def log_loss(self, loss: float, n_epochs: int, epoch: int,
                 task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss_a', loss, task_number * n_epochs + epoch)

    def log_stage_loss_(self, name: str, loss: float, n_epochs: int, epoch: int,
                 task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar(name, loss, task_number * n_epochs + epoch)

    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)

    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()
