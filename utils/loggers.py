# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il' or 'domain-2il' or 'domain-supcif':
        mean_acc, _, avg_acc, voting_acc, confident_acc, second_acc, third_acc = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
        print('Second Accuracy for {} task(s): {} %'.format(
            task_number, round(second_acc, 2)), file=sys.stderr)
        print('Third Accuracy for {} task(s): {} %'.format(
            task_number, round(third_acc, 2)), file=sys.stderr)
        print('Average Accuracy for {} task(s): {} %'.format(
            task_number, round(avg_acc, 2)), file=sys.stderr)
        print('Voting Accuracy for {} task(s): {} %'.format(
            task_number, round(voting_acc, 2)), file=sys.stderr)
        print('Confident Accuracy for {} task(s): {} %'.format(
            task_number, round(confident_acc, 2)), file=sys.stderr)

    else:
        mean_acc_class_il, mean_acc_task_il, _, _ = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, log_dir: str) -> None:
        self.accs = []
        self.avg_accs = []
        self.voting_accs = []
        self.confident_accs = []
        self.second_accs = []
        self.third_accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.log_dir = log_dir
        create_if_not_exists(self.log_dir)
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il' or self.setting == 'domain-2il' or self.setting == 'domain-supcif':
            mean_acc, _, mean_avg_acc, mean_voting_acc, mean_confident_acc, mean_second_acc, mean_third_acc = mean_acc
            self.accs.append(mean_acc)
            self.avg_accs.append(mean_avg_acc)
            self.voting_accs.append(mean_voting_acc)
            self.confident_accs.append(mean_confident_acc)
            self.second_accs.append(mean_second_acc)
            self.third_accs.append(mean_third_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args['task' + str(i + 1)] = acc
            new_cols.append('task' + str(i + 1))

        args['second_acc'] = self.second_accs[-1]
        new_cols.append('second_acc')

        args['third_acc'] = self.third_accs[-1]
        new_cols.append('third_acc')

        args['avg_acc'] = self.avg_accs[-1]
        new_cols.append('avg_acc')

        args['voting_acc'] = self.voting_accs[-1]
        new_cols.append('voting_acc')

        args['confident_acc'] = self.confident_accs[-1]
        new_cols.append('confident_acc')

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns

        # create_if_not_exists(base_path() + "results/" + self.setting)
        # create_if_not_exists(base_path() + "results/" + self.setting +
        #                      "/" + self.dataset)
        # create_if_not_exists(base_path() + "results/" + self.setting +
        #                      "/" + self.dataset + "/" + self.model)

        write_headers = False
        path = self.log_dir + "/" + "mean_accs_1.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        # log all runs in a common file
        if args['csv_filename'] != '':
            results_file = "/{}".format(args['csv_filename'])
        else:
            results_file = "/results_{}tasks.csv".format(args['num_tasks'])
        write_headers = False
        path = "/".join(self.log_dir.split("/")[:-1]) + results_file
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        if self.setting == 'class-il':
            # create_if_not_exists(base_path() + "results/task-il/"
            #                      + self.dataset)
            # create_if_not_exists(base_path() + "results/task-il/"
            #                      + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                args['task' + str(i + 1)] = acc

            args['forward_transfer'] = self.fwt_mask_classes
            args['backward_transfer'] = self.bwt_mask_classes
            args['forgetting'] = self.forgetting_mask_classes

            write_headers = False
            path = self.log_dir + "/" + "mean_accs_2.csv"
            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)

def plot(img, name):
    dir = os.path.join(os.getcwd(), 'vis')  #"/volumes2/feature_prior_project/art/feature_prior/vis"
    out = rf"{dir}/{name}.jpg"
    img.save(out)
