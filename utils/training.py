# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import DataParallel
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from torch.nn import functional as F
import sys
import math
import csv
from tqdm import tqdm
import numpy as np
from utils.eval_c import evaluate_natural_robustness
from torch.optim import SGD

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def save_task_perf(savepath, results, n_tasks):

    results_array = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i >= j:
                results_array[i, j] = results[i][j]

    np.savetxt(savepath, results_array, fmt='%.2f')


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate_overlap(model: ContinualModel, dataset: ContinualDataset) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :param eval_ema: flag to indicate if an exponential weighted average model
                     should be evaluated (For CLS-ER)
    :param ema_model: if eval ema is set to True, which ema model (plastic or stable)
                      should be evaluated (For CLS-ER)
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """

    curr_model = model.net
    status = curr_model.training
    curr_model.eval()

    sample_counter = 0
    overlap_counter = 0
    missing_counter = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = curr_model(inputs)

            m = torch.nn.Softmax(dim=1)
            if isinstance(outputs, dict):
                _, pred = torch.max(outputs['logits1'].data, 1)
                _, second_pred = torch.max(outputs['logits2'].data, 1)

            overlap_counter += torch.sum(pred == second_pred).item()
            missing_counter += torch.sum(pred != second_pred).item()
            sample_counter += pred.shape[0]

    return sample_counter, overlap_counter, missing_counter


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, eval_ema=False,
             ema_model=None, find_overlap=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :param eval_ema: flag to indicate if an exponential weighted average model
                     should be evaluated (For CLS-ER)
    :param ema_model: if eval ema is set to True, which ema model (plastic or stable)
                      should be evaluated (For CLS-ER)
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """

    curr_model = model.net
    if eval_ema:
        if ema_model == 'ema_model':
            print('setting evaluation model to EMA model')
            curr_model = model.ema_model
        else:
            raise NotImplementedError("EMA model type is not recognized")

    status = curr_model.training
    curr_model.eval()
    accs, second_acc, third_acc, avg_acc, voting_acc, confident_acc, accs_mask_classes = [], [], [], [], [], [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, second_correct, third_correct, avg_correct, voting_correct, confident_correct, \
        correct_mask_classes, total = .0, .0, .0, .0, .0, .0, .0, 0.
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = curr_model(inputs, k)
            else:
                outputs = curr_model(inputs)

            m = torch.nn.Softmax(dim=1)
            # FIXME: Changing eval script to take logits from first head alone
            if isinstance(outputs, dict):
                _, pred = torch.max(outputs['logits1'].data, 1)
                _, second_pred = torch.max(outputs['logits2'].data, 1)
                if outputs['logits3'] is not None:
                    _, third_pred = torch.max(outputs['logits3'].data, 1)
                else:
                    third_pred = None
                stacked_output = torch.stack((m(outputs['logits1']), m(outputs['logits2'])))
                _, average_pred = torch.max(stacked_output.mean(dim=0).data, 1)
                _, voting_pred = torch.max(torch.mean(F.one_hot(stacked_output.argmax(2),
                                                      num_classes=stacked_output.shape[-1]).type(torch.float32),
                                                      dim=0), 1)
                stacked_output = stacked_output.permute(1,0,2)
                _, confident_pred = torch.max(stacked_output[torch.arange(len(stacked_output)),
                                                             stacked_output.amax(dim=2).max(dim=1)[1]], 1)
            else:
                _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            second_correct += torch.sum(second_pred == labels).item()
            if third_pred is None:
                third_correct = 0.
            else:
                third_correct += torch.sum(third_pred == labels).item()
            avg_correct += torch.sum(average_pred == labels).item()
            voting_correct += torch.sum(voting_pred == labels).item()
            confident_correct += torch.sum(confident_pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs['logits1'], dataset, k)
                _, pred = torch.max(outputs['logits1'].data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        second_acc.append(second_correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        third_acc.append(third_correct / total * 100
                          if 'class-il' in model.COMPATIBILITY else 0)
        avg_acc.append(avg_correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        voting_acc.append(voting_correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        confident_acc.append(confident_correct / total * 100
                          if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    curr_model.train(status)
    return accs, accs_mask_classes, avg_acc, voting_acc, confident_acc, second_acc, third_acc


def train_epoch(train_loader, model, epoch, t, args, n_epochs, tb_logger, model_stash, stage):
    loss_a, loss_b, loss_c, loss_aux_ce, loss_aux_logit = 0, 0, 0, 0, 0

    # set drift calculation to true if maximum discrepancy is the model
    if ("max" in model.NAME or "er" in model.NAME) and args.calculate_drift and t==0 and epoch>45:
        model.calculate_drift = True

    for i, data in enumerate(train_loader):
        inputs, labels, not_aug_inputs = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        task_ids = torch.repeat_interleave(torch.tensor(t), inputs.shape[0])
        model.opt.zero_grad()
        if stage == 'b' and "max" in model.NAME:
            model.net.freeze(name='backbone')
            if args.freeze_one_cls_in_b_and_c:
                model.net.freeze(name='classifier_1')
            loss, loss_aux_1, loss_aux_2 = model.stepb(inputs, labels, not_aug_inputs, task_ids)
            loss_b += loss
        elif stage == 'c' and "max" in model.NAME:
            model.net.freeze(name='classifiers')
            loss, loss_aux_1, loss_aux_2 = model.stepc(inputs, labels, t)
            loss_c += loss
        elif stage == 'd' and "max" in model.NAME:
            model.net.unfreeze()
            loss, loss_aux_1, loss_aux_2 = model.stepd()
        else:
            model.net.unfreeze()
            if t > 0 and args.freezeb_stepa:
                model.net.freeze(name='backbone')
            loss, loss_aux_1, loss_aux_2 = model.observe(inputs, labels, not_aug_inputs, task_ids, epoch=epoch)
            loss_a += loss

        loss_aux_ce += loss_aux_1
        loss_aux_logit += loss_aux_2
        progress_bar(i, len(train_loader), epoch, t, loss)

        if args.tensorboard:
            tb_logger.log_stage_loss_('loss_a', loss_a / len(train_loader), n_epochs, epoch, t)
            tb_logger.log_stage_loss_('loss_b', loss_b / len(train_loader), n_epochs, epoch, t)
            tb_logger.log_stage_loss_('loss_c', loss_c / len(train_loader), n_epochs, epoch, t)
            tb_logger.log_stage_loss_('memory_ce_loss', loss_aux_ce / len(train_loader), n_epochs, epoch, t)
            tb_logger.log_stage_loss_('memory_logit_loss', loss_aux_logit / len(train_loader), n_epochs, epoch, t)

        model_stash['batch_idx'] = i + 1
    model_stash['epoch_idx'] = epoch + 1
    model_stash['batch_idx'] = 0


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    if torch.cuda.device_count() > 1:
        model.net = DataParallel(model.net)
    model.net.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    ema_loggers = {}
    ema_results = {}
    ema_results_mask_classes = {}
    ema_task_perf_paths = {}

    if hasattr(model, 'ema_model'):
        ema_results['ema_model'], ema_results_mask_classes['ema_model'] = [], []

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, tb_logger.get_log_dir())
        task_perf_path = os.path.join(tb_logger.get_log_dir(),  'task_performance.txt')
        if hasattr(model, 'ema_model'):
            print(f'Creating Logger for ema_model')
            print('=' * 50)
            ema_loggers['ema_model'] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, os.path.join(tb_logger.get_log_dir() , 'ema_model'))
            ema_task_perf_paths['ema_model'] = os.path.join(tb_logger.get_log_dir(),  'task_performance_ema.txt')

    dataset_copy = get_dataset(args)
    # the loop is mainly to populate the test loaders for each task
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders(task_id=t)
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task, _, _, _, _, _ = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    # list to store task 1 accuracies
    if args.log_accuracies:
        task_accuracies = {
            'acc_task1' : [],
            'acc_task2' : [],
            'acc_task3' : [],
            'acc_task4' : [],
            'acc_task5' : [],
            'acc_task6' : []
        }

        acc_alltasks = []
    for t in range(dataset.N_TASKS):
        if t >= 1 and args.reduce_lr:
            model.opt = SGD(model.net.parameters(), lr=args.lr/10.)

        model.net.train()
        # unfreeze whole model (require_grad = True)
        for name, param in model.net.named_parameters():
            param.requires_grad = True

        train_loader, test_loader = dataset.get_data_loaders(task_id=t)
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

            if hasattr(model, 'ema_model'):
                ema_accs = evaluate(model, dataset, eval_ema=True, ema_model='ema_model', last=True)
                ema_results['ema_model'][t - 1] = ema_results['ema_model'][t - 1] + ema_accs[0]

                if dataset.SETTING == 'class-il':
                    ema_results_mask_classes['ema_model'][t - 1] = ema_results_mask_classes['ema_model'][t - 1] + \
                                                                 ema_accs[1]

        n_epochs = args.n_epochs
        b_epochs = math.floor(args.b_percent * n_epochs)
        c_epochs = math.floor(args.c_percent * n_epochs) + b_epochs

        if model.NAME!='joint' or (model.NAME=='joint' and t==dataset.N_TASKS):
            for epoch in range(n_epochs):
                if not args.each_epoch:
                    if t < 1 and not args.max_first_task:
                        stage = 'a'
                    else:
                        if epoch < b_epochs:
                            stage = 'b'
                        elif c_epochs > epoch >= b_epochs:
                            stage = 'c'
                        elif epoch >= c_epochs:
                            stage = 'a'

                elif args.each_epoch:
                    # train epochs in order of step B, C, and A
                    if t < 1 and not args.max_first_task:
                        stage = 'a'
                    elif not args.no_stepa:
                        if epoch%3==0:
                            stage = 'b'
                        elif epoch%3==1:
                            stage = 'c'
                        elif epoch%3==2:
                            stage = 'a'
                    elif t > 0 and args.no_stepa:
                        if epoch % 2 == 0:
                            stage = 'b'
                        elif epoch % 2 == 1:
                            stage = 'c'

                train_epoch(train_loader=train_loader, model=model, epoch=epoch, t=t, args=args,
                                n_epochs=n_epochs, tb_logger=tb_logger, model_stash=model_stash, stage=stage)

                if t>=1 and args.log_accuracies:
                    temp_acc = evaluate(model, dataset)
                    task_accuracies["acc_task1"].append(temp_acc[0][0])
                    cur_task_key = f"acc_task{t+1}"
                    task_accuracies[cur_task_key].append(temp_acc[0][t])
                    for it in [2,3,4,5,6]:
                        if it != t+1:
                            cur_task_key = f"acc_task{it}"
                            task_accuracies[cur_task_key].append(0)

                    acc_alltasks.append(np.mean(temp_acc, axis=1)[0])

            if args.each_epoch and (n_epochs-1)%3!=2 and t>0:
                # one last epoch with derpp loss in case if the last epoch for the task was trained with B or C
                train_epoch(train_loader=train_loader, model=model, epoch=epoch, t=t, args=args,
                            n_epochs=n_epochs, tb_logger=tb_logger, model_stash=model_stash, stage='a')

            if args.buffer_only and t == dataset.N_TASKS-1:
                # last training of first classifier on buffered samples alone
                for epoch in range(n_epochs):
                    train_epoch(train_loader=train_loader, model=model, epoch=epoch, t=t, args=args,
                                n_epochs=n_epochs, tb_logger=tb_logger, model_stash=model_stash, stage='d')

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if hasattr(model, 'final_finetune') and t == dataset.N_TASKS-1:
            model.final_finetune()

        accs = evaluate(model, dataset)

        # log task 1 accuracies
        if t == 0 and args.log_accuracies:
            acc_alltasks.append(accs[0][0])
            task_accuracies["acc_task1"].append(accs[0][0])
            task_accuracies["acc_task2"].append(0)
            task_accuracies["acc_task3"].append(0)
            task_accuracies["acc_task4"].append(0)
            task_accuracies["acc_task5"].append(0)
            task_accuracies["acc_task6"].append(0)

        if t+1 == dataset.N_TASKS and args.log_accuracies:
            with open(os.path.join(tb_logger.get_log_dir(), 'task1_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task1"])

            with open(os.path.join(tb_logger.get_log_dir(), 'task2_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task2"])

            with open(os.path.join(tb_logger.get_log_dir(), 'task3_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task3"])

            with open(os.path.join(tb_logger.get_log_dir(), 'task4_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task4"])

            with open(os.path.join(tb_logger.get_log_dir(), 'task5_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task5"])

            with open(os.path.join(tb_logger.get_log_dir(), 'task6_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(task_accuracies["acc_task6"])

            with open(os.path.join(tb_logger.get_log_dir(), 'all_task_accuracies.csv'), 'w') as f:
                write = csv.writer(f)
                write.writerow(acc_alltasks)

        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)

        if args.tensorboard:
            # tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
            tb_logger.log_all_accuracy(np.array(accs), mean_acc, args, t)
            csv_logger.log(mean_acc)

        # Evaluate on EMA model
        if hasattr(model, 'ema_model'):
            print('=' * 30)
            print(f'Evaluating ema_model')
            print('=' * 30)
            ema_accs = evaluate(model, dataset, eval_ema=True, ema_model='ema_model')

            ema_results['ema_model'].append(ema_accs[0])
            ema_results_mask_classes['ema_model'].append(ema_accs[1])
            ema_mean_acc = np.mean(ema_accs, axis=1)
            print_mean_accuracy(ema_mean_acc, t + 1, dataset.SETTING)

            if args.tensorboard:
                tb_logger.log_all_accuracy(np.array(ema_accs), ema_mean_acc, args, t, identifier='ema_model_')
                ema_loggers['ema_model'].log(ema_mean_acc)

        if args.plot_results:
            # save task checkpoint
            fname = os.path.join(tb_logger.get_log_dir(), 'checkpoint_{}.pth'.format(str(t+1)))
            if torch.cuda.device_count() > 1 and args.plot_results:
                torch.save(model.net.module.state_dict(), fname)
            elif args.plot_results:
                torch.save(model.net.state_dict(), fname)


    if args.tensorboard:
        tb_logger.close()
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)

        if 'ema_model' in ema_loggers:
            ema_loggers['ema_model'].add_bwt(ema_results['ema_model'], ema_results_mask_classes['ema_model'])
            ema_loggers['ema_model'].add_forgetting(ema_results['ema_model'], ema_results_mask_classes['ema_model'])

        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

        if 'ema_model' in ema_loggers:
            args_dict = vars(args)
            args_dict['ema_acc'] = np.mean(ema_accs, axis=1)[0]
            args_dict['ema_acc_average'] = np.mean(ema_accs, axis=1)[2]

        sample_counter, overlap_counter, missing_counter = evaluate_overlap(model, dataset)
        args_dict = vars(args)
        args_dict['sample_counter'] = sample_counter
        args_dict['overlap_counter'] = overlap_counter
        args_dict['missing_counter'] = missing_counter

        csv_logger.write(vars(args))
        save_task_perf(task_perf_path, results, dataset.N_TASKS)

        if 'ema_model' in ema_loggers:
            ema_loggers['ema_model'].write(vars(args))
            save_task_perf(ema_task_perf_paths['ema_model'], ema_results['ema_model'], dataset.N_TASKS)

    # store mse_distances between buffered representations
    if args.calculate_drift:
        with open(os.path.join(tb_logger.get_log_dir(), 'mse_distances.csv'), 'w') as f:
            write = csv.writer(f)
            write.writerow(model.drift)

    # save checkpoint
    fname = os.path.join(tb_logger.get_log_dir(), 'checkpoint.pth')
    if torch.cuda.device_count() > 1 and args.plot_results:
        torch.save(model.net.module.state_dict(), fname)
    elif args.plot_results:
        torch.save(model.net.state_dict(), fname)

    # evaluate natural corruption
    if args.eval_c:
        evaluate_natural_robustness(model.net, tb_logger.get_log_dir())
