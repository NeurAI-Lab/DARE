# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def add_arguments(parser):
    parser.add_argument('--pretext_task', type=str, default='mse',
                        help='SSL training algorithm as a pretext task fo DER++')
    parser.add_argument('--er_weight', type=float, default=1,
                        help='weight for loss computed on the buffered images in ER.')
    parser.add_argument('--barlow_on_weight', type=float, default=0.5,
                        help='weight for barlow twin on_diag')
    parser.add_argument('--barlow_off_weight', type=float, default=0.05,
                        help='weight for barlow twin off_diag')
    parser.add_argument('--dino_weight', type=float, default=1,
                        help='weight for Dino Loss')
    parser.add_argument('--byol_weight', type=float, default=1,
                        help='weight for BYOL Loss')
    parser.add_argument('--simclr_weight', type=float, default=0.05,
                        help='weight for Dino Loss')
    parser.add_argument('--align_weight', type=float, default=0.5,
                        help='multitask weight for alignment loss')
    parser.add_argument('--uni_weight', type=float, default=0.1,
                        help='multitask weight for uniformity loss')
    parser.add_argument('--mi_weight', type=float, default=1,
                        help='multitask weight for mutual information')
    parser.add_argument('--img_size', type=int, required=True,
                        help='Input image size')
    parser.add_argument('--eval_c', action='store_true',
                        help='Use trained model for evaluation on natural corruption datasets')
    parser.add_argument('--relic', action='store_true',
                        help='Use kl-divergence to enforce invariance on cosine similarity matrix')
    parser.add_argument('--relicv2', action='store_true',
                        help='Use kl-divergence to enforce invariance without cosine similarity')
    parser.add_argument('--multicrop', action='store_true',
                        help='multicrop augmentation for buffered images')
    parser.add_argument('--size_crops', nargs='+', default=[64, 32],
                        help='size crops for multicrop')
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--mnist_seed', type=int, default=0)
    parser.add_argument('--b_percent', type=float, default=0.34,
                        help='The percentage of training time for step B.')
    parser.add_argument('--c_percent', type=float, default=0.34,
                        help='The percentage of training time for step C.')
    parser.add_argument('--maxd_weight', type=float, default=1)
    parser.add_argument('--mind_weight', type=float, default=1)
    parser.add_argument('--logitb_weight', type=float, default=1)
    parser.add_argument('--logitc_weight', type=float, default=1)
    parser.add_argument('--plot_results', action='store_true', help='Enable heatmap plotting')
    parser.add_argument('--log_accuracies', action='store_true', help='Enable tracking accuracies acorss tasks')
    parser.add_argument('--each_epoch', action='store_true',
                        help='Individual epochs for step B, C and A one after the other')
    parser.add_argument('--exclude_logit_loss_in_b_and_c', action='store_true',
                        help='exclude logit loss during stages b and c')
    parser.add_argument('--freeze_one_cls_in_b_and_c', action='store_true',
                        help='Freeze only one classifier during stages b and c')
    parser.add_argument('--reduce_lr', action='store_true',
                        help='Reduce the learning rate after task 1')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Set number of workers for dataloader')
    parser.add_argument('--output_folder', type=str, default='/output/mammoth/max_discrepancy_new/',
                        help='Output folder to store the results and logs.')
    parser.add_argument('--csv_filename', type=str, default='',
                       help='CSV filename to store the final accuracies.')
    parser.add_argument('--corruptions', '--names-list', nargs='+', default=['gaussian_noise', 'motion_blur', 'snow',
                                                                             'pixelate'],
                        help='List of corruptions to create domain-il on dataset of choice')
    parser.add_argument('--max_v', type=str, default='v1',
                        help='More iterations for step C and A with max_v > 1')


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--combine_category', action="store_true", default=False,
                        help='Whether to combine different noises from the same category as one task.')
    parser.add_argument('--max_first_task', action="store_true", default=False,
                        help='Whether to consider initialization as first task.')
    parser.add_argument('--maximize_task', type=str, default='l1',
                        help='Loss for maximizing and minimizing the discrepancy ["mse", "kl", "l1", "l2", "linf", '
                             '"cosine", "l1abs"]')
    parser.add_argument('--buffer_only', action="store_true", default=False,
                        help='Train one classifier only on buffered samples.')
    parser.add_argument('--no_stepc_buffer', action="store_true", default=False,
                        help='Buffer loss during step C.')
    parser.add_argument('--iterative_buffer', action="store_true", default=False,
                        help='Populate buffer every iteration.')
    parser.add_argument('--task_buffer', action="store_true", default=False,
                        help='Buffer sampling based on task id.')
    parser.add_argument('--dt_stepc', action="store_true", default=False,
                        help='CE loss on current task samples during step C.')
    parser.add_argument('--dt_stepb', action="store_true", default=False,
                        help='CE loss on current task samples during step B.')
    parser.add_argument('--no_stepa', action="store_true", default=False,
                        help='Remove step A loss after first task.')
    parser.add_argument('--freezeb_stepa', action="store_true", default=False,
                        help='Freeze backbone in step A after first task.')
    parser.add_argument('--linear_alpha', type=float, default=0,
                        help='Alpha value for linear mode connectivity.')
    parser.add_argument('--ema_update_freq', type=float, default=0,
                        help='Frequency with which we take ema update of the classifiers.')
    parser.add_argument('--cross_distill', action="store_true", default=False,
                        help="Distill first classifier's logits to second classifier on buffered samples")
    parser.add_argument('--adam_lr', type=float, default=0,
                        help='Train classifiers with Adam optimizer.')
    parser.add_argument('--norm_feature', action="store_true", default=False,
                        help="Normalize the features before final classifier layer")
    parser.add_argument('--diff_classifier', action="store_true", default=False,
                        help="Have separate classifier design to learn different functions")
    parser.add_argument('--finetune_classifiers', action="store_true", default=False,
                        help="Finetune the classifiers at the end of the task with buffer data")
    parser.add_argument('--finetuning_epochs', type=int, default=15,
                        help='Number of epochs to finetuning classifier')
    parser.add_argument('--finetune_lr', type=float, default=0.,
                        help='Learning rate during finetuning stage')
    parser.add_argument('--frozen_finetune', action="store_true", default=False,
                        help="Freeze the backbone during finetuning stage")
    parser.add_argument('--num_rotations', type=int, default=0,
                        help='Number of rotation classes for auxiliary loss')
    parser.add_argument('--rot_weight', type=float, default=0.,
                        help='Weight for rotation prediction loss')
    parser.add_argument('--supcon_weight', type=float, default=0.,
                        help='Weight for supervised contrast loss on the second classifier')
    parser.add_argument('--supcon_temp', type=float, default=0.,
                        help='Temperature scaling for supcon loss')
    parser.add_argument('--frozen_supcon', action="store_true", default=False,
                        help="Freeze the second classifier during supcon stage")
    parser.add_argument('--maxf_weight', type=float, default=0.,
                        help='Weight for CE final cross entropy')
    parser.add_argument('--lln', action='store_true',
                        help='set last linear to be linear and normalized')
    parser.add_argument('--use_bce', action='store_true',
                        help='Use binary cross entropy instead of CE')
    parser.add_argument('--intermediate_sampling', action='store_true',
                        help='store the samples in the buffer midway through training on the task')
    parser.add_argument('--std', type=float, default=1.,
                        help='Std dev for buffer sampling')
    parser.add_argument('--skewness', type=float, default=0.,
                        help='Skewness for normal distribution used in buffer sampling')
    parser.add_argument('--machine', type=str, default='',
                        help='Machine on which the job ran')
    parser.add_argument('--calculate_drift', action='store_true',
                        help='Calculate drift for buffered samples at every iteration')
    parser.add_argument('--weight_l1', type=float, default=1.,
                        help='Weight for current task loss l1')
    parser.add_argument('--weight_l3', type=float, default=1.,
                        help='Weight for buffer loss l3')
    parser.add_argument('--pretrained', action='store_true',
                        help='Enable pretrained weights for ResNet-18')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--aug_norm', action='store_true')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')

def add_auxiliary_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--aux', type=str, default='shape',
                        help='The type of auxiliary data')
    parser.add_argument('--shape_filter', type=str, default='sobel',
                        help='The type of auxiliary data')
    parser.add_argument('--shape_upsample_size', type=int, default=64,
                        help='size to upsample for sobel filter')
    parser.add_argument('--sobel_gauss_ksize', default=3, type=int)
    parser.add_argument('--sobel_ksize', default=3, type=int)
    parser.add_argument('--sobel_upsample', type=str, default='True')
    parser.add_argument('--loss_type', nargs='*', type=str, default=['kl'], help="--loss_type kl at")
    parser.add_argument('--loss_wt', nargs='*', type=float, default=[1.0, 1.0])
    parser.add_argument('--dir_aux', action='store_true')
    parser.add_argument('--buf_aux', action='store_true')

def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')
