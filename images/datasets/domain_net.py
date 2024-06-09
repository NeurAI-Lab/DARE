# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from backbone.ResNet18 import resnet18
from backbone.ResNet18_mod import resnet18_mod
import torch.nn.functional as F
from utils.conf import base_data_path
from datasets.utils.continual_dataset import ContinualDataset, store_domain_loaders
from datasets.transforms.denormalization import DeNormalize
from PIL import Image
from utils import bce_with_logits


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    if type(flist) is list:
        for cur_flist in flist:
            with open(cur_flist, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel = line.strip().split(sep)
                    imlist.append((impath, int(imlabel)))
    else:
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                impath, imlabel = line.strip().split(sep)
                imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, not_aug_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = flist_reader(flist, sep)
        self.targets = np.array([datapoint[1] for datapoint in self.imlist])
        self.data = np.array([datapoint[0] for datapoint in self.imlist])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.not_aug_transform = not_aug_transform

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        original_img = img.copy()

        if self.not_aug_transform:
            not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        if self.not_aug_transform:
            return img, target, not_aug_img
        else:
            return img, target


    def __len__(self):
        return len(self.imlist)


class DomainNet(ContinualDataset):
    NAME = 'domain-net'
    SETTING = 'domain-2il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 6
    IMG_SIZE = 64
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    DOMAIN_LST = ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    TRANSFORM = [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                 transforms.RandomCrop(IMG_SIZE, padding=4), #transforms.RandomResizedCrop(IMG_SIZE),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()]
    TRANSFORM_NORM = [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                 transforms.RandomCrop(IMG_SIZE, padding=4), #transforms.RandomResizedCrop(IMG_SIZE),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize]
    TRANSFORM_TEST = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    NOT_AUG_TRANSFORM = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]

    data_path = base_data_path() + 'domain_net'
    annot_path = os.path.join(base_data_path() + 'domain_net_cl', 'version2')

    def get_data_loaders(self, task_id=None):

        if self.args.aug_norm:
            self.TRANSFORM.append(self.normalize)
            self.TRANSFORM_TEST.append(self.normalize)
            # self.NOT_AUG_TRANSFORM.append(self.normalize)

        if self.args.model == 'joint':
            transform = transforms.Compose([transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                     transforms.RandomCrop(self.IMG_SIZE, padding=4),  # transforms.RandomResizedCrop(IMG_SIZE),
                     transforms.ColorJitter(),
                     transforms.RandomGrayscale(),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()])
        else:
            transform = transforms.Compose(self.TRANSFORM)
        test_transform = transforms.Compose(self.TRANSFORM_TEST)
        not_aug_transform = transforms.Compose(self.NOT_AUG_TRANSFORM)

        train_dataset = ImageFilelist(
            root=self.data_path,
            flist=os.path.join(self.annot_path, self.DOMAIN_LST[self.i] + "_train.txt"),
            transform=transform,
            not_aug_transform=not_aug_transform,
            )

        test_dataset = ImageFilelist(
            root=self.data_path,
            flist=os.path.join(self.annot_path, self.DOMAIN_LST[self.i] + "_test.txt"),
            transform=test_transform,
            )

        train, test = store_domain_loaders(train_dataset, test_dataset, self)

        return train, test

    def not_aug_dataloader(self, batch_size):
        pass
        # return DataLoader(self.train_loader.dataset,
        #                   batch_size=batch_size, shuffle=True)

    @staticmethod
    def get_backbone(num_classifier=1, norm_feature=False, diff_classifier=False, num_rot=0, ema_classifier=False,
                     lln=False, dist_linear=False, algorithm='None', pretrained=False):
        if num_classifier == 3:
            return resnet18_mod(DomainNet.N_CLASSES_PER_TASK, num_rot=num_rot)
        else:
            return resnet18(DomainNet.N_CLASSES_PER_TASK, norm_feature=norm_feature, diff_classifier=diff_classifier,
                            num_rot=num_rot, ema_classifier=ema_classifier, lln=lln, algorithm=algorithm,
                            pretrained=pretrained)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Compose(DomainNet.TRANSFORM)])
        return transform

    @staticmethod
    def get_norm_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Compose(DomainNet.TRANSFORM_NORM)])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_loss(use_bce=False):
        if use_bce:
            return bce_with_logits
        else:
            return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform