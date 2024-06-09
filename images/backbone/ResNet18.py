# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from utils import freeze_parameters


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class NormalizedLinear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight/weight_norm
        return F.linear(X, self.lln_weight if self.training else self.lln_weight.detach(), self.bias)


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, norm_feature: bool=False, diff_classifier: bool=False,
                 num_rot: int=0, ema_classifier: bool=False, lln: bool=False, algorithm='None', pretrained=False) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.num_rot = num_rot
        self.norm_feature = norm_feature
        self.ema_classifier = ema_classifier
        self.lln = lln
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.algorithm = algorithm

        if self.lln:
            self.classifier1 = NormalizedLinear(nf * 8 * block.expansion, num_classes)
        else:
            self.classifier1 = nn.Linear(nf * 8 * block.expansion, num_classes)


        if 'max' in self.algorithm:
            if not diff_classifier:
                self.classifier2 = nn.Linear(nf * 8 * block.expansion, num_classes)
            elif self.lln:
                self.classifier2 = NormalizedLinear(nf * 8 * block.expansion, num_classes)
            else:
                self.classifier2 = nn.Sequential(
                    nn.Linear(nf * 8 * block.expansion, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )

        # C3 can be either rotation prediction or ema update from C1
        if num_rot > 0:
            self.classifier3 = nn.Sequential(
                nn.Linear(nf * 8 * block.expansion, nf * 8 * block.expansion),
                nn.BatchNorm1d(nf * 8 * block.expansion),
                nn.ReLU(inplace=True),
                # nn.Linear(nf * 8 * block.expansion, nf * 8 * block.expansion),
                # nn.BatchNorm1d(nf * 8 * block.expansion),
                # nn.ReLU(inplace=True),
                nn.Linear(nf * 8 * block.expansion, num_rot)
            )
        elif ema_classifier:
            self.classifier3 = nn.Linear(nf * 8 * block.expansion, num_classes)
        # self.prediction = nn.Linear(num_classes, num_classes)
        # self.classifier = self.linear

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze(self, name):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        self.unfreeze()
        self.train()

        if name == 'backbone':
            freeze_parameters(self, requires_grad=requires_grad)
            freeze_parameters(self.classifier1, requires_grad=not requires_grad)
            freeze_parameters(self.classifier2, requires_grad=not requires_grad)
            if hasattr(self, 'classifier3'):
                freeze_parameters(self.classifier3, requires_grad=not requires_grad)
        elif name == 'classifiers':
            freeze_parameters(self.classifier1, requires_grad=requires_grad)
            freeze_parameters(self.classifier2, requires_grad=requires_grad)
        elif name == 'classifier_1':
            freeze_parameters(self.classifier1, requires_grad=requires_grad)
        elif name == 'classifier_2':
            freeze_parameters(self.classifier2, requires_grad=requires_grad)
        else:
            raise NotImplementedError(f'Unknown name={name}.')

    def unfreeze(self):
        """Unfreeze the whole module."""
        freeze_parameters(self, requires_grad=True)
        self.train()

    def forward(self, x: torch.Tensor, return_rep: bool=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        features = copy.deepcopy(out.data)
        if self.norm_feature:
            out = F.normalize(out, p=2.0, dim=-1)
        logits1 = self.classifier1(out)
        if 'max' in self.algorithm:
            logits2 = self.classifier2(out)
        else:
            logits2 = self.classifier1(out)
        if self.num_rot > 0 or self.ema_classifier:
            logits3 = self.classifier3(out)
        else:
            logits3 = None
        if return_rep:
            output = {
                'logits1': logits1,
                'logits2': logits2,
                'logits3': logits3,
                'features': features
            }
        else:
            output = {
                'logits1': logits1,
                'logits2': logits2,
                'logits3': logits3,
            }
        return output

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def resnet18(nclasses: int, nf: int=64, norm_feature: bool=False, diff_classifier: bool=False, num_rot: int=0,
             ema_classifier: bool=False, lln: bool=False, algorithm='None', pretrained=False):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, norm_feature=norm_feature, diff_classifier=diff_classifier,
                  num_rot=num_rot, ema_classifier=ema_classifier, lln=lln, algorithm=algorithm, pretrained=pretrained)
    # resnet = models.resnet18(pretrained=False)
    # resnet.fc = torch.nn.Linear(resnet.fc.in_features, nclasses)
    # return resnet


def get_resnet18(nclasses: int=1000, nf: int=64):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)