# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import xavier, num_flat_features
from utils import freeze_parameters

import copy


class MNISTMLP(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(MNISTMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )
        self.classifier1 = nn.Linear(100, self.output_size)
        self.classifier2 = nn.Linear(100, self.output_size)
        # self.linear = self.classifier
        # self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

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

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (100)
        """
        x = x.view(-1, num_flat_features(x))
        return self._features(x)

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self._features.apply(xavier)
        self.classifier1.apply(xavier)
        self.classifier2.apply(xavier)

    def forward(self, x: torch.Tensor, return_rep: bool=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))
        out = self._features(x)
        features = copy.deepcopy(out.data)
        logits1 = self.classifier1(out)
        logits2 = self.classifier2(out)
        if return_rep:
            output = {
                'logits1': logits1,
                'logits2': logits2,
                'logits3': None,
                'features': features
            }
        else:
            output = {
                'logits1': logits1,
                'logits2': logits2,
                'logits3': None
            }
        return output

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (input_size * 100
                    + 100 + 100 * 100 + 100 + 100 * output_size + output_size)
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
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.parameters()):
            if pp.grad is None:
                grads.append(torch.zeros(pp.shape).view(-1).to(pp.device))
            else:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads