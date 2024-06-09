import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.conf import set_random_seed
from datasets.seq_cifar10 import SequentialCIFAR10
from utils.conf import base_data_path

# Configuration
use_cuda = True

set_random_seed(10)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# ======================================================================================
# Load Clean Test data
# ======================================================================================
# set up data loader

test_transform = transforms.Compose(
            [transforms.ToTensor(), SequentialCIFAR10.get_normalization_transform()])

testset = torchvision.datasets.CIFAR10(base_data_path() + 'CIFAR10', train=False, download=True, transform=test_transform)
clean_test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, **kwargs)


# ======================================================================================
# Load Distortion data
# ======================================================================================
class DistortedDataset(Dataset):
    def __init__(self, data, targets, image_transform=None):

        self.data = data
        self.targets = targets
        self.image_transform = image_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img)
        target = self.targets[idx]
        target = np.array(target)

        if self.image_transform:
            img = self.image_transform(img)

        target = torch.tensor(target, dtype=torch.long)

        return img, target


transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# ======================================================================================
# Evaluate Robustness
# ======================================================================================
def eval_robustness(model, device, clean_data_loader, distorted_data_loader, condition_on_correct=True):

    model.eval()
    clean_data_iter = iter(clean_data_loader)
    distorted_data_iter = iter(distorted_data_loader)

    num_batches = len(clean_test_loader)

    clean_correct = 0
    clean_total = 0
    noisy_test_total = 0
    noisy_correct_total = 0

    for i in range(num_batches):
        c_X, c_y = next(clean_data_iter)
        d_X, d_y = next(distorted_data_iter)

        c_X, c_y, d_X, d_y = c_X.to(device), c_y.to(device), d_X.to(device), d_y.to(device)
        with torch.no_grad():
            c_out, _, _, _ = model(c_X)

            pred = c_out.max(1, keepdim=True)[1]
            clean_correct += pred.eq(c_y.view_as(pred)).sum().item()

            if condition_on_correct:
                select_idx = c_out.data.max(1)[1] == c_y.data
                d_X = d_X[select_idx]
                d_y = d_y[select_idx]

            d_out, _, _, _  = model(d_X)
            if isinstance(d_out, tuple):
                d_out = d_out[0]

            pred = d_out.max(1, keepdim=True)[1]
            noisy_correct_total += pred.eq(d_y.view_as(pred)).sum().item()
            noisy_test_total += len(d_X)
            clean_total += len(c_X)

    robustness = noisy_correct_total / noisy_test_total
    nat_acc = clean_correct / clean_total

    return nat_acc, robustness


def evaluate_natural_robustness(model, output_dir):

    lst_seeds = [10]

    lst_distortions = [
         'brightness',
         'contrast',
         'defocus_blur',
         'elastic_transform',
         'fog',
         'frost',
         'gaussian_blur',
         'gaussian_noise',
         'glass_blur',
         'impulse_noise',
         'jpeg_compression',
         'motion_blur',
         'pixelate',
         'saturate',
         'shot_noise',
         'snow',
         'spatter',
         'speckle_noise',
         'zoom_blur',
         ]

    rob_analysis_dict = dict()
    rob_analysis_dict['seed'] = []

    for distortion in lst_distortions:
        rob_analysis_dict[distortion] = []

    for seed in lst_seeds:
        print('-' * 60 + '\nSeed %s\n' % seed + '-' * 60)

        try:
            model.eval()

            rob_analysis_dict['seed'].append(seed)

            for distortion in lst_distortions:
                print('*' * 60 + '\nDistortion: %s\n' % distortion + '*' * 60)

                print('evaluate robustness')
                X = np.load(r'/input/CIFAR-10-C/%s.npy' % distortion)
                y = np.load(r'/input/CIFAR-10-C/labels.npy')

                rob_avg = 0
                nat_avg = 0

                for i in range(5):

                    print('+' * 60 + '\nSeverity %s\n' % (i + 1) + '+' * 60)
                    X_sel = X[i * 10000: (i + 1) * 10000]
                    y_sel = y[i * 10000: (i + 1) * 10000]

                    testset = DistortedDataset(X_sel, y_sel, transform_test)
                    distorted_test_loader = torch.utils.data.DataLoader(testset,
                                                                        batch_size=32,
                                                                        shuffle=False,
                                                                        **kwargs)

                    nat_acc, rob_acc = eval_robustness(model, device, clean_test_loader, distorted_test_loader, True)
                    # nat_acc, rob_acc = 1, 1

                    print('Natural Accuracy:', nat_acc)
                    print('Robustness Accuracy:', rob_acc)

                    rob_avg += rob_acc
                    nat_avg += nat_acc

                rob_avg /= 5
                nat_avg /= 5

                print('!' * 60 + '\nAverage Accuracy %s\n' % nat_avg + '!' * 60)
                print('!' * 60 + '\nAverage Robustness %s\n' % rob_avg + '!' * 60)

                rob_analysis_dict[distortion].append(rob_avg)

        except Exception as e:
            print(e)

    df = pd.DataFrame(rob_analysis_dict)
    df.to_csv(os.path.join(output_dir, 'nat_corruption_absolute.csv'), index=False)

