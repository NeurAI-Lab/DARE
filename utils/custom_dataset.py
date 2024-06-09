import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data, targets, image_transform=None, train=False):
        self.data = data
        self.targets = targets
        self.image_transform = image_transform
        self.train = train
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        target = self.targets[idx]
        target = np.array(target)

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.image_transform:
            img = self.image_transform(img)

        target = torch.tensor(target, dtype=torch.long)

        if self.train:
            return img, target, not_aug_img
        else:
            return img, target