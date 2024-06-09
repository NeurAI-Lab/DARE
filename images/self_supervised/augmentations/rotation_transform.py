# from torchvision import transforms
# from torchvision.transforms import functional as F
# import torch
#
#
# class RotationTransform:
#     """
#     Rotation transform as an SSL pretext task
#     """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, x):
#         p = torch.rand(1)
#         if p < 0.25:
#             return F.rotate(x, 90), torch.tensor(0)
#         elif 0.25 <= p < 0.5:
#             return F.rotate(x, 180), torch.tensor(1)
#         elif 0.50 <= p < 0.75:
#             return F.rotate(x, 270), torch.tensor(2)
#         else:
#             return x, torch.tensor(3)
#

from torchvision import transforms
from torchvision.transforms import functional as F
import torch


class RotationTransform:
    """
    Rotation transform as an SSL pretext task
    """

    def __init__(self, num_rotations=12):
        self.num_rotations = num_rotations
        self.lst_prob = []
        self.lst_angle = []
        self.prob = 1 / self.num_rotations
        self.angle = 360 / self.num_rotations
        for i in range(0, self.num_rotations + 1):
            self.lst_prob.append(i * self.prob)
            self.lst_angle.append(i * self.angle)

    def __call__(self, x):
        p = torch.rand(1)
        for i in range(1, self.num_rotations+1):
            if self.lst_prob[i-1] <= p < self.lst_prob[i]:
                angle = self.lst_angle[i-1]
                if angle > 0:
                    return F.rotate(x, angle), torch.tensor(i-1)
                else:
                    return x, torch.tensor(0)

