import torch
import torch.nn as nn


def positive_mask(batch_size):
    """
    Create a mask for masking positive samples
    :param batch_size:
    :return: A mask that can segregate 2(N-1) negative samples from a batch of N samples
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool)
    mask[torch.eye(N).bool()] = 0
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


class NTXent(nn.Module):
    """
    The Normalized Temperature-scaled Cross Entropy Loss
    Source: https://github.com/Spijkervet/SimCLR
    """

    def __init__(self):
        super(NTXent, self).__init__()
        self.temperature = 0.5
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = None
        self.N = None
        self.mask = None

    def forward(self, zx, zy, labels=None):
        """
        zx: projection output of batch zx
        zy: projection output of batch zy
        :return: normalized loss
        """
        self.batch_size = zx.shape[0]
        self.N = 2 * self.batch_size
        self.mask = positive_mask(self.batch_size)
        positive_samples, negative_samples = self.sample_no_dict(zx, zy)
        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= self.N
        return loss

    def sample_no_dict(self, zx, zy):
        """
        Positive and Negative sampling without dictionary
        """
        z = torch.cat((zx, zy), dim=0)
        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = torch.div(torch.matmul(z, z.T), self.temperature)

        # Extract positive samples
        sim_xy = torch.diag(sim, self.batch_size)
        sim_yx = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_xy, sim_yx), dim=0).reshape(self.N, 1)

        # Extract negative samples
        negative_samples = sim[self.mask].reshape(self.N, -1)
        return positive_samples, negative_samples

