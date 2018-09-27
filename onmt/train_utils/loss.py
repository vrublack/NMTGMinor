import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    """
    from https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510/4

    Calcultes the entropy (not cross-entropy)
    Expects output after softmax
    """
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = x * x.log()
        b = -1.0 * b.sum()
        return b
