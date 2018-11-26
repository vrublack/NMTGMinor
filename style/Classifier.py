import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

class RepresentationClassifier(nn.Module):
    """
    Simple lstm classifier with 1 hidden layer
    """
    def __init__(self, opt, input_dim, classes=2):
        super(RepresentationClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, classes)
        self.cuda = (len(opt.gpus) >= 1)


    def forward(self, x):
        # input: (batch x seq_len x input_size)

        # simply classify each timestep separately, then sum
        x = self.fc(x)

        x = x.sum(dim=1)

        return log_softmax(x, dim=1)
