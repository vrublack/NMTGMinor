import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

class RepresentationClassifier(nn.Module):
    """
    Simple feed-forward classifier with 1 hidden layer
    """
    def __init__(self, input_dim, hidden_dim, classes=2):
        super(RepresentationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        x.squeeze(1)
        return softmax(x, dim=1)