import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

class RepresentationClassifier(nn.Module):
    """
    Simple feed-forward classifier with 1 hidden layer
    """
    def __init__(self, input_dim, hidden_dim, classes=2):
        super(RepresentationClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, classes)
        self.hidden_dim = hidden_dim

    def init_hidden(self, size_batch):
        # num_layers x minibatch_size x hidden_dim
        return (torch.zeros(1, size_batch, self.hidden_dim),
                torch.zeros(1, size_batch, self.hidden_dim))

    def forward(self, x):
        # input: (batch x seq_len x input_size) but expected (seq_len x batch x input_size)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.transpose(0, 1)
        # hidden are ideally created only once but the batch size changes
        hidden = self.init_hidden(x.shape[1])
        lstm_out, hidden = self.lstm(x, hidden)
        x = self.fc(lstm_out[-1,:,:])
        x = x.squeeze(1)
        return softmax(x, dim=1)