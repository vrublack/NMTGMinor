import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

class RepresentationClassifier(nn.Module):
    """
    Simple lstm classifier with 1 hidden layer
    """
    def __init__(self, opt, input_dim, hidden_dim, dropout=0, classes=2):
        super(RepresentationClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes)
        self.hidden_dim = hidden_dim
        self.cuda = (len(opt.gpus) >= 1)


    def init_hidden(self, size_batch):
        # num_layers x minibatch_size x hidden_dim
        h, c = (torch.zeros(1, size_batch, self.hidden_dim),
                torch.zeros(1, size_batch, self.hidden_dim))
        if self.cuda:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self, x):
        # input: (batch x seq_len x input_size) but expected (seq_len x batch x input_size)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.transpose(0, 1)
        # hidden are ideally created only once but the batch size changes
        hidden = self.init_hidden(x.shape[1])
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out[-1,:,:]
        h = self.dropout(h)
        x = self.fc(h)
        x = x.squeeze(1)
        return log_softmax(x, dim=1)