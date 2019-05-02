import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class Discriminator(nn.Module):
    """
    Simple lstm classifier with 1 hidden layer
    """

    def __init__(self, opt, input_dim, hidden_dim, dropout=0, classes=2):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes)
        self.hidden_dim = hidden_dim
        self.use_cuda = opt.cuda

    def init_hidden(self, size_batch):
        # num_layers x minibatch_size x hidden_dim
        h, c = (torch.zeros(1, size_batch, self.hidden_dim),
                torch.zeros(1, size_batch, self.hidden_dim))
        if self.use_cuda:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self, x):
        # input: (seq_len x batch x input_size)
        # hidden are ideally created only once but the batch size changes
        hidden = self.init_hidden(x.shape[1])
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out[-1, :, :]
        h = self.dropout(h)
        x = self.fc(h)
        x = x.squeeze(1)
        return log_softmax(x, dim=1)

    @staticmethod
    def add_options(parser):
        parser.add_argument('-discriminator_size', type=int, default=20,
                            help='Dimension of hidden layer (rnn) in discriminator')
        parser.add_argument('-discriminator_dropout', type=float, default=0.1,
                            help='Dropout applied to rnn in discriminator')
        parser.add_argument('-discriminator', action='store_true',
                            help='Use a discriminator')
        parser.add_argument('-discriminator_weight', type=float, default=1.0,
                            help='Multiplier for the discriminator loss')

