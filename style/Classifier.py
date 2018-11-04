import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

from onmt.modules.Transformer import TransformerDecoder


class RepresentationClassifier(nn.Module):
    """
    Simple lstm classifier with 1 hidden layer
    """

    def __init__(self, opt, preinit_decoders, dicts, positional_encoder, hidden_dim, dropout=0, classes=2):
        super(RepresentationClassifier, self).__init__()
        self.fc = nn.Linear(opt.model_size, classes)
        self.lstm = nn.LSTM(opt.model_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, classes)
        self.hidden_dim = hidden_dim
        self.cuda = (len(opt.gpus) >= 1)
        self.decoder = TransformerDecoder(opt, dicts, positional_encoder)

        # pre-initialize classifier decoder weights with reconstruction decoder weights
        self.avg_weights(self.decoder, preinit_decoders)

    def avg_weights(self, target_model, source_models):
        with torch.no_grad():
            dict_params = dict(source_models[0].named_parameters())

            for name, param in source_models[0].named_parameters():
                preinitialized = param
                for other in source_models[1:]:
                    preinitialized += dict(other.named_parameters())[name]
                preinitialized_avg = preinitialized / len(source_models)
                dict_params[name] = preinitialized_avg.detach()

            target_model.load_state_dict(dict_params, strict=False)

    def init_hidden(self, size_batch):
        # num_layers x minibatch_size x hidden_dim
        h, c = (torch.zeros(1, size_batch, self.hidden_dim),
                torch.zeros(1, size_batch, self.hidden_dim))
        if self.cuda:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self, tgt, context, src, grow=False):
        x, _ = self.decoder(tgt, context, src, grow=grow)

        # input: (batch x seq_len x input_size) but expected (seq_len x batch x input_size)
        x = x.transpose(0, 1)
        # hidden are ideally created only once but the batch size changes
        hidden = self.init_hidden(x.shape[1])
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out[-1, :, :]
        h = self.dropout(h)
        x = self.fc(h)
        x = x.squeeze(1)
        return log_softmax(x, dim=1)
