from torch import nn


class MultiDecoder(nn.Module):
    """
    Wrapper for multiple decoders
    """

    def __init__(self, *decoders):
        super(MultiDecoder, self).__init__()
        self.n = len(decoders)
        self.decoders = decoders
        self.active_decoder = 0

    def set_active(self, i):
        if i >= self.n:
            raise ValueError()
        self.active_decoder = i

    def renew_buffer(self, new_len):
        return self.decoders[self.active_decoder].renew_buffer(new_len)

    def mark_pretrained(self):
        return self.decoders[self.active_decoder].mark_pretrained()

    def add_layers(self, n_new_layer):
        return self.decoders[self.active_decoder].add_layers(n_new_layer)

    def forward(self, input, context, src, **kwargs):
        return self.decoders[self.active_decoder].forward(input, context, src, **kwargs)

    def step(self, input, decoder_state):
        return self.decoders[self.active_decoder].step(input, decoder_state)

    def get_word_lut(self):
        return self.decoders[self.active_decoder].word_lut
