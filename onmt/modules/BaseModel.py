import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import onmt


#~ from onmt.modules.Transformer.Layers import XavierLinear

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        
    def forward(self, input, log_softmax=True):
        
        logits = self.linear(input)
        
        if log_softmax:
            output = F.log_softmax(logits, dim=-1)
        else:
            output = logits
        return output
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator        
        
    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator.linear.weight = self.decoder.word_lut.weight
        
    
    def share_enc_dec_embedding(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight


class Reconstructor(nn.Module):
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator        
    
    
    
