import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import onmt, math


#~ from onmt.modules.Transformer.Layers import XavierLinear

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        #~ self.linear = onmt.modules.Transformer.Layers.XavierLinear(hidden_size, output_size)
        self.linear = nn.Linear(hidden_size, output_size)
        
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        
        self.linear.bias.data.zero_()
            
        
        
    def forward(self, input, log_softmax=True):
        
        # added float to the end 
        logits = self.linear(input).float() 
        
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
        raise NotImplementedError('Not possible for multidecoder')
        
    
    def share_enc_dec_embedding(self):
        raise NotImplementedError('Not possible for multidecoder')
        
    def mark_pretrained(self):
        
        self.encoder.mark_pretrained()
        self.decoder.mark_pretrained()
        
    


class Reconstructor(nn.Module):
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator        
    

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    
    
