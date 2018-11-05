import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from onmt.ModelConstructor import build_model
import torch.nn.functional as F


model_list = ['transformer', 'stochastic_transformer']

class EnsembleTranslator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self.beta = opt.beta
        self.alpha = opt.alpha
        self.start_with_bos = opt.start_with_bos
        
        self.models = list()
        self.model_types = list()
        
        # models are string with | as delimiter
        models = opt.model.split("|")
        
        print(models)
        self.n_models = len(models)
        self._type = 'text'
        
        for i, model in enumerate(models):
            if opt.verbose:
                print('Loading model from %s' % model)
            checkpoint = torch.load(model, map_location='cpu')
                               
            model_opt = checkpoint['opt']
            
            if i == 0:
                self.src_dict = checkpoint['dicts']['style1']
                self.tgt_dict = checkpoint['dicts']['style1']
            
            # Build model from the saved option
            model = build_model(model_opt, checkpoint['dicts'])
            
            model.load_state_dict(checkpoint['model'])
            
            if model_opt.model in model_list:
                model.decoder.check_renew_buffer(self.opt.max_sent_length)
            
            if opt.cuda:
                model = model.cuda()
            else:
                model = model.cpu()

            self.models.append(model)
            self.model_types.append(model_opt.model)
            
        self.cuda = opt.cuda
        self.ensemble_op = opt.ensemble_op
        
        if opt.verbose:
            print('Done')

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}
    
    # Combine distributions from different models
    def _combineOutputs(self, outputs):
        
        if len(outputs) == 1:
            return outputs[0]
        
        if self.ensemble_op == "logSum":
            output = (outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = F.log_softmax(output, dim=-1)
        elif self.ensemble_op == "mean":
            output = torch.exp(outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = torch.log(output)
        elif self.ensemble_op == 'gmean':
            output = torch.exp(outputs[0])
            
            # geometric mean of the probabilities
            for i in range(1, len(outputs)):
                output *= torch.exp(outputs[i])
                
            # have to normalize
            output.pow_(1.0 / float(len(outputs)))
            norm_ = torch.norm(output, p=1, dim=-1)
            output.div_(norm_.unsqueeze(-1))

            
            output = torch.log(output)
        else:
            raise ValueError('Emsemble operator needs to be "mean" or "logSum", the current value is %s' % self.ensemble_op)
        
        return output
    
    # Take the average of attention scores
    def _combineAttention(self, attns):
        
        attn = attns[0]
        
        for i in range(1, len(attns)):
            attn += attns[i]
        
        attn.div(len(attns))
        
        return attn

    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)
            
    def to_variable(self, data):
        
        for i, t in enumerate(data):
            if data[i] is not None:
                if self.cuda:
                    data[i] = Variable(data[i].cuda())
                else:
                    data[i] = Variable(data[i])
            else:
                data[i] = None
        return data

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        
        if self.start_with_bos:
            srcData = [self.src_dict.convertToIdx(b,
                              onmt.Constants.UNK_WORD,
                              onmt.Constants.BOS_WORD)
                       for b in srcBatch]
        else:
            srcData = [self.src_dict.convertToIdx(b,
                              onmt.Constants.UNK_WORD)
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, 9999,
                            [self.opt.gpu], volatile=True,
                            data_type=self._type, max_seq_num =self.opt.batch_size)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        
        return tokens

    def translateBatch(self, rawSrcBatch, srcBatch, tgtBatch, lr):

        # Batch size is in different location depending on data.

        beamSize = self.opt.beam_size
        batchSize = self._getBatchSize(srcBatch)
                    
        vocab_size = self.tgt_dict.size()
        allHyp, allScores, allAttn, allLengths = [], [], [], []
        
        # srcBatch should have size len x batch
        # tgtBatch should have size len x batch
        
        contexts = dict()

        src = srcBatch.transpose(0, 1)

        def auto_lr(raw_sent):
            len_wo_bpe = len(' '.join(raw_sent).replace('@@ ', ''))
            if len_wo_bpe <= 15:
                return 30
            elif len_wo_bpe <= 25:
                return 6
            elif len_wo_bpe <= 35:
                return 2
            elif len_wo_bpe <= 45:
                return 2
            elif len_wo_bpe <= 55:
                return 1
            elif len_wo_bpe <= 65:
                return 2
            elif len_wo_bpe <= 75:
                return 1.5
            elif len_wo_bpe <= 85:
                return 2.5
            else:
                return 2.5
        
        #  (1) run the encoders on the src
        for i in range(self.n_models):
            contexts[i], src_mask = self.models[i].encoder(src)
            contexts[i] = torch.autograd.Variable(contexts[i], requires_grad=True)
            self.models[i].repr_classifier.cuda = False
            classified_repr = self.models[i].repr_classifier(contexts[i])
            modified_context = contexts[i]
            for b in range(classified_repr.shape[0]):
                classified_repr[b, self.opt.target_style - 1].backward(retain_graph=True)
                if lr is None:
                    learning_rate = auto_lr(rawSrcBatch[b])
                else:
                    learning_rate = lr
                modified_context = modified_context + learning_rate * contexts[i].grad
                contexts[i].grad.zero_()

            contexts[i] = modified_context


        goldScores = contexts[0].data.new(batchSize).zero_()
        goldWords = 0
        
        if tgtBatch is not None:
            # Use the first model to decode
            model_ = self.models[0]
        
            tgtBatchInput = tgtBatch[:-1]
            tgtBatchOutput = tgtBatch[1:]
            tgtBatchInput = tgtBatchInput.transpose(0,1)
            
            output, coverage = model_.decoder(tgtBatchInput, contexts[0], src)
            output = output.transpose(0, 1) # transpose to have time first, like RNN models
            
            
            #  (2) if a target is specified, compute the 'goldScore'
            #  (i.e. log likelihood) of the target under the model
            for dec_t, tgt_t in zip(output, tgtBatchOutput.data):
                gen_t = model_.generator(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores.squeeze(1)
                goldWords += tgt_t.ne(onmt.Constants.PAD).sum()
            
            
        #  (3) Start decoding
            
        # time x batch * beam
        src = Variable(srcBatch.data.repeat(1, beamSize))
        
        # initialize the beam
        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        
        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        
        decoder_states = dict()
        
        decoder_hiddens = dict()
        
        for i in range(self.n_models):
            decoder_states[i] = self.models[i].create_decoder_state(src.clone(), contexts[i], beamSize)
        
        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            
            # input size: 1 x ( batch * beam )
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
            
            """  
                Inefficient decoding implementation
                We re-compute all states for every time step
                A better buffering algorithm will be implemented
            """
           
            decoder_input = Variable(input)
            
            # require batch first for everything
            outs = dict()
            attns = dict()
            
            for i in range(self.n_models):
                #~ decoder_hidden, coverage, buffers[i] = self.models[i].decoder.step(decoder_input.transpose(0,1) , contexts[i].transpose(0, 1), src.transpose(0, 1), buffer=buffers[i])
                decoder_hidden, coverage = self.models[i].decoder.step(decoder_input.clone(), decoder_states[i])
                
                # take the last decoder state
                decoder_hidden = decoder_hidden.squeeze(1)
                attns[i] = coverage[:, -1, :].squeeze(1) # batch * beam x src_len
                
                # batch * beam x vocab_size 
                outs[i] = self.models[i].generator(decoder_hidden)
            
            out = self._combineOutputs(outs)
            attn = self._combineAttention(attns)
                
            wordLk = out.view(beamSize, remainingSents, -1) \
                        .transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1) \
                       .transpose(0, 1).contiguous()
                       
            active = []
            
            for b in range(batchSize):
                if beam[b].done:
                    continue
                
                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]
                    
                for i in range(self.n_models):
                    decoder_states[i]._update_beam(beam, b, remainingSents, idx)
               
                
            if not active:
                break
                
            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}
            
            
            for i in range(self.n_models):
                decoder_states[i]._prune_complete_beam(activeIdx, remainingSents)
               
            
            
            remainingSents = len(active)
            
        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best
        allLengths = []

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn, length = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            allLengths += [length]
            valid_attn = srcBatch.data[:1, b].ne(onmt.Constants.PAD) \
                                            .nonzero().squeeze(1)
            attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])
            
        
        torch.set_grad_enabled(True)

        return allHyp, allScores, allAttn, allLengths, goldScores, goldWords

    def translate(self, srcBatch, goldBatch, lr):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = self.to_variable(dataset.next()[0])
        src, tgt = batch
        batchSize = self._getBatchSize(src)

        #  (2) translate
        pred, predScore, attn, predLength, goldScore, goldWords = self.translateBatch(srcBatch, src, tgt, lr)
        

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, predLength, goldScore, goldWords


