from __future__ import division

import math
import torch
from torch import tensor, Tensor
from torch.autograd import Variable

import onmt

# TODO properly implement this. This turns source into style 1 and target into style 2. Ideally, there shouldn't be an adapter for this,
# so no source/target setup in the other code files but instead style 1, style 2
class NonParallelDataset(object):
    '''
    batchSize is now changed to have word semantic (probably better)
    '''
    def __init__(self, srcData, tgtData, dict, batchSize, gpus,
                 volatile=False, data_type="text", balance=False, max_seq_num=128,
                 multiplier=8, pad_count=True):
        self.dict = dict

        n = len(srcData)
        print('WARNING: dividing src/tgt data in half, only for debugging purposes')
        style1 = srcData[: n // 2]
        style2 = tgtData[: n // 2]

        concatSrc, targets = self.concat(style1, style2)
        self.n_style1 = len(style1)
        self.n = len(concatSrc)

        self.src = concatSrc
        self._type = data_type
        self.tgt = targets
        assert(len(self.src) == len(self.tgt))
        self.cuda = (len(gpus) > 0)
        self.fullSize = len(self.src)
        self.n_gpu = len(gpus)

        self.batchSize = batchSize
        #~ self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

        self.balance = balance
        self.max_seq_num = max_seq_num

        self.multiplier = multiplier


        # if self.balance:
        self.allocateBatch()
        self.cur_index = 0
        self.batchOrder = None
        # else:
            # self.numBatches = math.ceil(len(self.src)/batchSize)


    def concat(self, style1, style2):
        return style1 + style2, [[0, 1]] * len(style1) + [[1, 0]] * len(style2)


    #~ # This function allocates the mini-batches (grouping sentences with the same size)
    def allocateBatch(self):
        self.numBatches_style1 = 0

        # The sentence pairs are sorted by source already (cool)
        self.batches = []

        cur_batch = [0]
        cur_batch_size = self.src[0].size(0)

        i = 1
        while i < self.fullSize:
        #~ for i in range(1, self.fullSize):

            sentence_length = self.src[i].size(0)

            oversized = False

            if ( cur_batch_size + sentence_length > self.batchSize ) or len(cur_batch) == self.max_seq_num:
                oversized = True
            # if the current length makes the batch exceeds
            # or it is the first one of style2 (don't mix styles!)
            # the we create a new batch
            if oversized or i == self.n_style1:
                self.batches.append(cur_batch) # add this batch into the batch list
                cur_batch = [] # reset the current batch
                cur_batch_size = 0

            if i == self.n_style1:
                self.numBatches_style1 = len(self.batches)


            cur_batch.append(i)
            cur_batch_size += sentence_length

            i = i + 1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.numBatches = len(self.batches)
        self.numBatches_style2 = self.numBatches - self.numBatches_style1

    def _batchify(self, data, align_right=False,
                  include_lengths=False, dtype="text"):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def to_target(self, src):
        target = []
        for sent in src:
            sos = tensor([self.dict.labelToIdx['<s>']])
            eos = tensor([self.dict.labelToIdx['</s>']])
            target.append(torch.cat((sos, sent, eos)))

        return target

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        batch = self.batches[index]
        srcData = [self.src[i] for i in batch]
        srcBatch, lengths = self._batchify(
            srcData,
            align_right=False, include_lengths=True, dtype=self._type)

        tgtSeqData = self.to_target(srcData)
        tgtSeqBatch, _ = self._batchify(
            tgtSeqData,
            align_right=False, include_lengths=True, dtype=self._type)


        if self.tgt:
            tgtBatch = [self.tgt[i] for i in batch]
        else:
            tgtBatch = None


        def wrap(b, dtype="text"):
            if b is None:
                return b
            #~ print (b)
            #~ b = torch.stack(b, 0)
            b = b.t().contiguous()

            return b

        srctensor = wrap(srcBatch, self._type)
        tgtSeqtensor = wrap(tgtSeqBatch, self._type)
        tgttensor = Tensor(tgtBatch)

        return [srctensor, tgtSeqtensor, tgttensor]


    def __len__(self):
        return self.numBatches

    def create_order(self):
        self.batchOrder = torch.arange(self.numBatches).long()
        # alternate between style1 and style2
        count = 0
        for i in range(max(self.numBatches_style1, self.numBatches_style2)):
            if i < self.numBatches_style1:
                self.batchOrder[count] = i
                count += 1
            if i < self.numBatches_style2:
                self.batchOrder[count] = i + self.numBatches_style1
                count += 1

        self.cur_index = 0

        return self.batchOrder

    def next(self, curriculum=False, reset=True, split_sizes=1):

         # reset iterator if reach data size limit
        if self.cur_index >= self.numBatches:

            if reset:
                self.cur_index = 0
            else: return None

        if curriculum or self.batchOrder is None:
            batch_index = self.cur_index
        else:
            batch_index = self.batchOrder[self.cur_index]

        batch = self[batch_index]

        # move the iterator one step
        self.cur_index += 1

        #split that batch to number of gpus

        samples = []
        split = self.n_gpu if self.n_gpu > 0 else 1
        split_size = int(math.ceil(batch[0].size(1) / split))
        # maybe we need a more smart splitting function ?

        batch_split = zip(batch[0].split(split_size, dim=1),
                          batch[1].split(split_size, dim=1),
                          batch[2].split(split_size, dim=0))

        batch_split = [ [b[0], b[1], b[2]] for i, b in enumerate(batch_split) ]

        return batch_split


    def set_index(self, iteration):

        assert iteration >= 0 and iteration < self.numBatches
        self.cur_index = iteration

# class AugmentedDataset(object):
