from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
import random
import numpy as np

from grad_utils import plot_grad_flow
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.ModelConstructor import init_model_parameters
from onmt.train_utils.loss import HLoss
import translate


separator = '-' * 70


class BaseTrainer(object):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):

        self.model = model
        self.trainData = trainData
        self.validData = validData
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1)

        self.loss_function = loss_function

        self.start_time = 0

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def to_variable(self, data):

        for i, t in enumerate(data):
            if self.cuda:
                data[i] = Variable(data[i].cuda())
            else:
                data[i] = Variable(data[i])

        return data

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        super().__init__(model, loss_function, trainData, validData, dataset, opt)
        self.optim = onmt.Optim(opt)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        self.optim.set_parameters(self.model.parameters())

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):

        opt, dataset = self.opt, self.dataset
        model = self.model

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'iteration': iteration,
            'batchOrder': batchOrder,
            'optim': optim_state_dict
        }

        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        return file_name

    def eval(self, data):
        epoch_loss = 0
        total_words = 0
        nSamples = len(data)
        num_accumulated_sents = 0

        batch_order = data.create_order()
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(nSamples):

                samples = data.next()

                batch = self.to_variable(samples[0])

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                if int(batch[2][0].cpu().numpy()) == 0:
                    self.model.decoder.set_active(0)
                else:
                    self.model.decoder.set_active(1)

                outputs = self.model(batch)
                targets = batch[1][1:]

                batch_size = targets.size(1)

                loss_reconstruction, _ = self.loss_function(outputs, targets, generator=self.model.generator,
                                                            backward=False)

                loss_reconstruction = loss_reconstruction.data.cpu().numpy()

                epoch_loss += loss_reconstruction
                total_words += targets.data.ne(onmt.Constants.PAD).sum().item()
                num_accumulated_sents += batch_size

        self.model.train()
        return epoch_loss / total_words


    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):
        """

        :param epoch:
        :param p: Percent of total training completed
        :param resume:
        :param batchOrder:
        :param iteration:
        :return:
        """


        opt = self.opt
        trainData = self.trainData

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.

        if resume:
            trainData.batchOrder = batchOrder
            trainData.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batchOrder = trainData.create_order()
            iteration = 0
        # if batchOrder is not None:
        # batchOrder = trainData.create_order()
        # else:
        # trainData.batchOrder = batchOrder

        # if iteration is not None and iteration > -1:
        # trainData.set_index(iteration)
        # print("Resuming from iteration: %d" % iteration)

        epoch_loss = 0
        epoch_loss_reconstruction = 0
        total_words = 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset

        num_accumulated_sents = 0
        num_total_sents = 0

        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)

            samples = trainData.next(curriculum=curriculum)

            batch = self.to_variable(samples[0])

            oom = False

            if int(batch[2][0].cpu().numpy()) == 0:
                self.model.decoder.set_active(0)
            else:
                self.model.decoder.set_active(1)

            targets = batch[1][1:]

            batch_size = targets.size(1)

            try:
                outputs = self.model(batch)

                tgt_mask = targets.data.ne(onmt.Constants.PAD)
                tgt_size = tgt_mask.sum()

                tgt_mask = torch.autograd.Variable(tgt_mask)
                # ~ tgt_mask = None
                normalizer = 1

                if self.opt.normalize_gradient:
                    normalizer = tgt_size

                loss_reconstruction, _ = self.loss_function(outputs, targets, generator=self.model.generator,
                                                                       backward=False, mask=tgt_mask,
                                                                       normalizer=normalizer)
                loss_reconstruction.backward()

                # update parameters immediately
                self.optim.step(grad_denom=1)
                self.model.zero_grad()


            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e

            if not oom:
                num_accumulated_sents += batch_size
                num_total_sents += batch_size

                src_size = batch[0].data.ne(onmt.Constants.PAD).sum().item()
                tgt_size = targets.data.ne(onmt.Constants.PAD).sum().item()

                # important: convert to numpy (or set requires_grad to False), otherwise the statistics variables are tensors and contain
                # the history of the whole epoch, leading to a memory overflow
                loss_reconstruction = loss_reconstruction.data.cpu().numpy()

                num_words = tgt_size
                report_loss += loss_reconstruction
                report_tgt_words += num_words
                report_src_words += src_size
                epoch_loss += loss_reconstruction
                epoch_loss_reconstruction += loss_reconstruction
                total_words += num_words

                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i + 1, len(trainData),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words / (time.time() - start),
                           report_tgt_words / (time.time() - start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()

        return epoch_loss_reconstruction / total_words

    def run(self, save_file=None):

        opt = self.opt
        model = self.model
        dataset = self.dataset
        optim = self.optim

        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)

        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])

            if opt.reset_optim == False:
                self.optim.load_state_dict(checkpoint['optim'])
                batchOrder = checkpoint['batchOrder']
                iteration = checkpoint['iteration'] + 1
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume = True
            else:
                batchOrder = None
                iteration = 0
                resume = False

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batchOrder = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False

        reconstr = self.eval(self.validData)
        reconstr_ppl = math.exp(min(reconstr, 100))
        print('{}\nValid reconstruction ppl: {}\n{}\n'.format(separator, reconstr_ppl, separator))

        self.start_time = time.time()

        best_train_loss = (None, None, None, None)
        best_val_loss = (None, None, None, None)

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            reconstr = self.train_epoch(epoch, resume=resume, batchOrder=batchOrder, iteration=iteration)
            reconstr_ppl = math.exp(min(reconstr, 100))
            print('{}\nTrain reconstruction ppl: {}\n{}\n'.format(separator, reconstr_ppl, separator))

            if best_train_loss[0] is None or best_train_loss[0] > reconstr_ppl:
                best_train_loss = (reconstr_ppl,)

            #  (2) evaluate on the validation set
            reconstr = self.eval(self.validData)
            reconstr_ppl = math.exp(min(reconstr, 100))
            print('{}\nValid reconstruction ppl: {}\n{}\n'.format(separator, reconstr_ppl, separator))

            if best_val_loss[0] is None or best_val_loss[0] > reconstr_ppl:
                best_val_loss = (reconstr_ppl,)

            if epoch % opt.save_every_epoch == 0:
                model_fname = self.save(epoch, reconstr_ppl)

                # print translation to make it easier to see if the model is good
                if opt.translate_src is not None:
                    separator2 = '.' * 70
                    print('\n' + separator2)
                    for target_style in range(1, 3):
                        print('\nSample translation with style {}:\n'.format(target_style))
                        translate.translate(['-src', opt.translate_src, '-model', model_fname, '-target_style', str(target_style),
                                             '-max_sent_length', '30', '-output', 'stdout', '-remove_bpe', '-diff'])
                    print('\n' + separator2 + '\n')

            batchOrder = None
            iteration = None
            resume = False


        print('\n\nBEST RESULTS:\n')

        print('{}\nTrain reconstruction ppl: {}\n{}\n'.format(separator, best_train_loss[0], separator))
        print('{}\nValid reconstruction ppl: {}\n{}\n'.format(separator, best_val_loss[0], separator))



